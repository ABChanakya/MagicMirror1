"""
main.py — MagicMirror3 camera pipeline

Captures frames from a webcam, runs:
  1. Gesture detection  (MediaPipe, every frame)
  2. Face recognition   (InsightFace, async every N frames)
  3. Presence detection (derived from face results)

Sends events to MMM-CameraBridge via HTTP (state-change only — no flooding).
Serves a live JPEG debug view and Prometheus-style metrics on DEBUG_PORT.

Usage:
    python3 main.py [--device /dev/video0] [--bridge-port 8082] [--debug]
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from face_recognizer import FaceRecognizer
from gesture_detector import GestureDetector
from http_sender import HttpSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("camera.main")

# ── Config ───────────────────────────────────────────────────────────────────

FACE_STABLE_SECONDS   = float(os.getenv("FACE_STABLE_SECONDS",   "3.0"))
FACE_TOLERANCE        = float(os.getenv("FACE_TOLERANCE",         "0.45"))
PRESENCE_AWAY_AFTER   = float(os.getenv("PRESENCE_AWAY_AFTER",   "10.0"))
CAMERA_FPS_LIMIT      = int(os.getenv("CAMERA_FPS",              "15"))
CAMERA_WIDTH          = int(os.getenv("CAMERA_WIDTH",            "640"))
CAMERA_HEIGHT         = int(os.getenv("CAMERA_HEIGHT",           "480"))
AI_SCALE              = float(os.getenv("AI_SCALE",              "0.5"))
FACE_DETECT_EVERY     = int(os.getenv("FACE_DETECT_EVERY",       "3"))
HAND_FLICKER_TOLERANCE = int(os.getenv("HAND_FLICKER_FRAMES",    "3"))
HAND_GONE_GRACE       = float(os.getenv("HAND_GONE_GRACE",       "0.5"))
MIRROR_FLIP           = os.getenv("MIRROR_FLIP", "false").strip().lower() in ("1", "true", "yes")
DEBUG_PORT            = int(os.getenv("DEBUG_PORT",              "8082"))

# ── Debug JPEG server ─────────────────────────────────────────────────────────

_jpeg_lock   = threading.Lock()
_latest_jpeg = b""

# Prometheus-style counters
_counters = {"frames_processed": 0, "gestures_fired": 0, "face_hits": 0, "face_misses": 0}
_counters_lock = threading.Lock()


def _inc(name: str, n: int = 1):
    with _counters_lock:
        _counters[name] += n


def _update_jpeg(frame_bgr: np.ndarray):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        with _jpeg_lock:
            global _latest_jpeg
            _latest_jpeg = buf.tobytes()


class _DebugHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress per-request access log

    def do_GET(self):
        if self.path == "/camera-view":
            with _jpeg_lock:
                data = _latest_jpeg
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(503)
                self.end_headers()
        elif self.path == "/metrics":
            with _counters_lock:
                lines = [f"{k} {v}" for k, v in _counters.items()]
            body = "\n".join(lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def _start_debug_server(port: int):
    try:
        server = HTTPServer(("0.0.0.0", port), _DebugHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("Debug server: http://0.0.0.0:%d/camera-view", port)
    except Exception as e:
        logger.warning("Could not start debug server on port %d: %s", port, e)


# ── Annotation ────────────────────────────────────────────────────────────────

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]


def annotate_frame(
    frame_bgr: np.ndarray,
    disp_landmarks,
    hold_progress: float,
    current_gesture_state: str | None,
    faces: list,
    sent_gesture_state: str | None,
    fps: float,
) -> np.ndarray:
    """Draw all overlays onto a copy of frame_bgr and return it."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # ── Hand skeleton + bounding box ─────────────────────────────────────
    if disp_landmarks is not None:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in disp_landmarks]

        # Skeleton lines
        for a, b in _HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (255, 200, 0), 2)
        # Landmark dots
        for pt in pts:
            cv2.circle(out, pt, 4, (0, 200, 255), cv2.FILLED)

        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 12
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w, max(xs) + pad)
        y2 = min(h, max(ys) + pad)

        # Box color by state
        if sent_gesture_state is not None and hold_progress >= 1.0:
            box_color = (255, 255, 0)    # cyan — LOCKED
        elif hold_progress > 0:
            box_color = (0, 255, 120)    # green — BUILDING
        else:
            box_color = (200, 200, 200)  # grey  — IDLE

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        # Label above box — current_gesture_state is set by main.py each frame
        if current_gesture_state:
            label_y = max(20, y1 - 10)
            cv2.putText(out, current_gesture_state, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

        # Progress bar (only while building, not yet locked)
        if 0.0 < hold_progress < 1.0:
            cv2.rectangle(out, (x1, y2 + 6), (x2, y2 + 12), (60, 60, 60), cv2.FILLED)
            fill_x = x1 + int((x2 - x1) * hold_progress)
            cv2.rectangle(out, (x1, y2 + 6), (fill_x, y2 + 12), (0, 255, 120), cv2.FILLED)

    # ── Face boxes ────────────────────────────────────────────────────────
    for face in faces:
        top, right, bottom, left = face["location"]
        profile    = face["profile"]
        confidence = face["confidence"]
        known      = profile != "unknown"
        color      = (0, 220, 0) if known else (0, 80, 220)

        cv2.rectangle(out, (left, top), (right, bottom), color, 2)
        face_label = f"{profile}  {confidence:.0%}" if known else "unbekannt"
        lw, lh = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.rectangle(out, (left, top - lh - 8), (left + lw + 6, top), color, cv2.FILLED)
        cv2.putText(out, face_label, (left + 3, top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # ── FPS overlay ───────────────────────────────────────────────────────
    cv2.putText(out, f"FPS:{fps:.0f}", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera(device: str) -> cv2.VideoCapture:
    logger.info("Opening camera: %s  (%dx%d @ %d fps)", device, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS_LIMIT)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.error("Cannot open camera %s", device)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_LIMIT)
    logger.info("Camera opened (%.0fx%.0f @ %.0f fps)",
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS))
    return cap


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MagicMirror3 camera pipeline")
    p.add_argument("--device",      default=os.getenv("CAMERA_DEVICE", "/dev/video0"))
    p.add_argument("--bridge-port", type=int, default=int(os.getenv("BRIDGE_PORT", str(DEBUG_PORT))))
    p.add_argument("--debug",       action="store_true")
    return p.parse_args()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Gesture state machine ─────────────────────────────────────────────
    #
    #   IDLE ──[hand+progress>0]──► BUILDING ──[progress==1.0]──► LOCKED
    #    ▲                              │                             │
    #    │                    [count changes]                 [hand disappears]
    #    │                              ▼                             ▼
    #    └──────────[grace expires]── IDLE          GRACE ◄──────────┘
    #                                                 │
    #                               [hand returns within HAND_GONE_GRACE]
    #                                                 │
    #                                              LOCKED

    IDLE, BUILDING, LOCKED, GRACE = "IDLE", "BUILDING", "LOCKED", "GRACE"

    sender   = HttpSender(port=args.bridge_port)
    face_rec = FaceRecognizer(tolerance=FACE_TOLERANCE)
    gestures = GestureDetector()

    _start_debug_server(DEBUG_PORT)

    cap               = open_camera(args.device)
    shutdown_flag     = threading.Event()
    executor          = ThreadPoolExecutor(max_workers=1)

    def _shutdown(*_):
        logger.info("Shutdown signal received.")
        shutdown_flag.set()

    signal.signal(signal.SIGTERM, _shutdown)

    # ── Per-frame state ───────────────────────────────────────────────────
    presence           = False
    last_seen_time     = 0.0
    last_profile       = None
    profile_since      = 0.0
    face_confirmed     = False

    frame_interval     = 1.0 / CAMERA_FPS_LIMIT
    last_frame_time    = 0.0
    fps_buf: deque     = deque(maxlen=30)
    prev_frame_ts      = time.monotonic()

    # Face async
    face_frame_counter = 0
    cached_faces: list = []
    face_future        = None

    # Sticky display
    hand_lost_frames   = 0
    disp_landmarks     = None
    hand_gone_since: float | None = None

    # Gesture state machine
    gesture_state      = IDLE
    sent_gesture_state = None
    current_gesture_label: str | None = None

    # Read-failure counter for auto-reconnect
    read_fail_count    = 0

    logger.info("Camera pipeline running. Press Ctrl+C or send SIGTERM to stop.")

    try:
        while not shutdown_flag.is_set():
            now = time.monotonic()

            # Throttle to target FPS
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                read_fail_count += 1
                logger.warning("Frame read failed (%d consecutive)", read_fail_count)
                if read_fail_count >= 10:
                    logger.warning("Re-opening camera after 10 failures.")
                    cap.release()
                    time.sleep(1.0)
                    cap = open_camera(args.device)
                    read_fail_count = 0
                else:
                    time.sleep(0.1)
                continue
            read_fail_count = 0

            frame_ts       = time.monotonic()
            fps_buf.append(1.0 / max(1e-6, frame_ts - prev_frame_ts))
            prev_frame_ts  = frame_ts
            measured_fps   = sum(fps_buf) / len(fps_buf)
            _inc("frames_processed")

            # Optional mirror flip before any AI
            if MIRROR_FLIP:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # AI downsampled copy
            if AI_SCALE != 1.0:
                small = cv2.resize(rgb, (0, 0), fx=AI_SCALE, fy=AI_SCALE,
                                   interpolation=cv2.INTER_AREA)
            else:
                small = rgb

            # ── Gesture detection ─────────────────────────────────────────
            gestures.process_frame(small)

            raw_landmarks    = gestures.last_landmarks
            raw_finger_state = gestures.last_finger_state   # int | None
            hold_progress    = gestures.hold_progress

            # Flicker buffer for display
            if raw_landmarks is not None:
                hand_lost_frames = 0
                disp_landmarks   = raw_landmarks
                hand_gone_since  = None
            else:
                hand_lost_frames += 1
                if hand_lost_frames > HAND_FLICKER_TOLERANCE:
                    disp_landmarks = None
                if hand_gone_since is None and hand_lost_frames > HAND_FLICKER_TOLERANCE:
                    hand_gone_since = now

            # Display label = live finger count
            if raw_finger_state is not None and raw_finger_state > 0:
                current_gesture_label = f"fingers_{raw_finger_state}"
            elif raw_finger_state == 0:
                current_gesture_label = "fist"
            elif disp_landmarks is None:
                current_gesture_label = None

            # ── Gesture state machine ─────────────────────────────────────
            hand_present = disp_landmarks is not None

            if gesture_state == IDLE:
                if hand_present and hold_progress > 0:
                    gesture_state = BUILDING
                    logger.debug("Gesture: IDLE → BUILDING")

            elif gesture_state == BUILDING:
                if not hand_present:
                    gesture_state = IDLE
                    hold_progress = 0.0
                    logger.debug("Gesture: BUILDING → IDLE (hand lost)")
                elif raw_finger_state != (
                    int(sent_gesture_state.split("_")[1])
                    if sent_gesture_state and sent_gesture_state.startswith("fingers_")
                    else None
                ) and hold_progress == 0.0:
                    # count changed mid-build — restart
                    gesture_state = IDLE
                    logger.debug("Gesture: BUILDING → IDLE (count changed)")
                elif hold_progress >= 1.0:
                    gesture_state = LOCKED
                    new_state = current_gesture_label
                    if new_state != sent_gesture_state:
                        sent_gesture_state = new_state
                        logger.info("Gesture LOCKED: %s", new_state)
                        if new_state:
                            sender.send_gesture(new_state)
                            _inc("gestures_fired")

            elif gesture_state == LOCKED:
                if not hand_present:
                    gesture_state = GRACE
                    logger.debug("Gesture: LOCKED → GRACE")
                elif raw_finger_state is not None and current_gesture_label != sent_gesture_state:
                    # Pose changed while still showing hand
                    gesture_state = BUILDING
                    logger.debug("Gesture: LOCKED → BUILDING (pose changed)")

            elif gesture_state == GRACE:
                if hand_present:
                    gesture_state = LOCKED
                    logger.debug("Gesture: GRACE → LOCKED (hand returned)")
                elif hand_gone_since is not None and (now - hand_gone_since) >= HAND_GONE_GRACE:
                    gesture_state      = IDLE
                    sent_gesture_state = None
                    hold_progress      = 0.0
                    logger.info("Gesture: GRACE → IDLE (grace expired)")
                    sender.send_gesture(None)

            # ── Face / presence detection (async, every N frames) ─────────
            face_frame_counter += 1
            if face_frame_counter >= FACE_DETECT_EVERY:
                face_frame_counter = 0
                if face_future is None or face_future.done():
                    face_future = executor.submit(face_rec.identify, small)

            if face_future is not None and face_future.done():
                try:
                    cached_faces = face_future.result()
                except Exception as e:
                    logger.debug("Face future error: %s", e)
                    cached_faces = []
                face_future = None

            faces = cached_faces
            best_profile    = "unknown"
            best_confidence = 0.0

            if faces:
                last_seen_time = now
                if not presence:
                    presence = True
                    logger.info("Presence: present")
                    sender.send_presence("present")

                best = max(faces, key=lambda f: f["confidence"])
                best_profile    = best["profile"]
                best_confidence = float(best["confidence"])

                if best_profile != last_profile:
                    last_profile   = best_profile
                    profile_since  = now
                    face_confirmed = False

                stable = (now - profile_since) >= FACE_STABLE_SECONDS
                if stable and not face_confirmed and best_profile != "unknown":
                    logger.info("Face: %s (%.2f)", best_profile, best_confidence)
                    sender.send_face(best_profile, best_confidence)
                    face_confirmed = True
                    _inc("face_hits")
            else:
                _inc("face_misses")
                away_for = now - last_seen_time
                if presence and away_for >= PRESENCE_AWAY_AFTER:
                    presence       = False
                    last_profile   = None
                    face_confirmed = False
                    logger.info("Presence: away (%.1fs)", away_for)
                    sender.send_presence("away")

            # ── Annotate + push debug JPEG ────────────────────────────────
            annotated = annotate_frame(
                frame,
                disp_landmarks,
                hold_progress,
                current_gesture_label,
                faces,
                sent_gesture_state,
                measured_fps,
            )
            _update_jpeg(annotated)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        shutdown_flag.set()
        cap.release()
        gestures.close()
        executor.shutdown(wait=False)
        logger.info("Camera pipeline stopped.")


if __name__ == "__main__":
    main()
