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
DEBUG_PORT            = int(os.getenv("DEBUG_PORT",              "8083"))

# ── Debug server state ────────────────────────────────────────────────────────

import json as _json

_jpeg_lock   = threading.Lock()
_latest_jpeg = b""

_state_lock   = threading.Lock()
_latest_state: dict = {}

_counters      = {"frames_processed": 0, "gestures_fired": 0, "face_hits": 0, "face_misses": 0}
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


def _update_state(state: dict):
    with _state_lock:
        global _latest_state
        _latest_state = state


_DEBUG_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MagicMirror3 Camera Debug</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: monospace; display: flex;
         flex-direction: column; align-items: center; padding: 16px; gap: 12px; }
  h1 { font-size: 1.1rem; color: #aaa; letter-spacing: 2px; }
  #frame-wrap { position: relative; }
  #frame { display: block; max-width: 100%; border: 2px solid #333; border-radius: 4px; }
  #panel { width: 100%; max-width: 700px; display: grid;
           grid-template-columns: 1fr 1fr; gap: 8px; }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px;
          padding: 10px 14px; }
  .card h2 { font-size: 0.7rem; color: #666; text-transform: uppercase;
             letter-spacing: 1px; margin-bottom: 6px; }
  .row { display: flex; justify-content: space-between; padding: 2px 0; font-size: 0.85rem; }
  .val { color: #7df; font-weight: bold; }
  .val.green  { color: #4f4; }
  .val.yellow { color: #ff4; }
  .val.grey   { color: #888; }
  .val.red    { color: #f44; }
  #dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #f44; margin-right: 6px; }
  #dot.live { background: #4f4; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  #err { color: #f84; font-size: 0.75rem; display: none; margin-top: 4px; }
</style>
</head>
<body>
<h1><span id="dot"></span>MagicMirror3 — Camera Debug</h1>
<img id="frame" src="/camera-frame.jpg" width="640" height="480" alt="camera">
<div id="panel">
  <div class="card">
    <h2>Pipeline</h2>
    <div class="row"><span>FPS</span><span class="val" id="fps">—</span></div>
    <div class="row"><span>Presence</span><span class="val" id="presence">—</span></div>
    <div class="row"><span>Frames</span><span class="val" id="frames">—</span></div>
  </div>
  <div class="card">
    <h2>Gesture</h2>
    <div class="row"><span>State</span><span class="val" id="g-state">—</span></div>
    <div class="row"><span>Current</span><span class="val" id="g-current">—</span></div>
    <div class="row"><span>Sent</span><span class="val" id="g-sent">—</span></div>
    <div class="row"><span>Progress</span><span class="val" id="g-progress">—</span></div>
  </div>
  <div class="card">
    <h2>Face</h2>
    <div class="row"><span>Profile</span><span class="val" id="f-profile">—</span></div>
    <div class="row"><span>Confidence</span><span class="val" id="f-conf">—</span></div>
    <div class="row"><span>Hits / Misses</span><span class="val" id="f-hm">—</span></div>
  </div>
  <div class="card">
    <h2>Config</h2>
    <div class="row"><span>AI Scale</span><span class="val" id="ai-scale">—</span></div>
    <div class="row"><span>Flip</span><span class="val" id="flip">—</span></div>
    <div class="row"><span>Gestures fired</span><span class="val" id="g-fired">—</span></div>
  </div>
</div>
<div id="err">Connection lost — retrying…</div>
<script>
const $ = id => document.getElementById(id);
const dot = $('dot'), err = $('err');

// ── Live frame: reload img src with cache-buster every 120ms ──────────────
const img = $('frame');
let frameOk = true;
setInterval(() => {
  const next = new Image();
  next.onload  = () => { img.src = next.src; frameOk = true;
                         dot.className = 'live'; err.style.display='none'; };
  next.onerror = () => { frameOk = false;
                         dot.className = ''; err.style.display='block'; };
  next.src = '/camera-frame.jpg?t=' + Date.now();
}, 120);

// ── State polling every 350ms ─────────────────────────────────────────────
function colorClass(val) {
  if (!val || val === 'none' || val === 'away' || val === 'IDLE') return 'grey';
  if (val === 'present' || val === 'LOCKED') return 'green';
  if (val === 'BUILDING' || val === 'GRACE')  return 'yellow';
  return '';
}
function set(id, text, cls) {
  const el = $(id);
  el.textContent = text ?? '—';
  el.className = 'val ' + (cls || colorClass(String(text)));
}
setInterval(() => {
  fetch('/state').then(r => r.json()).then(s => {
    set('fps',        (s.fps||0).toFixed(1) + ' fps', s.fps > 10 ? 'green' : 'yellow');
    set('presence',   s.presence);
    set('frames',     s.frames_processed);
    set('g-state',    s.gesture_state);
    set('g-current',  s.gesture_current || 'none');
    set('g-sent',     s.gesture_sent    || 'none');
    set('g-progress', s.hold_progress != null ? (s.hold_progress*100).toFixed(0)+'%' : '—', '');
    set('f-profile',  s.face_profile  || 'none');
    set('f-conf',     s.face_confidence != null ? (s.face_confidence*100).toFixed(1)+'%' : '—', '');
    set('f-hm',       (s.face_hits||0) + ' / ' + (s.face_misses||0), '');
    set('ai-scale',   s.ai_scale, '');
    set('flip',       s.mirror_flip ? 'on' : 'off', s.mirror_flip ? 'yellow' : 'grey');
    set('g-fired',    s.gestures_fired, '');
  }).catch(() => {});
}, 350);
</script>
</body>
</html>
"""


class _DebugHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/camera-view":
            self._send(200, "text/html; charset=utf-8", _DEBUG_HTML.encode())

        elif self.path.startswith("/camera-frame.jpg"):
            with _jpeg_lock:
                data = _latest_jpeg
            if data:
                self._send(200, "image/jpeg", data)
            else:
                self.send_response(503); self.end_headers()

        elif self.path == "/state":
            with _state_lock:
                s = dict(_latest_state)
            with _counters_lock:
                s.update(_counters)
            s["ai_scale"]    = AI_SCALE
            s["mirror_flip"] = MIRROR_FLIP
            self._send(200, "application/json", _json.dumps(s).encode())

        elif self.path == "/metrics":
            with _counters_lock:
                lines = [f"{k} {v}" for k, v in _counters.items()]
            self._send(200, "text/plain", "\n".join(lines).encode())

        else:
            self.send_response(404); self.end_headers()


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
    p.add_argument("--bridge-port", type=int, default=int(os.getenv("BRIDGE_PORT", "8082")))
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
                    raw_faces = face_future.result()
                    # Scale bbox coords from AI-downsampled space back to full frame
                    if AI_SCALE != 1.0 and raw_faces:
                        inv = 1.0 / AI_SCALE
                        cached_faces = [
                            {**f, "location": tuple(int(c * inv) for c in f["location"])}
                            for f in raw_faces
                        ]
                    else:
                        cached_faces = raw_faces
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

            # ── Annotate + push debug JPEG + state ───────────────────────
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
            _update_state({
                "fps":              round(measured_fps, 1),
                "presence":         "present" if presence else "away",
                "gesture_state":    gesture_state,
                "gesture_current":  current_gesture_label,
                "gesture_sent":     sent_gesture_state,
                "hold_progress":    round(hold_progress, 2),
                "face_profile":     best_profile,
                "face_confidence":  round(best_confidence, 3),
            })

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
