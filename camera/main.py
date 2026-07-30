"""
main.py — MagicMirror3 camera pipeline (Phase 1: RTX 4090 rewrite)

Swipe detection (left/right/up/down) + hand Y/X position tracking.
No face recognition or presence detection in Phase 1.

Usage:
    python3 main.py [--device /dev/video0] [--bridge-port 8082] [--debug]
"""

import argparse
import json as _json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from gesture_detector_rulebased import GestureDetectorRuleBased as GestureDetector
from http_sender import HttpSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("camera.main")

# Config
CAMERA_FPS_LIMIT = int(os.getenv("CAMERA_FPS", "30"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
MIRROR_FLIP = os.getenv("MIRROR_FLIP", "false").strip().lower() in ("1", "true", "yes")
DEBUG_PORT = int(os.getenv("DEBUG_PORT", "8083"))

# Debug state
_jpeg_lock = threading.Lock()
_latest_jpeg = b""

_state_lock = threading.Lock()
_latest_state: dict = {}

_counters = {"frames_processed": 0, "swipes_sent": 0}
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
<title>MagicMirror3 Camera Debug (Phase 1)</title>
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
<h1><span id="dot"></span>MagicMirror3 — Camera Debug (Phase 1)</h1>
<img id="frame" src="/camera-frame.jpg" width="640" height="480" alt="camera">
<div id="panel">
  <div class="card">
    <h2>Pipeline</h2>
    <div class="row"><span>FPS</span><span class="val" id="fps">—</span></div>
    <div class="row"><span>Hand Present</span><span class="val" id="hand">—</span></div>
    <div class="row"><span>Frames</span><span class="val" id="frames">—</span></div>
  </div>
  <div class="card">
    <h2>Hand Position</h2>
    <div class="row"><span>X</span><span class="val" id="hand-x">—</span></div>
    <div class="row"><span>Y</span><span class="val" id="hand-y">—</span></div>
    <div class="row"><span>Swipes Sent</span><span class="val" id="swipes">—</span></div>
  </div>
</div>
<div id="err">Connection lost — retrying…</div>
<script>
const $ = id => document.getElementById(id);
const dot = $('dot'), err = $('err');

setInterval(() => {
  const next = new Image();
  next.onload  = () => { img.src = next.src; dot.className = 'live'; err.style.display='none'; };
  next.onerror = () => { dot.className = ''; err.style.display='block'; };
  next.src = '/camera-frame.jpg?t=' + Date.now();
}, 120);

const img = $('frame');

setInterval(() => {
  fetch('/state').then(r => r.json()).then(s => {
    $('fps').textContent = (s.fps||0).toFixed(1) + ' fps';
    $('fps').className = 'val ' + (s.fps > 10 ? 'green' : 'yellow');
    $('hand').textContent = s.hand_present ? 'yes' : 'no';
    $('hand').className = 'val ' + (s.hand_present ? 'green' : 'grey');
    $('hand-x').textContent = s.hand_x != null ? s.hand_x.toFixed(3) : '—';
    $('hand-y').textContent = s.hand_y != null ? s.hand_y.toFixed(3) : '—';
    $('frames').textContent = s.frames_processed;
    $('swipes').textContent = s.swipes_sent;
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
                self.send_response(503)
                self.end_headers()

        elif self.path == "/state":
            with _state_lock:
                s = dict(_latest_state)
            with _counters_lock:
                s.update(_counters)
            self._send(200, "application/json", _json.dumps(s).encode())

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


# Hand skeleton for annotation
_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def annotate_frame(frame_bgr: np.ndarray, landmarks, hand_x, hand_y, fps: float) -> np.ndarray:
    """Draw hand skeleton and position onto frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if landmarks is not None:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Skeleton
        for a, b in _HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (255, 200, 0), 2)

        # Landmarks
        for pt in pts:
            cv2.circle(out, pt, 4, (0, 200, 255), cv2.FILLED)

        # Bounding box
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 12
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w, max(xs) + pad)
        y2 = min(h, max(ys) + pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 2)

        # Position label
        if hand_x is not None and hand_y is not None:
            label = f"X:{hand_x:.3f} Y:{hand_y:.3f}"
            cv2.putText(out, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.putText(out, f"FPS:{fps:.0f}", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def open_camera(device: str) -> cv2.VideoCapture:
    logger.info("Opening camera: %s  (%dx%d @ %d fps)", device, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS_LIMIT)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.error("Cannot open camera %s", device)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_LIMIT)
    logger.info("Camera opened (%.0fx%.0f @ %.0f fps)",
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS))
    return cap


def parse_args():
    p = argparse.ArgumentParser(description="MagicMirror3 camera pipeline (Phase 1)")
    p.add_argument("--device", default=os.getenv("CAMERA_DEVICE", "/dev/video0"))
    p.add_argument("--bridge-port", type=int, default=int(os.getenv("BRIDGE_PORT", "8082")))
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    sender = HttpSender(port=args.bridge_port)
    gestures = GestureDetector()
    _start_debug_server(DEBUG_PORT)

    cap = open_camera(args.device)
    shutdown_flag = threading.Event()

    def _shutdown(*_):
        logger.info("Shutdown signal received.")
        shutdown_flag.set()

    signal.signal(signal.SIGTERM, _shutdown)

    frame_interval = 1.0 / CAMERA_FPS_LIMIT
    last_frame_time = 0.0
    fps_buf: deque = deque(maxlen=30)
    prev_frame_ts = time.monotonic()

    read_fail_count = 0

    logger.info("Camera pipeline running (Phase 1). Press Ctrl+C or send SIGTERM to stop.")

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

            frame_ts = time.monotonic()
            fps_buf.append(1.0 / max(1e-6, frame_ts - prev_frame_ts))
            prev_frame_ts = frame_ts
            measured_fps = sum(fps_buf) / len(fps_buf)
            _inc("frames_processed")

            if MIRROR_FLIP:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame for swipes + hand position
            gesture_event, hand_present = gestures.process_frame(rgb)

            # Send gesture event in background thread (non-blocking)
            if gesture_event:
                _inc("swipes_sent")
                logger.info("Gesture sent: %s", gesture_event["name"])
                threading.Thread(target=sender.send, args=(gesture_event,), daemon=True).start()

            # Annotate and push debug frame
            annotated = annotate_frame(
                frame,
                gestures.last_landmarks,
                gestures.hand_x,
                gestures.hand_y,
                measured_fps,
            )
            _update_jpeg(annotated)
            _update_state({
                "fps": round(measured_fps, 1),
                "hand_present": hand_present,
                "hand_x": gestures.hand_x,
                "hand_y": gestures.hand_y,
            })

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        shutdown_flag.set()
        cap.release()
        gestures.close()
        logger.info("Camera pipeline stopped.")


if __name__ == "__main__":
    main()
