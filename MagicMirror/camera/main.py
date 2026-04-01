"""
main.py — MagicMirror3 camera pipeline

Captures frames from a webcam, runs:
  1. Presence detection (is someone there?)
  2. Face recognition (who is it?)
  3. Gesture detection (what are they doing?)

Sends events to MMM-CameraBridge via HTTP.

Usage:
  python3 main.py [--device /dev/video0] [--bridge-port 8081] [--debug]
"""

import argparse
import logging
import os
import sys
import time

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


# ── Config ──────────────────────────────────────────────────────────────────

FACE_STABLE_SECONDS = float(os.getenv("FACE_STABLE_SECONDS", "3.0"))
FACE_TOLERANCE      = float(os.getenv("FACE_TOLERANCE", "0.45"))
PRESENCE_AWAY_AFTER = float(os.getenv("PRESENCE_AWAY_AFTER", "10.0"))  # seconds no face → away
CAMERA_FPS_LIMIT    = int(os.getenv("CAMERA_FPS", "15"))


def parse_args():
    p = argparse.ArgumentParser(description="MagicMirror3 camera pipeline")
    p.add_argument("--device",      default=os.getenv("CAMERA_DEVICE", "/dev/video0"))
    p.add_argument("--bridge-port", type=int, default=int(os.getenv("BRIDGE_PORT", "8081")))
    p.add_argument("--debug",       action="store_true")
    return p.parse_args()


def open_camera(device: str, fps_limit: int) -> cv2.VideoCapture:
    logger.info("Opening camera: %s", device)
    # Try device index first, then path
    try:
        idx = int(device.replace("/dev/video", ""))
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        logger.error("Cannot open camera %s", device)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps_limit)
    logger.info("Camera opened (%.0fx%.0f @ %d fps)",
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS))
    return cap


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    sender   = HttpSender(port=args.bridge_port)
    face_rec = FaceRecognizer(tolerance=FACE_TOLERANCE)
    gestures = GestureDetector()
    cap      = open_camera(args.device, CAMERA_FPS_LIMIT)

    # State
    presence         = False          # is someone in front of the mirror?
    last_seen_time   = 0.0
    last_profile     = None
    profile_since    = 0.0            # when did we first see this profile?
    face_confirmed   = False          # have we sent this profile to MM yet?
    frame_interval   = 1.0 / CAMERA_FPS_LIMIT
    last_frame_time  = 0.0

    logger.info("Camera pipeline running. Press Ctrl+C to stop.")

    try:
        while True:
            now = time.monotonic()

            # Throttle to target FPS
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.monotonic()

            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read frame — retrying...")
                time.sleep(0.5)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Gesture detection (runs every frame) ──────────────────────
            gesture = gestures.process_frame(rgb)
            if gesture:
                logger.info("Gesture: %s", gesture)
                sender.send_gesture(gesture)

            # ── Face / presence detection (runs every frame) ──────────────
            faces = face_rec.identify(rgb)

            if faces:
                last_seen_time = now

                if not presence:
                    presence = True
                    logger.info("Presence: present")
                    sender.send_presence("present")

                # Use highest-confidence face
                best = max(faces, key=lambda f: f["confidence"])
                profile = best["profile"]

                if profile != last_profile:
                    last_profile   = profile
                    profile_since  = now
                    face_confirmed = False

                # Only send face event after it's been stable for N seconds
                stable = (now - profile_since) >= FACE_STABLE_SECONDS
                if stable and not face_confirmed and profile != "unknown":
                    logger.info("Face recognised: %s (confidence=%.2f)", profile, best["confidence"])
                    sender.send_face(profile, best["confidence"])
                    face_confirmed = True

            else:
                # No face in frame
                away_for = now - last_seen_time
                if presence and away_for >= PRESENCE_AWAY_AFTER:
                    presence       = False
                    last_profile   = None
                    face_confirmed = False
                    logger.info("Presence: away (no face for %.1fs)", away_for)
                    sender.send_presence("away")

    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        cap.release()
        gestures.close()


if __name__ == "__main__":
    main()
