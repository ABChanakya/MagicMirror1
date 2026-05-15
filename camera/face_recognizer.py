"""
face_recognizer.py
Face detection and recognition for Jetson Nano.

Primary:  face_recognition library (dlib-based, 128-dim embeddings)
          pip3 install dlib==19.21.1 face_recognition==1.3.0
          (dlib compiles from source — run setup.sh once)

Fallback: OpenCV Haar cascade (system OpenCV 3.2) — presence detection only,
          all faces reported as "unknown" if face_recognition is not installed.

Python 3.6+ compatible — no walrus, no str|None, no match/case.
"""

import logging
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model" / "encodings.pkl"


def _find_haar_cascade():
    # type: () -> str
    candidates = [
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv-3.2/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    try:
        candidates.insert(0, cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except AttributeError:
        pass
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, tolerance=0.55):
        # tolerance: max Euclidean distance to accept a match (lower = stricter)
        # face_recognition default is 0.6; 0.55 is slightly stricter
        self.tolerance = tolerance
        self.model_path = model_path
        self.known_encodings = []
        self.known_names = []
        self._fr = None          # face_recognition module, or None if unavailable
        self._detector = None    # Haar cascade fallback

        self._init_face_recognition()
        self._load_model(model_path)

    # ── Setup ───────────────────────────────────────────────────────────────

    def _init_face_recognition(self):
        try:
            import face_recognition
            self._fr = face_recognition
            logger.info("FaceRecognizer: face_recognition (dlib) ready")
        except ImportError:
            logger.warning("face_recognition not installed — falling back to Haar cascade")
            logger.warning("To enable: pip3 install dlib==19.21.1 face_recognition==1.3.0")
            self._init_haar_fallback()

    def _init_haar_fallback(self):
        cascade_path = _find_haar_cascade()
        if not cascade_path:
            logger.error("Haar cascade not found. Install: sudo apt install python3-opencv")
            return
        self._detector = cv2.CascadeClassifier(cascade_path)
        logger.info("FaceRecognizer: Haar cascade fallback active (presence only)")

    def _load_model(self, path):
        if not path.exists():
            logger.warning("No encodings at %s — run train.py first", path)
            return
        with open(str(path), "rb") as f:
            data = pickle.load(f)
        self.known_encodings = data["encodings"]
        self.known_names = data["names"]
        logger.info("Loaded %d encoding(s): profiles=%s",
                    len(self.known_names), set(self.known_names))

    # ── Public API ──────────────────────────────────────────────────────────

    def identify(self, rgb_frame):
        """
        Detect faces and optionally identify them.
        Returns list of {"profile": str, "confidence": float, "location": (top,right,bottom,left)}.
        """
        if self._fr is not None:
            return self._identify_fr(rgb_frame)
        if self._detector is not None:
            return self._identify_haar(rgb_frame)
        return []

    def reload(self):
        """Hot-reload face encodings from disk."""
        logger.info("Reloading face encodings from %s", self.model_path)
        self.known_encodings = []
        self.known_names = []
        self._load_model(self.model_path)

    # ── face_recognition path ───────────────────────────────────────────────

    def _identify_fr(self, rgb_frame):
        try:
            locations = self._fr.face_locations(rgb_frame, model="hog")
        except Exception as exc:
            logger.debug("face_locations error: %s", exc)
            return []

        if not locations:
            return []

        output = []

        if self.known_encodings:
            try:
                encodings = self._fr.face_encodings(rgb_frame, locations)
            except Exception as exc:
                logger.debug("face_encodings error: %s", exc)
                encodings = []

            known_np = [np.array(e) for e in self.known_encodings]

            for location, encoding in zip(locations, encodings):
                profile, confidence = self._match_fr(encoding, known_np)
                output.append({"profile": profile, "confidence": confidence,
                               "location": location})

            # Faces with no encoding (edge case)
            for location in locations[len(encodings):]:
                output.append({"profile": "unknown", "confidence": 0.0,
                               "location": location})
        else:
            for location in locations:
                output.append({"profile": "unknown", "confidence": 0.0,
                               "location": location})

        return output

    def _match_fr(self, encoding, known_np):
        # type: (np.ndarray, list) -> tuple
        distances = self._fr.face_distance(known_np, encoding)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        # Convert distance to a confidence-like score: 0.0 = no match, 1.0 = perfect
        confidence = max(0.0, 1.0 - best_dist)
        if best_dist <= self.tolerance:
            return self.known_names[best_idx], confidence
        return "unknown", confidence

    # ── Haar cascade fallback path ──────────────────────────────────────────

    def _identify_haar(self, rgb_frame):
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        rects = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(rects) == 0:
            return []
        output = []
        for (x, y, w, h) in rects:
            location = (y, x + w, y + h, x)   # (top, right, bottom, left)
            output.append({"profile": "unknown", "confidence": 0.0, "location": location})
        return output
