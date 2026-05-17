"""
face_recognizer.py
Detects faces with a Haar cascade and identifies them with OpenCV's LBPH
recognizer. Designed for OpenCV 3.2 on Jetson Nano (Python 3.6, ARM64) —
no dlib, no onnxruntime, no extra pip packages beyond what main.py already uses.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR        = Path(__file__).parent / "model"
LBPH_MODEL_PATH  = MODEL_DIR / "lbph_model.yml"
LABEL_MAP_PATH   = MODEL_DIR / "label_map.pkl"

FACE_SIZE        = (100, 100)            # match train.py
UNKNOWN_LABEL    = "unknown"
DEFAULT_THRESHOLD = 0.4                   # min confidence (0-1) to accept a match
LBPH_DISTANCE_SCALE = 200.0               # distance -> confidence normaliser


def _resolve_cascade_path() -> Optional[str]:
    """Find a usable frontal-face Haar cascade XML, trying several known
    locations so we work on both stock Jetson OpenCV 3.2 (apt) and pip OpenCV."""
    candidates = []  # type: List[str]

    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidates.append(os.path.join(data_dir, "haarcascade_frontalface_default.xml"))

    candidates.extend([
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml",
    ])

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


class FaceRecognizer:
    def __init__(self,
                 model_path: Path = LBPH_MODEL_PATH,
                 label_map_path: Path = LABEL_MAP_PATH,
                 tolerance: float = DEFAULT_THRESHOLD):
        # `tolerance` is kept in the signature for compatibility with main.py
        # (which passes FACE_TOLERANCE). It is treated here as the minimum
        # confidence (0-1) required to accept a recognised face.
        self.min_confidence = float(tolerance)
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)

        self._recognizer = None       # type: Optional[Any]
        self._label_map  = {}         # type: Dict[int, str]
        self._face_module_ok = True

        cascade_path = _resolve_cascade_path()
        if cascade_path is None:
            logger.error("Could not locate a Haar cascade XML — face detection disabled.")
            self._detector = None
        else:
            self._detector = cv2.CascadeClassifier(cascade_path)
            if self._detector.empty():
                logger.error("Haar cascade at %s failed to load — face detection disabled.", cascade_path)
                self._detector = None
            else:
                logger.info("Loaded Haar cascade from %s", cascade_path)

        self._load_model()

    # ── public API ──────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Reload the trained LBPH model and label map from disk."""
        self._recognizer = None
        self._label_map = {}
        self._load_model()

    def identify(self, rgb_frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find all faces in an RGB frame.
        Returns list of {"profile": str, "confidence": float, "location": (top, right, bottom, left)}.

        - profile is "unknown" when no trained model exists, when the face
          does not match any known profile, or when confidence is below the
          configured threshold.
        - confidence is in [0.0, 1.0] (higher is better).
        """
        if self._detector is None or not self._face_module_ok:
            return []
        if rgb_frame is None or rgb_frame.size == 0:
            return []

        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        results = []  # type: List[Dict[str, Any]]
        for (x, y, w, h) in detections:
            top, right, bottom, left = int(y), int(x + w), int(y + h), int(x)
            location = (top, right, bottom, left)

            face_crop = gray[y:y + h, x:x + w]
            if face_crop.size == 0:
                continue
            face_crop = cv2.resize(face_crop, FACE_SIZE)

            if self._recognizer is None or not self._label_map:
                # Presence-only: face detected but no trained model.
                results.append({
                    "profile": UNKNOWN_LABEL,
                    "confidence": 0.0,
                    "location": location,
                })
                continue

            try:
                label, distance = self._recognizer.predict(face_crop)
            except cv2.error as exc:
                logger.warning("LBPH predict failed: %s", exc)
                results.append({
                    "profile": UNKNOWN_LABEL,
                    "confidence": 0.0,
                    "location": location,
                })
                continue

            confidence = max(0.0, 1.0 - (float(distance) / LBPH_DISTANCE_SCALE))
            profile = self._label_map.get(int(label), UNKNOWN_LABEL)
            if confidence < self.min_confidence:
                profile = UNKNOWN_LABEL

            results.append({
                "profile": profile,
                "confidence": confidence,
                "location": location,
            })

        return results

    # ── internals ───────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        face_mod = getattr(cv2, "face", None)
        if face_mod is None or not hasattr(face_mod, "LBPHFaceRecognizer_create"):
            logger.warning(
                "cv2.face module not available — install opencv-contrib or skip "
                "recognition. Returning empty results from identify()."
            )
            self._face_module_ok = False
            return

        if not self.model_path.exists() or not self.label_map_path.exists():
            logger.warning(
                "No trained LBPH model at %s (or label map at %s) — "
                "run train.py first. Falling back to presence-only.",
                self.model_path, self.label_map_path,
            )
            return

        try:
            recognizer = face_mod.LBPHFaceRecognizer_create()
            recognizer.read(str(self.model_path))
        except cv2.error as exc:
            logger.error("Failed to load LBPH model from %s: %s", self.model_path, exc)
            return

        try:
            with open(str(self.label_map_path), "rb") as f:
                label_map = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as exc:
            logger.error("Failed to load label map from %s: %s", self.label_map_path, exc)
            return

        if not isinstance(label_map, dict):
            logger.error("Label map at %s is not a dict — got %s", self.label_map_path, type(label_map))
            return

        self._recognizer = recognizer
        self._label_map  = {int(k): str(v) for k, v in label_map.items()}
        logger.info(
            "Loaded LBPH model with %d profile(s): %s",
            len(self._label_map), sorted(self._label_map.values()),
        )
