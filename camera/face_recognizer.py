"""
face_recognizer.py
Crash-proof face recognition for Jetson Nano (OpenCV 3.2, Python 3.6).

Detection: Haar cascade (cv2.CascadeClassifier).
Recognition: cv2.face.createLBPHFaceRecognizer(), trained IN-MEMORY at startup
from a pickle of raw face crops + labels. We never call .read() / .load() on
a serialized LBPH .yml — both crash OpenCV 3.2 on this hardware.

Any failure (missing pickle, missing cv2.face, broken model, bad frame) falls
back to presence-only mode and is logged. identify() never raises.
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
FACE_DATA_PATH   = MODEL_DIR / "face_data.pkl"

FACE_SIZE            = (100, 100)
UNKNOWN_LABEL        = "unknown"
DEFAULT_THRESHOLD    = 0.4
LBPH_DISTANCE_SCALE  = 200.0


def _resolve_cascade_path():
    # type: () -> Optional[str]
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


def _lbph_factory():
    # type: () -> Optional[Any]
    """Return an LBPH constructor that works on OpenCV 3.2 (and 3.3+/4.x)."""
    face_mod = getattr(cv2, "face", None)
    if face_mod is None:
        return None
    for name in ("createLBPHFaceRecognizer", "LBPHFaceRecognizer_create"):
        fn = getattr(face_mod, name, None)
        if callable(fn):
            return fn
    return None


class FaceRecognizer:
    def __init__(self,
                 model_path=FACE_DATA_PATH,
                 tolerance=DEFAULT_THRESHOLD):
        # `model_path` keeps the old positional name but now points at face_data.pkl.
        # `tolerance` is treated as the minimum confidence (0-1) to accept a match.
        self.min_confidence = float(tolerance) if tolerance is not None else DEFAULT_THRESHOLD
        self.face_data_path = Path(model_path) if model_path is not None else FACE_DATA_PATH

        self._recognizer = None       # type: Optional[Any]
        self._label_map  = {}         # type: Dict[int, str]
        self._detector   = None       # type: Optional[cv2.CascadeClassifier]

        # ── detector ──────────────────────────────────────────────────────
        cascade_path = _resolve_cascade_path()
        if cascade_path is None:
            logger.error("Haar cascade XML not found — face detection disabled.")
        else:
            try:
                detector = cv2.CascadeClassifier(cascade_path)
                if detector.empty():
                    logger.error("Haar cascade at %s loaded empty — detection disabled.", cascade_path)
                else:
                    self._detector = detector
                    logger.info("Loaded Haar cascade from %s", cascade_path)
            except cv2.error as exc:
                logger.error("Failed to construct Haar cascade from %s: %s", cascade_path, exc)

        # ── recognizer ────────────────────────────────────────────────────
        self._load_recognizer()

    # ── public API ──────────────────────────────────────────────────────────

    def reload(self):
        # type: () -> None
        """Reload training data from disk and retrain the recognizer in-memory."""
        self._recognizer = None
        self._label_map = {}
        self._load_recognizer()

    def identify(self, rgb_frame):
        # type: (np.ndarray) -> List[Dict[str, Any]]
        """
        Detect faces in an RGB frame and try to recognise them.

        Returns a list of dicts:
          {"profile": str, "confidence": float, "location": (top,right,bottom,left)}

        profile == "unknown" when no model is trained, when recognition fails,
        or when confidence is below the configured threshold.
        Never raises — all errors are caught and logged.
        """
        try:
            return self._identify_unsafe(rgb_frame)
        except Exception as exc:  # last-line-of-defence; identify() must never crash main
            logger.exception("identify() failed unexpectedly: %s", exc)
            return []

    # ── internals ───────────────────────────────────────────────────────────

    def _identify_unsafe(self, rgb_frame):
        # type: (np.ndarray) -> List[Dict[str, Any]]
        if self._detector is None:
            return []
        if rgb_frame is None or not hasattr(rgb_frame, "size") or rgb_frame.size == 0:
            return []

        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
        except cv2.error as exc:
            logger.warning("Color conversion failed: %s", exc)
            return []

        try:
            detections = self._detector.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(30, 30),
            )
        except cv2.error as exc:
            logger.warning("Face detection failed: %s", exc)
            return []

        results = []  # type: List[Dict[str, Any]]
        for det in detections:
            x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            top, right, bottom, left = y, x + w, y + h, x
            location = (top, right, bottom, left)

            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            try:
                crop = cv2.resize(crop, FACE_SIZE)
            except cv2.error as exc:
                logger.warning("Face resize failed: %s", exc)
                continue

            if self._recognizer is None or not self._label_map:
                results.append({
                    "profile": UNKNOWN_LABEL,
                    "confidence": 0.0,
                    "location": location,
                })
                continue

            try:
                label, distance = self._recognizer.predict(crop)
            except cv2.error as exc:
                logger.warning("LBPH predict failed: %s", exc)
                results.append({
                    "profile": UNKNOWN_LABEL,
                    "confidence": 0.0,
                    "location": location,
                })
                continue
            except Exception as exc:
                logger.warning("LBPH predict raised %s: %s", type(exc).__name__, exc)
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

    def _load_recognizer(self):
        # type: () -> None
        factory = _lbph_factory()
        if factory is None:
            logger.warning(
                "cv2.face LBPH recognizer not available — running presence-only. "
                "Install opencv-contrib or use a build with face module."
            )
            return

        if not self.face_data_path.exists():
            logger.warning(
                "No training data at %s — run train.py first. Presence-only mode.",
                self.face_data_path,
            )
            return

        try:
            with open(str(self.face_data_path), "rb") as f:
                data = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
            logger.error("Failed to load training data from %s: %s", self.face_data_path, exc)
            return
        except Exception as exc:
            logger.error("Unexpected error loading %s: %s", self.face_data_path, exc)
            return

        faces, labels, label_map = self._validate_data(data)
        if faces is None:
            return

        try:
            recognizer = factory()
            recognizer.train(faces, np.asarray(labels, dtype=np.int32))
        except cv2.error as exc:
            logger.error("LBPH in-memory training failed: %s", exc)
            return
        except Exception as exc:
            logger.error("Unexpected error training LBPH: %s", exc)
            return

        self._recognizer = recognizer
        self._label_map  = label_map
        logger.info(
            "Trained LBPH in-memory from %d sample(s), %d profile(s): %s",
            len(faces), len(label_map), sorted(label_map.values()),
        )

    @staticmethod
    def _validate_data(data):
        # type: (Any) -> Tuple[Optional[List[np.ndarray]], Optional[List[int]], Dict[int, str]]
        if not isinstance(data, dict):
            logger.error("Training data must be a dict, got %s", type(data).__name__)
            return None, None, {}

        faces_raw  = data.get("faces")
        labels_raw = data.get("labels")
        label_map_raw = data.get("label_map")

        if not faces_raw or not labels_raw or not label_map_raw:
            logger.error("Training data missing one of: faces, labels, label_map")
            return None, None, {}
        if len(faces_raw) != len(labels_raw):
            logger.error("faces (%d) and labels (%d) length mismatch", len(faces_raw), len(labels_raw))
            return None, None, {}

        faces = []  # type: List[np.ndarray]
        for arr in faces_raw:
            if not isinstance(arr, np.ndarray):
                logger.error("Non-ndarray entry in faces; aborting load.")
                return None, None, {}
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.shape != FACE_SIZE:
                try:
                    arr = cv2.resize(arr, FACE_SIZE)
                except cv2.error as exc:
                    logger.error("Could not normalise face crop shape %s: %s", arr.shape, exc)
                    return None, None, {}
            faces.append(arr)

        try:
            labels = [int(v) for v in labels_raw]
        except (TypeError, ValueError) as exc:
            logger.error("Could not coerce labels to int: %s", exc)
            return None, None, {}

        try:
            label_map = {int(k): str(v) for k, v in label_map_raw.items()}
        except (AttributeError, TypeError, ValueError) as exc:
            logger.error("Could not coerce label_map: %s", exc)
            return None, None, {}

        return faces, labels, label_map
