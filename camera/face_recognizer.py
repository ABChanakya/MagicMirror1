"""
face_recognizer.py
Face detection + optional recognition for Jetson Nano.

Detection:   OpenCV Haar cascade (system OpenCV 3.2, no pip needed)
Embedding:   onnxruntime CPU with a 112x112 face embedding ONNX model
             Expects: camera/model/face_embedding.onnx
             Compatible with MobileFaceNet, ArcFace-MobileNet, etc.

Falls back gracefully:
  - No onnxruntime  → presence detection only (faces detected, names unknown)
  - No ONNX model   → presence detection only
  - No encodings    → faces detected, identified as "unknown"

Python 3.6+ compatible.
"""

import logging
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model" / "encodings.pkl"
ONNX_MODEL_PATH = Path(__file__).parent / "model" / "face_embedding.onnx"


def _find_haar_cascade():
    # type: () -> str
    """Find haarcascade_frontalface_default.xml across common system locations."""
    candidates = [
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv-3.2/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    # cv2.data available in OpenCV 3.4+
    try:
        candidates.insert(0, cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except AttributeError:
        pass

    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, tolerance=0.45):
        self.tolerance = tolerance
        self.model_path = model_path
        self.known_embeddings = []
        self.known_names = []
        self._detector = None
        self._session = None

        self._init_detector()
        self._init_embedding()
        self._load_model(model_path)

    # ── Setup ───────────────────────────────────────────────────────────────

    def _init_detector(self):
        cascade_path = _find_haar_cascade()
        if not cascade_path:
            logger.error("Haar cascade XML not found. Install: sudo apt install python3-opencv")
            return
        self._detector = cv2.CascadeClassifier(cascade_path)
        logger.info("Face detector: Haar cascade (%s)", cascade_path)

    def _init_embedding(self):
        if not ONNX_MODEL_PATH.exists():
            logger.warning("ONNX face model not found at %s", ONNX_MODEL_PATH)
            logger.warning("Face recognition disabled — presence detection only")
            logger.warning("Place a 112x112 face embedding ONNX model at that path to enable.")
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(ONNX_MODEL_PATH),
                providers=["CPUExecutionProvider"],
            )
            inp = self._session.get_inputs()[0]
            logger.info("Face embedding model loaded: %s (input: %s %s)",
                        ONNX_MODEL_PATH.name, inp.name, inp.shape)
        except ImportError:
            logger.warning("onnxruntime not installed — face recognition disabled")
            logger.warning("Run: pip3 install onnxruntime")
        except Exception as e:
            logger.warning("Could not load ONNX model: %s", e)

    def _load_model(self, path):
        if not path.exists():
            logger.warning("No encodings at %s — run train.py first", path)
            return
        with open(str(path), "rb") as f:
            data = pickle.load(f)
        self.known_embeddings = data["encodings"]
        self.known_names = data["names"]
        logger.info("Loaded %d embedding(s): profiles=%s",
                    len(self.known_names), set(self.known_names))

    # ── Public API ──────────────────────────────────────────────────────────

    def identify(self, rgb_frame):
        """
        Detect faces and optionally identify them.
        Returns list of {"profile": str, "confidence": float, "location": (top,right,bottom,left)}.
        """
        if self._detector is None:
            return []

        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        rects = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(rects) == 0:
            return []

        output = []
        for (x, y, w, h) in rects:
            location = (y, x + w, y + h, x)  # (top, right, bottom, left)

            if self._session is not None and self.known_embeddings:
                embedding = self._get_embedding(rgb_frame, x, y, w, h)
                if embedding is not None:
                    profile, confidence = self._match(embedding)
                    output.append({"profile": profile, "confidence": confidence, "location": location})
                    continue

            output.append({"profile": "unknown", "confidence": 0.0, "location": location})

        return output

    def reload(self):
        """Hot-reload face embeddings from disk without restarting the pipeline."""
        logger.info("Reloading face embeddings from %s", self.model_path)
        self.known_embeddings = []
        self.known_names = []
        self._load_model(self.model_path)

    # ── Internal ────────────────────────────────────────────────────────────

    def _get_embedding(self, rgb_frame, x, y, w, h):
        """
        Crop face, preprocess for 112x112 NCHW input, run ONNX session.
        Normalization: (pixel - 127.5) / 128.0 — works with MobileFaceNet / ArcFace variants.
        """
        try:
            face = rgb_frame[y:y + h, x:x + w]
            face = cv2.resize(face, (112, 112))
            face = face.astype(np.float32)
            face = (face - 127.5) / 128.0
            face = np.transpose(face, (2, 0, 1))   # HWC -> CHW
            face = np.expand_dims(face, 0)          # (1, 3, 112, 112)

            input_name = self._session.get_inputs()[0].name
            result = self._session.run(None, {input_name: face})
            embedding = result[0][0]

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            logger.debug("Embedding error: %s", e)
            return None

    def _match(self, embedding):
        sims = [float(np.dot(embedding, np.array(e))) for e in self.known_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        threshold = 1.0 - self.tolerance
        if best_sim >= threshold:
            return self.known_names[best_idx], best_sim
        return "unknown", best_sim
