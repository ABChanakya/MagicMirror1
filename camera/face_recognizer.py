"""
face_recognizer.py
GPU-accelerated face recognition using InsightFace + ONNX Runtime (CUDA/TensorRT).

Works on:
  - RTX 4090 (development) — CUDAExecutionProvider
  - Jetson Nano            — CUDAExecutionProvider (JetPack ONNX Runtime)
  - CPU fallback           — CPUExecutionProvider
"""

import logging
import os
import pickle
import site
from pathlib import Path

import numpy as np


def _patch_cuda_libs():
    """Add pip-installed CUDA libs to LD_LIBRARY_PATH so ONNX Runtime can find them."""
    cuda_lib_dirs = []
    for sp in site.getsitepackages():
        nvidia_dir = Path(sp) / "nvidia"
        if nvidia_dir.exists():
            for lib_dir in nvidia_dir.glob("*/lib"):
                cuda_lib_dirs.append(str(lib_dir))
    if cuda_lib_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_lib_dirs) + (":" + existing if existing else "")


_patch_cuda_libs()

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model" / "encodings.pkl"


class FaceRecognizer:
    def __init__(self, model_path: Path = MODEL_PATH, tolerance: float = 0.45):
        self.tolerance         = tolerance
        self.model_path        = model_path
        self.known_embeddings: list      = []
        self.known_names:      list[str] = []
        self._app              = None
        self._load_model(model_path)
        self._init_insightface()

    # ── Setup ───────────────────────────────────────────────────────────────

    def _load_model(self, path: Path):
        if not path.exists():
            logger.warning("No face model at %s — run train.py first", path)
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.known_embeddings = data["encodings"]
        self.known_names      = data["names"]
        logger.info("Loaded %d embedding(s): profiles=%s",
                    len(self.known_names), set(self.known_names))

    def _init_insightface(self):
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("InsightFace using GPU (CUDA)")
            else:
                providers = ["CPUExecutionProvider"]
                logger.info("InsightFace using CPU")

            self._app = FaceAnalysis(name="buffalo_sc", providers=providers)
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace ready")
        except Exception as e:
            logger.error("InsightFace init failed: %s — face recognition disabled", e)

    # ── Public API ──────────────────────────────────────────────────────────

    def identify(self, rgb_frame: np.ndarray) -> list[dict]:
        """
        Find all faces in an RGB frame.
        Returns list of {"profile": str, "confidence": float, "location": (top,right,bottom,left)}.
        """
        if self._app is None:
            return []

        try:
            import cv2
            bgr   = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            faces = self._app.get(bgr)
        except Exception as e:
            logger.debug("InsightFace error: %s", e)
            return []

        output = []
        for face in faces:
            bbox               = face.bbox.astype(int)
            x1, y1, x2, y2    = bbox
            location           = (y1, x2, y2, x1)  # (top, right, bottom, left)
            embedding          = face.normed_embedding

            if not self.known_embeddings:
                output.append({"profile": "unknown", "confidence": 0.0, "location": location})
                continue

            profile, confidence = self._match(embedding)
            output.append({"profile": profile, "confidence": confidence, "location": location})

        return output

    def reload(self):
        """Hot-reload face embeddings from disk without restarting the pipeline."""
        logger.info("Reloading face embeddings from %s", self.model_path)
        self.known_embeddings = []
        self.known_names      = []
        self._load_model(self.model_path)

    # ── Internal ────────────────────────────────────────────────────────────

    def _match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Cosine similarity — embeddings are pre-normalised so just dot product."""
        sims     = [float(np.dot(embedding, np.array(e))) for e in self.known_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        # tolerance=0.45 → threshold=0.55 (stricter); tolerance=0.7 → threshold=0.3 (looser)
        threshold = 1.0 - self.tolerance
        if best_sim >= threshold:
            return self.known_names[best_idx], best_sim
        return "unknown", best_sim
