"""
face_recognizer.py
Loads known face encodings from disk and identifies faces in frames.
"""

import logging
import pickle
from pathlib import Path

import face_recognition
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model" / "encodings.pkl"


class FaceRecognizer:
    def __init__(self, model_path: Path = MODEL_PATH, tolerance: float = 0.45):
        self.tolerance = tolerance
        self.known_encodings: list = []
        self.known_names: list[str] = []
        self._load_model(model_path)

    def _load_model(self, path: Path):
        if not path.exists():
            logger.warning("No face model found at %s — run train.py first", path)
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.known_encodings = data["encodings"]
        self.known_names = data["names"]
        logger.info("Loaded %d face encoding(s): %s", len(self.known_names), set(self.known_names))

    def identify(self, rgb_frame: np.ndarray) -> list[dict]:
        """
        Find all faces in an RGB frame.
        Returns list of {"profile": str, "confidence": float, "location": tuple}.
        profile is "unknown" if no match found.
        """
        locations = face_recognition.face_locations(rgb_frame, model="hog")
        if not locations:
            return []

        encodings = face_recognition.face_encodings(rgb_frame, locations)
        results = []

        for encoding, location in zip(encodings, locations):
            if not self.known_encodings:
                results.append({"profile": "unknown", "confidence": 0.0, "location": location})
                continue

            distances = face_recognition.face_distance(self.known_encodings, encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            confidence = max(0.0, 1.0 - best_dist)

            if best_dist <= self.tolerance:
                profile = self.known_names[best_idx]
            else:
                profile = "unknown"

            results.append({
                "profile": profile,
                "confidence": confidence,
                "location": location,
            })

        return results
