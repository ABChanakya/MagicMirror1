"""
train.py — Build face embeddings from dataset/

Uses the same pipeline as face_recognizer.py:
  Detection:  OpenCV Haar cascade
  Embedding:  onnxruntime CPU + camera/model/face_embedding.onnx

Usage:
  python3 train.py [--output path/to/encodings.pkl]

Dataset layout:
  camera/dataset/
    your_name/
      photo1.jpg
      photo2.jpg
    other_person/
      photo1.jpg

Run again whenever you add new photos.
Python 3.6+ compatible.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "encodings.pkl"
ONNX_MODEL_PATH = Path(__file__).parent / "model" / "face_embedding.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def parse_args():
    p = argparse.ArgumentParser(description="Train face embeddings for MagicMirror3")
    p.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Output path for encodings.pkl")
    return p.parse_args()


def get_embedding(session, rgb_frame, x, y, w, h):
    """Identical preprocessing to face_recognizer.py._get_embedding."""
    try:
        face = rgb_frame[y:y + h, x:x + w]
        face = cv2.resize(face, (112, 112))
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, 0)

        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: face})
        embedding = result[0][0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


def train(model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Detector
    cascade_path = _find_haar_cascade()
    if not cascade_path:
        logger.error("Haar cascade not found. Install: sudo apt install python3-opencv")
        return
    detector = cv2.CascadeClassifier(cascade_path)
    logger.info("Face detector: Haar cascade")

    # Embedding model
    if not ONNX_MODEL_PATH.exists():
        logger.error("ONNX face model not found at %s", ONNX_MODEL_PATH)
        logger.error("Place a 112x112 face embedding model (MobileFaceNet / ArcFace) there.")
        return
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
        logger.info("Embedding model: %s", ONNX_MODEL_PATH.name)
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip3 install onnxruntime")
        return
    except Exception as e:
        logger.error("Could not load ONNX model: %s", e)
        return

    # Dataset
    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        logger.error("No images found in %s — add face photos first.", DATASET_DIR)
        return

    profiles = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not profiles:
        logger.error("No profile folders in %s", DATASET_DIR)
        return

    embeddings = []
    names = []
    summary = []  # list of (name, encoded, skipped)

    for profile_dir in tqdm(profiles, desc="Profiles"):
        profile_name = profile_dir.name
        images = sorted(
            [f for f in profile_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )

        if not images:
            logger.warning("No images in %s — skipping", profile_dir)
            summary.append((profile_name, 0, 0))
            continue

        logger.info("Processing '%s' (%d images)...", profile_name, len(images))
        encoded = 0
        skipped = 0

        for img_path in images:
            try:
                pil_img = Image.open(str(img_path))
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
                rgb = np.array(pil_img)
            except Exception as e:
                logger.warning("  Cannot open %s: %s", img_path.name, e)
                skipped += 1
                continue

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            rects = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(rects) == 0:
                logger.warning("  No face in %s — skipping", img_path.name)
                skipped += 1
                continue

            if len(rects) > 1:
                # Use largest face
                rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
                logger.warning("  Multiple faces in %s — using largest", img_path.name)

            x, y, w, h = rects[0]
            embedding = get_embedding(session, rgb, x, y, w, h)
            if embedding is None:
                skipped += 1
                continue

            embeddings.append(embedding.tolist())
            names.append(profile_name)
            encoded += 1

        logger.info("  Added %d embedding(s) for '%s'", encoded, profile_name)
        summary.append((profile_name, encoded, skipped))

    if not embeddings:
        logger.error("No valid embeddings produced. Check your photos.")
        return

    with open(str(model_path), "wb") as f:
        pickle.dump({"encodings": embeddings, "names": names}, f)

    logger.info("Saved %d embedding(s) for %d profile(s) -> %s",
                len(embeddings), len(set(names)), model_path)

    print("\n-- Training Summary ----------------------------------------")
    print("{:<20} {:>8} {:>8}".format("Profile", "Encoded", "Skipped"))
    print("-" * 40)
    for name, enc, skip in summary:
        print("{:<20} {:>8} {:>8}".format(name, enc, skip))
    print("-" * 40)
    print("{:<20} {:>8} {:>8}".format(
        "TOTAL",
        sum(e for _, e, _ in summary),
        sum(s for _, _, s in summary),
    ))


if __name__ == "__main__":
    args = parse_args()
    train(args.output)
