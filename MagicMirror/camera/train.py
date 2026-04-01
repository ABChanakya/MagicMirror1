"""
train.py — Build face encodings from dataset/

Usage:
  python3 train.py

Expects this folder structure:
  camera/dataset/
    kind1/  <- folder name becomes the profile name
      photo1.jpg
      photo2.jpg
      ...
    mama/
      photo1.jpg
      ...

Saves encodings to camera/model/encodings.pkl
Run this again whenever you add new face photos.
"""

import logging
import pickle
from pathlib import Path

import face_recognition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

DATASET_DIR = Path(__file__).parent / "dataset"
MODEL_PATH  = Path(__file__).parent / "model" / "encodings.pkl"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def train():
    MODEL_PATH.parent.mkdir(exist_ok=True)

    encodings = []
    names     = []

    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        logger.error("No images found in %s — add face photos first.", DATASET_DIR)
        return

    profiles = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not profiles:
        logger.error("No profile folders found in %s", DATASET_DIR)
        return

    for profile_dir in profiles:
        profile_name = profile_dir.name
        images = [f for f in profile_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]

        if not images:
            logger.warning("No images in %s — skipping", profile_dir)
            continue

        logger.info("Processing profile '%s' (%d image(s))...", profile_name, len(images))
        count = 0

        for img_path in images:
            img = face_recognition.load_image_file(str(img_path))
            found = face_recognition.face_encodings(img)

            if not found:
                logger.warning("  No face detected in %s — skipping", img_path.name)
                continue

            if len(found) > 1:
                logger.warning("  Multiple faces in %s — using first only", img_path.name)

            encodings.append(found[0])
            names.append(profile_name)
            count += 1

        logger.info("  Added %d encoding(s) for '%s'", count, profile_name)

    if not encodings:
        logger.error("No valid encodings produced. Check your photos.")
        return

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    logger.info("Saved %d encoding(s) for %d profile(s) to %s",
                len(encodings), len(set(names)), MODEL_PATH)


if __name__ == "__main__":
    train()
