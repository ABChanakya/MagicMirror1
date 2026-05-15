"""
train.py — Build face encodings from dataset/ using face_recognition (dlib).

Replaces the previous InsightFace / ONNX Runtime approach.

Requires:
  pip3 install dlib==19.21.1 face_recognition==1.3.0
  (dlib compiles from source — run setup.sh, takes ~30 min on Jetson Nano)

Usage:
  python3 train.py [--output path/to/encodings.pkl]

Dataset layout:
  camera/dataset/
    your_name/
      photo1.jpg
      photo2.jpg
    other_person/
      photo1.jpg

Python 3.6+ compatible.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "encodings.pkl"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Train face encodings for MagicMirror3")
    p.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Output path for encodings.pkl")
    return p.parse_args()


def train(model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import face_recognition
    except ImportError:
        logger.error("face_recognition not installed.")
        logger.error("Run: pip3 install dlib==19.21.1 face_recognition==1.3.0")
        return

    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        logger.error("No images in %s — add face photos first.", DATASET_DIR)
        return

    profiles = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not profiles:
        logger.error("No profile folders in %s", DATASET_DIR)
        return

    encodings = []
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

            locations = face_recognition.face_locations(rgb, model="hog")
            if not locations:
                logger.warning("  No face in %s — skipping", img_path.name)
                skipped += 1
                continue

            if len(locations) > 1:
                # Use the largest face by bounding-box area
                locations = [max(locations,
                                 key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))]
                logger.warning("  Multiple faces in %s — using largest", img_path.name)

            face_encs = face_recognition.face_encodings(rgb, locations)
            if not face_encs:
                logger.warning("  Could not encode face in %s — skipping", img_path.name)
                skipped += 1
                continue

            encodings.append(face_encs[0].tolist())
            names.append(profile_name)
            encoded += 1

        logger.info("  Added %d encoding(s) for '%s'", encoded, profile_name)
        summary.append((profile_name, encoded, skipped))

    if not encodings:
        logger.error("No valid encodings produced. Check your photos.")
        return

    with open(str(model_path), "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    logger.info("Saved %d encoding(s) for %d profile(s) -> %s",
                len(encodings), len(set(names)), model_path)

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
