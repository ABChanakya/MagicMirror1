"""
train.py — Train an OpenCV LBPH face recognizer from dataset/

Usage:
  python3 train.py

Expects this folder structure:
  camera/dataset/
    Chanakya/   <- folder name becomes the profile name
      IMG_0001.jpg
      IMG_0002.jpg
      ...
    SomeoneElse/
      photo1.jpg
      ...

Outputs:
  camera/model/lbph_model.yml   — the trained LBPH recognizer
  camera/model/label_map.pkl    — pickled {int -> profile name}

Re-run this whenever you add or remove face photos.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

DATASET_DIR     = Path(__file__).parent / "dataset"
MODEL_DIR       = Path(__file__).parent / "model"
LBPH_MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABEL_MAP_PATH  = MODEL_DIR / "label_map.pkl"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_SIZE  = (100, 100)


def _resolve_cascade_path() -> Optional[str]:
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


def _load_gray_image(path: Path) -> Optional[np.ndarray]:
    """Load an image as a grayscale numpy array, respecting EXIF orientation."""
    try:
        with Image.open(str(path)) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("L")
            return np.asarray(im, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        logger.warning("  Could not open %s: %s", path.name, exc)
        return None


def _detect_largest_face(detector: cv2.CascadeClassifier,
                          gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    equalised = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        equalised,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    # Pick the largest detection — typically the subject of a portrait shot.
    x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
    return int(x), int(y), int(w), int(h)


def train() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    face_mod = getattr(cv2, "face", None)
    if face_mod is None or not hasattr(face_mod, "LBPHFaceRecognizer_create"):
        logger.error(
            "cv2.face module not available. Install opencv-contrib (e.g. "
            "`pip3 install opencv-contrib-python`) or use an OpenCV build "
            "that includes the contrib modules."
        )
        return 1

    cascade_path = _resolve_cascade_path()
    if cascade_path is None:
        logger.error("Could not locate a Haar cascade XML on this system.")
        return 1
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        logger.error("Failed to load Haar cascade from %s", cascade_path)
        return 1
    logger.info("Using Haar cascade: %s", cascade_path)

    if not DATASET_DIR.exists():
        logger.error("Dataset directory %s does not exist — add face photos first.", DATASET_DIR)
        return 1

    profiles = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not profiles:
        logger.error("No profile folders found in %s", DATASET_DIR)
        return 1

    samples = []     # type: List[np.ndarray]
    labels  = []     # type: List[int]
    label_map = {}   # type: Dict[int, str]
    stats = []       # type: List[Tuple[str, int, int]]

    next_label = 0
    for profile_dir in profiles:
        profile_name = profile_dir.name
        images = sorted([f for f in profile_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])

        if not images:
            logger.warning("No images in %s — skipping", profile_dir)
            stats.append((profile_name, 0, 0))
            continue

        label = next_label
        next_label += 1
        label_map[label] = profile_name

        logger.info("Processing '%s' (%d image(s))...", profile_name, len(images))
        used = 0
        skipped = 0

        for img_path in images:
            gray = _load_gray_image(img_path)
            if gray is None:
                skipped += 1
                continue

            face = _detect_largest_face(detector, gray)
            if face is None:
                logger.warning("  No face detected in %s — skipping", img_path.name)
                skipped += 1
                continue

            x, y, w, h = face
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                skipped += 1
                continue
            crop = cv2.resize(crop, FACE_SIZE)

            samples.append(crop)
            labels.append(label)
            used += 1

        logger.info("  Added %d sample(s) for '%s' (%d skipped)", used, profile_name, skipped)
        stats.append((profile_name, used, skipped))

    if not samples:
        logger.error("No usable face samples found. Check your photos.")
        return 1

    recognizer = face_mod.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.asarray(labels, dtype=np.int32))
    recognizer.write(str(LBPH_MODEL_PATH))

    with open(str(LABEL_MAP_PATH), "wb") as f:
        pickle.dump(label_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        "Saved LBPH model (%d sample(s), %d profile(s)) to %s",
        len(samples), len(label_map), LBPH_MODEL_PATH,
    )
    logger.info("Saved label map to %s", LABEL_MAP_PATH)

    _print_summary(stats)
    return 0


def _print_summary(stats: List[Tuple[str, int, int]]) -> None:
    name_w = max(7, max((len(s[0]) for s in stats), default=7))
    header = "{name:<{w}}  {used:>11}  {skipped:>14}".format(
        name="profile", w=name_w, used="photos used", skipped="photos skipped",
    )
    print()
    print(header)
    print("-" * len(header))
    for name, used, skipped in stats:
        print("{name:<{w}}  {used:>11d}  {skipped:>14d}".format(
            name=name, w=name_w, used=used, skipped=skipped,
        ))
    print()


if __name__ == "__main__":
    sys.exit(train())
