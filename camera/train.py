"""
train.py — Build raw face training data from dataset/

Usage:
  python3 train.py

Folder structure:
  camera/dataset/
    your_face/               <- folder name becomes the profile label
      IMG_0001.JPG
      ...
    other_person/
      ...

Output:
  camera/model/face_data.pkl   — pickle of {"faces": [...], "labels": [...], "label_map": {...}}

We deliberately do NOT save a serialized LBPH .yml — OpenCV 3.2 on Jetson
crashes when reading it back. face_recognizer.py retrains LBPH in-memory
from this pickle each time the camera process starts.
"""

import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

DATASET_DIR    = Path(__file__).parent / "dataset"
MODEL_DIR      = Path(__file__).parent / "model"
FACE_DATA_PATH = MODEL_DIR / "face_data.pkl"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_SIZE  = (100, 100)

MIN_AUGS_PER_IMAGE       = 4
MIN_SAMPLES_PER_PROFILE  = int(os.getenv("MIN_SAMPLES_PER_PROFILE", "100"))


def _augmentation_pool(crop):
    # type: (np.ndarray) -> List[np.ndarray]
    """Generate up to 20 augmented variants of a 100x100 grayscale crop.

    Order matters: earlier entries are gentler (common conditions), later
    entries are more aggressive (extreme angles, contrast). We take the
    first N from this list per source image, where N is chosen so each
    profile reaches MIN_SAMPLES_PER_PROFILE.
    """
    h, w = crop.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def _rotate(angle):
        m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(
            crop, m, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _brightness(delta):
        return np.clip(crop.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    def _gamma(g):
        table = np.array(
            [((i / 255.0) ** g) * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(crop, table)

    def _zoom(factor):
        # >1 = crop inward (zoom in); <1 = pad with replicated border (zoom out)
        if factor >= 1.0:
            new_w = max(1, int(w / factor))
            new_h = max(1, int(h / factor))
            x = (w - new_w) // 2
            y = (h - new_h) // 2
            sub = crop[y:y + new_h, x:x + new_w]
            return cv2.resize(sub, (w, h), interpolation=cv2.INTER_LINEAR)
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
        small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        return cv2.copyMakeBorder(
            small, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
            cv2.BORDER_REPLICATE,
        )

    pool = [
        crop,                                # 1  identity
        cv2.flip(crop, 1),                   # 2  hflip
        _rotate(-7),                         # 3  small CCW
        _rotate(7),                          # 4  small CW
        cv2.flip(_rotate(-7), 1),            # 5  small CCW + flip
        cv2.flip(_rotate(7), 1),             # 6  small CW + flip
        _brightness(-25),                    # 7  darker
        _brightness(25),                     # 8  brighter
        cv2.flip(_brightness(-25), 1),       # 9
        cv2.flip(_brightness(25), 1),        # 10
        _gamma(0.75),                        # 11 high-contrast lift
        _gamma(1.3),                         # 12 low-contrast crush
        _zoom(1.1),                          # 13 slight zoom-in
        _zoom(0.9),                          # 14 slight zoom-out
        _rotate(-15),                        # 15 larger CCW
        _rotate(15),                         # 16 larger CW
        _brightness(15),                     # 17 mild brighter
        _brightness(-15),                    # 18 mild darker
        cv2.flip(_gamma(0.8), 1),            # 19
        cv2.flip(_gamma(1.2), 1),            # 20
    ]
    return pool


_POOL_SIZE = 20  # keep in sync with _augmentation_pool


def _augs_per_image(num_source_images):
    # type: (int) -> int
    """How many augmentations to take per source image so the profile
    reaches MIN_SAMPLES_PER_PROFILE total. Bounded by [MIN_AUGS, POOL_SIZE]."""
    if num_source_images <= 0:
        return MIN_AUGS_PER_IMAGE
    needed = int(math.ceil(MIN_SAMPLES_PER_PROFILE / float(num_source_images)))
    return max(MIN_AUGS_PER_IMAGE, min(_POOL_SIZE, needed))


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


def _load_gray_image(path):
    # type: (Path) -> Optional[np.ndarray]
    try:
        with Image.open(str(path)) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("L")
            return np.asarray(im, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        logger.warning("  Could not open %s: %s", path.name, exc)
        return None


def _detect_largest_face(detector, gray):
    # type: (cv2.CascadeClassifier, np.ndarray) -> Optional[Tuple[int, int, int, int]]
    equalised = cv2.equalizeHist(gray)
    try:
        faces = detector.detectMultiScale(
            equalised,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )
    except cv2.error as exc:
        logger.warning("  Detection failed: %s", exc)
        return None
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
    return int(x), int(y), int(w), int(h)


def _progress(prefix, current, total, width=30):
    # type: (str, int, int, int) -> None
    if total <= 0:
        return
    frac = min(1.0, current / float(total))
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write("\r  {} [{}] {}/{}".format(prefix, bar, current, total))
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def train():
    # type: () -> int
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

    faces_all  = []  # type: List[np.ndarray]
    labels_all = []  # type: List[int]
    label_map  = {}  # type: Dict[int, str]
    stats      = []  # type: List[Tuple[str, int, int, int]]

    started = time.time()
    next_label = 0
    for profile_dir in profiles:
        profile_name = profile_dir.name
        images = sorted([f for f in profile_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])

        if not images:
            logger.warning("No images in %s — skipping", profile_dir)
            stats.append((profile_name, 0, 0, 0))
            continue

        label = next_label
        next_label += 1
        label_map[label] = profile_name

        augs_n = _augs_per_image(len(images))
        logger.info(
            "Processing '%s' (%d image(s), %d aug(s)/image → target %d sample(s))...",
            profile_name, len(images), augs_n, len(images) * augs_n,
        )
        used = 0
        skipped = 0
        total = len(images)

        for idx, img_path in enumerate(images, start=1):
            gray = _load_gray_image(img_path)
            if gray is None:
                skipped += 1
                _progress(profile_name, idx, total)
                continue

            face = _detect_largest_face(detector, gray)
            if face is None:
                skipped += 1
                _progress(profile_name, idx, total)
                continue

            x, y, w, h = face
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                skipped += 1
                _progress(profile_name, idx, total)
                continue
            try:
                crop = cv2.resize(crop, FACE_SIZE)
            except cv2.error as exc:
                logger.warning("  Resize failed on %s: %s", img_path.name, exc)
                skipped += 1
                _progress(profile_name, idx, total)
                continue

            for aug in _augmentation_pool(crop)[:augs_n]:
                faces_all.append(aug)
                labels_all.append(label)
            used += 1
            _progress(profile_name, idx, total)

        profile_total_samples = used * augs_n
        if profile_total_samples < MIN_SAMPLES_PER_PROFILE and used > 0:
            logger.warning(
                "  '%s' produced only %d sample(s) (target %d). Add more photos.",
                profile_name, profile_total_samples, MIN_SAMPLES_PER_PROFILE,
            )
        logger.info(
            "  '%s' done: %d photo(s) used, %d skipped → %d sample(s) after aug",
            profile_name, used, skipped, profile_total_samples,
        )
        stats.append((profile_name, used, skipped, profile_total_samples))

    if not faces_all:
        logger.error("No usable face samples found. Check your photos.")
        return 1

    payload = {
        "faces": faces_all,
        "labels": labels_all,
        "label_map": label_map,
        "face_size": FACE_SIZE,
        "version": 1,
    }
    try:
        with open(str(FACE_DATA_PATH), "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        logger.error("Failed to write %s: %s", FACE_DATA_PATH, exc)
        return 1

    elapsed = time.time() - started
    logger.info(
        "Saved %d sample(s), %d profile(s) to %s (%.1fs)",
        len(faces_all), len(label_map), FACE_DATA_PATH, elapsed,
    )
    _print_summary(stats)
    return 0


def _print_summary(stats):
    # type: (List[Tuple[str, int, int, int]]) -> None
    name_w = max(7, max((len(s[0]) for s in stats), default=7))
    header = "{name:<{w}}  {used:>11}  {skipped:>14}  {samples:>14}".format(
        name="profile", w=name_w,
        used="photos used", skipped="photos skipped", samples="samples (aug)",
    )
    print()
    print(header)
    print("-" * len(header))
    for name, used, skipped, samples in stats:
        print("{name:<{w}}  {used:>11d}  {skipped:>14d}  {samples:>14d}".format(
            name=name, w=name_w, used=used, skipped=skipped, samples=samples,
        ))
    print()


if __name__ == "__main__":
    sys.exit(train())
