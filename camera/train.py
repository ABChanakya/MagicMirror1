"""
train.py — Build face embeddings from dataset/ using InsightFace (GPU)

Usage:
  python3 train.py [--output path/to/encodings.pkl]

Expects:
  camera/dataset/
    your_face/        <- folder name = profile name
      photo1.jpg
      photo2.jpg
    mama/
      photo1.jpg

Saves to camera/model/encodings.pkl (or --output path).
Run again whenever you add new photos.
"""

import argparse
import logging
import os
import pickle
import site
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def _patch_cuda_libs():
    for sp in site.getsitepackages():
        nvidia_dir = Path(sp) / "nvidia"
        if nvidia_dir.exists():
            dirs = [str(p) for p in nvidia_dir.glob("*/lib")]
            if dirs:
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + (":" + existing if existing else "")


_patch_cuda_libs()

DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "encodings.pkl"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Train face embeddings for MagicMirror3")
    p.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Output path for encodings.pkl (default: camera/model/encodings.pkl)")
    return p.parse_args()


def train(model_path: Path):
    model_path.parent.mkdir(exist_ok=True)

    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using GPU (CUDA)")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("Using CPU")

    app = FaceAnalysis(name="buffalo_sc", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings: list = []
    names:      list[str] = []

    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        logger.error("No images in %s — add face photos first.", DATASET_DIR)
        return

    profiles = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not profiles:
        logger.error("No profile folders in %s", DATASET_DIR)
        return

    # Summary counters per profile
    summary: list[tuple[str, int, int]] = []  # (name, encoded, skipped)

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
                rgb     = np.array(pil_img)
                bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning("  Cannot open %s: %s", img_path.name, e)
                skipped += 1
                continue

            faces = app.get(bgr)
            if not faces:
                logger.warning("  No face in %s — skipping", img_path.name)
                skipped += 1
                continue

            if len(faces) > 1:
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True,
                )
                logger.warning("  Multiple faces in %s — using largest", img_path.name)

            embeddings.append(faces[0].normed_embedding.tolist())
            names.append(profile_name)
            encoded += 1

        logger.info("  Added %d embedding(s) for '%s'", encoded, profile_name)
        summary.append((profile_name, encoded, skipped))

    if not embeddings:
        logger.error("No valid embeddings produced. Check your photos.")
        return

    with open(model_path, "wb") as f:
        pickle.dump({"encodings": embeddings, "names": names}, f)

    logger.info("Saved %d embedding(s) for %d profile(s) → %s",
                len(embeddings), len(set(names)), model_path)

    # Summary table
    print("\n── Training Summary ────────────────────────────────")
    print(f"{'Profile':<20} {'Encoded':>8} {'Skipped':>8}")
    print("─" * 40)
    for name, enc, skip in summary:
        print(f"{name:<20} {enc:>8} {skip:>8}")
    print("─" * 40)
    print(f"{'TOTAL':<20} {sum(e for _,e,_ in summary):>8} {sum(s for _,_,s in summary):>8}")


if __name__ == "__main__":
    args = parse_args()
    train(args.output)
