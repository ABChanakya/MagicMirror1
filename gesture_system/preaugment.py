#!/usr/bin/env python3
"""
preaugment.py — Offline augmentation.

Loads every .pt file from data/processed, generates augmented copies,
and saves them alongside the originals. Training then loads everything
as a plain dataset with NO online augmentation.

Each original gets these fixed augmentations:
  1. horizontal_flip
  2. color_jitter (mild)
  3. color_jitter (strong)
  4. temporal_jitter (slow 0.7x)
  5. temporal_jitter (fast 1.3x)
  6. landmark_noise
  7. rotation (+3°)
  8. rotation (-3°)
  9. frame_dropout
 10. flip + color_jitter + noise   (combo)
 11. flip + temporal_jitter + noise (combo)

Result: 12× samples per original (1 original + 11 augmented)
~498 originals × 12 = ~5976 total samples
"""

import random
import shutil
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

from augmentation import (
    horizontal_flip, color_jitter, temporal_jitter,
    landmark_noise, rotation, frame_dropout, grayscale,
    SWIPE_LEFT, SWIPE_RIGHT,
)

PROCESSED_DIR = Path("/media/bhaskara/Volume/Data_Gesture/raw")
AUGMENTED_DIR = Path("/media/bhaskara/Volume/Data_Gesture/augmented")
ROTATION_MAX  = 3.0   # hard limit — ±3° only


def augment_sample(lm, fr, label):
    """Returns list of (lm, fr, label) augmented variants."""
    variants = []

    # 1. Horizontal flip
    lm1, fr1, lb1 = horizontal_flip(lm, fr, label)
    variants.append((lm1, fr1, lb1))

    # 2. Color jitter mild
    variants.append((lm.clone(), color_jitter(fr, brightness=0.2, contrast=0.1, saturation=0.1, hue=0.03), label))

    # 3. Color jitter strong
    variants.append((lm.clone(), color_jitter(fr, brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05), label))

    # 4. Temporal jitter slow
    lm4, fr4 = temporal_jitter(lm, fr, speed_range=(0.7, 0.75))
    variants.append((lm4, fr4, label))

    # 5. Temporal jitter fast
    lm5, fr5 = temporal_jitter(lm, fr, speed_range=(1.25, 1.3))
    variants.append((lm5, fr5, label))

    # 6. Landmark noise
    variants.append((landmark_noise(lm, std=0.01), fr.clone(), label))

    # 7. Rotation +3°
    import math
    lm7, fr7 = rotation(lm, fr, max_degrees=ROTATION_MAX)
    variants.append((lm7, fr7, label))

    # 8. Rotation -3° (force negative by calling rotation with a fixed negative seed)
    random.seed(0)  # force negative angle
    lm8, fr8 = rotation(lm, fr, max_degrees=ROTATION_MAX)
    random.seed()   # restore random state
    variants.append((lm8, fr8, label))

    # 9. Frame dropout
    lm9, fr9 = frame_dropout(lm, fr, max_dropout=2)
    variants.append((lm9, fr9, label))

    # 10. Combo: flip + color jitter + noise
    lm10, fr10, lb10 = horizontal_flip(lm, fr, label)
    fr10 = color_jitter(fr10, brightness=0.3, contrast=0.2, saturation=0.2, hue=0.04)
    lm10 = landmark_noise(lm10, std=0.008)
    variants.append((lm10, fr10, lb10))

    # 11. Combo: flip + temporal jitter + noise
    lm11, fr11, lb11 = horizontal_flip(lm, fr, label)
    lm11, fr11 = temporal_jitter(lm11, fr11, speed_range=(0.8, 1.2))
    lm11 = landmark_noise(lm11, std=0.008)
    variants.append((lm11, fr11, lb11))

    # 12. Grayscale
    variants.append((lm.clone(), grayscale(fr), label))

    # 13. Grayscale + flip
    lm13, fr13, lb13 = horizontal_flip(lm, fr, label)
    variants.append((lm13, grayscale(fr13), lb13))

    # 14. Grayscale + color jitter (simulates very poor/harsh lighting)
    variants.append((lm.clone(), color_jitter(grayscale(fr), brightness=0.4, contrast=0.3, saturation=0.0, hue=0.0), label))

    return variants


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["data"]["classes"]

    # Find all original .pt files (not augmented ones)
    orig_files = [p for p in sorted(PROCESSED_DIR.rglob("*.pt"))
                  if "_aug" not in p.stem]

    print(f"\n{'='*60}")
    print(f"  Offline Augmentation")
    print(f"  Found {len(orig_files)} original samples")
    print(f"  Generating 11 augmented variants each → {len(orig_files)*12} total")
    print(f"  Rotation hard limit: ±{ROTATION_MAX}°")
    print(f"{'='*60}\n")

    # Set up augmented output dirs mirroring raw structure
    for cls in classes:
        (AUGMENTED_DIR / str(cls)).mkdir(parents=True, exist_ok=True)

    # Remove old augmented files
    aug_files = list(AUGMENTED_DIR.rglob("*.pt"))
    if aug_files:
        print(f"  Removing {len(aug_files)} old augmented files...")
        for f in aug_files:
            f.unlink()

    skipped = 0
    saved = 0

    for pt_path in tqdm(orig_files, desc="Augmenting"):
        data   = torch.load(str(pt_path), weights_only=True)
        lm     = data["landmarks"]   # (30, 63)
        fr     = data["frames"]      # (30, 3, H, W)
        label  = int(data["label"])

        variants = augment_sample(lm, fr, label)

        # Determine class subdir from label
        cls_name = classes[label]
        out_dir  = AUGMENTED_DIR / str(cls_name)

        for i, (aug_lm, aug_fr, aug_label) in enumerate(variants):
            out_path = out_dir / f"{pt_path.stem}_aug{i+1:02d}.pt"
            torch.save({
                "landmarks": aug_lm,
                "frames":    aug_fr,
                "label":     aug_label,
            }, str(out_path))
            saved += 1

    print(f"\n  ✅ Saved {saved} augmented samples")
    print(f"  Total dataset: {len(orig_files) + saved} samples")
    print()

    # Per-class summary
    print("  Per-class counts (raw + augmented):")
    for cls in classes:
        raw_count = len(list((PROCESSED_DIR / str(cls)).glob("*.pt")))
        aug_count = len(list((AUGMENTED_DIR / str(cls)).glob("*.pt")))
        print(f"    {str(cls):15} {raw_count:4} raw + {aug_count:5} aug = {raw_count+aug_count:5} total")

    print(f"\n  Run training: python train.py (augmentation disabled — all data pre-computed)")


if __name__ == "__main__":
    main()
