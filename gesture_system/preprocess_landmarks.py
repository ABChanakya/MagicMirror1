#!/usr/bin/env python3
"""
preprocess_landmarks.py — extract MediaPipe landmarks into a single .npz cache.

No video frames are stored. Each clip becomes a (30, 66) float32 array:
    [0:63]  hand shape      — 21 joints, wrist-relative, hand-scale normalised
    [63:66] wrist trajectory — displacement from frame 0, in hand-size units

The whole dataset is ~4 MB, so training loads it entirely into RAM and
augments on the fly. That removes the need for pre-augmented files on disk
(and with them the train/val leak that pre-augmentation kept reintroducing).

Usage
-----
python preprocess_landmarks.py                    # writes data/landmarks.npz
python preprocess_landmarks.py --out other.npz
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from dataset import (
    _extract_frames,
    _extract_landmarks_from_frame,
    _init_mediapipe,
    _interpolate_missing_landmarks,
    _normalise_landmarks,
)

_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--out', default='data/landmarks.npz')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg   = cfg['data']
    raw_dir    = Path(data_cfg['raw_dir'])
    num_frames = data_cfg['num_frames']
    classes    = [str(c) for c in data_cfg['classes']]

    hands = _init_mediapipe()

    X, y, clip_ids = [], [], []
    # Per-class tally of how many frames MediaPipe found no hand in. High
    # numbers mean interpolation is inventing most of the trajectory.
    miss_stats = {c: [0, 0] for c in classes}   # class -> [missed_frames, total_frames]
    failed = []

    for label, cls in enumerate(classes):
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"[preprocess] {cls_dir} missing — skipping")
            continue

        videos = sorted(p for p in cls_dir.iterdir()
                        if p.suffix.lower() in _VIDEO_EXTENSIONS)
        if not videos:
            print(f"[preprocess] no videos in {cls_dir}")
            continue

        for vid in tqdm(videos, desc=f"{cls:12}"):
            frames_np = _extract_frames(str(vid), num_frames=num_frames)
            if frames_np is None:
                failed.append(str(vid))
                continue

            lm_seq = np.full((num_frames, 63), np.nan, dtype=np.float32)
            missed = 0
            for t, frame in enumerate(frames_np):
                lm = _extract_landmarks_from_frame(hands, frame)
                if lm is not None:
                    lm_seq[t] = lm
                else:
                    missed += 1

            miss_stats[cls][0] += missed
            miss_stats[cls][1] += num_frames

            # A clip where the hand was never found carries no signal at all
            if missed == num_frames:
                failed.append(f"{vid} (no hand detected in any frame)")
                continue

            lm_seq = _interpolate_missing_landmarks(lm_seq)
            lm_seq = _normalise_landmarks(lm_seq)      # (30, 66)

            X.append(lm_seq)
            y.append(label)
            # clip_id keeps class+stem so splitting can never put two views of
            # the same recording on both sides of the split
            clip_ids.append(f"{cls}/{vid.stem}")

    if not X:
        print("[preprocess] No samples produced — nothing written.")
        return

    X = np.stack(X).astype(np.float32)          # (N, 30, 66)
    y = np.asarray(y, dtype=np.int64)           # (N,)
    clip_ids = np.asarray(clip_ids)             # (N,)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, clip_ids=clip_ids, classes=np.asarray(classes))

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[preprocess] Wrote {out_path}  ({size_mb:.1f} MB)")
    print(f"[preprocess] X={X.shape}  y={y.shape}")
    print("\n  Per-class counts and MediaPipe detection quality:")
    for label, cls in enumerate(classes):
        n = int((y == label).sum())
        miss, total = miss_stats[cls]
        pct = (miss / total * 100) if total else 0.0
        flag = "  <-- high, check lighting" if pct > 20 else ""
        print(f"    {cls:12} {n:4} clips   frames with no hand: {pct:5.1f}%{flag}")

    if failed:
        print(f"\n  {len(failed)} clip(s) dropped:")
        for f in failed[:10]:
            print(f"    {f}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")


if __name__ == '__main__':
    main()
