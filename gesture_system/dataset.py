"""
Dataset preprocessing and loading for the gesture recognition system.

preprocess_all(config)
    Walks data/raw/<class>/ for video files, extracts 30 uniformly-sampled
    frames, runs MediaPipe Hands, normalises landmarks, and saves .pt files
    to data/processed/.

GestureDataset(torch.utils.data.Dataset)
    Loads .pt files with optional online augmentation.
"""

import os
import glob
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[dataset] Warning: mediapipe not installed — preprocessing unavailable.")


# ImageNet normalisation constants
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_frames(video_path: str, num_frames: int = 30) -> Optional[np.ndarray]:
    """
    Uniformly sample exactly `num_frames` frames from a video file.

    Returns
    -------
    np.ndarray of shape (num_frames, H, W, 3) in RGB uint8, or None on error.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    # Uniform indices in [0, total-1]
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            # Use last valid frame or zeros
            if frames:
                frames.append(frames[-1].copy())
            else:
                # Determine frame size from cap properties
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 224
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W, 3)


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark extraction
# ──────────────────────────────────────────────────────────────────────────────

def _init_mediapipe():
    """Return a configured MediaPipe Hands solution object."""
    assert _MP_AVAILABLE, "mediapipe is not installed"
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _extract_landmarks_from_frame(
    hands_model,
    frame_rgb: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Run MediaPipe on one RGB frame.

    Returns np.ndarray of shape (63,) with [x0,y0,z0, x1,y1,z1, ...]
    or None if no hand is detected.
    """
    results = hands_model.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21, 3)
    return coords.flatten()  # (63,)


def _interpolate_missing_landmarks(lm_seq: np.ndarray) -> np.ndarray:
    """
    lm_seq: (T, 63) with np.nan where hand was not detected.
    Fill NaN frames via linear interpolation from neighbours.
    Remaining NaNs (entire clip) → zeros.
    """
    T, D = lm_seq.shape
    for d in range(D):
        col = lm_seq[:, d]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        good = np.where(~nan_mask)[0]
        if len(good) == 0:
            col[:] = 0.0
        else:
            col[nan_mask] = np.interp(
                np.where(nan_mask)[0], good, col[good]
            )
        lm_seq[:, d] = col
    return lm_seq


def _normalise_landmarks(lm_seq: np.ndarray) -> np.ndarray:
    """
    Normalise a (T, 63) landmark sequence into a (T, 66) two-stream encoding.

    Output layout per frame:
      [0:63]  SHAPE      — 21 joints, wrist-relative, hand-scale normalised
      [63:66] TRAJECTORY — wrist displacement from frame 0, in hand-size units

    Why two streams: subtracting the wrist per frame (the shape stream) makes
    hand pose translation-invariant, but it also sets the wrist to (0,0,0) in
    every frame — which deletes the swipe entirely. swipe_left, swipe_right and
    swipe_up all collapse to identical tensors. The trajectory stream carries
    that displacement back, normalised by hand size so it does not depend on
    how far the person stands from the camera or how big their hand is, and
    measured relative to frame 0 so it does not depend on where in the frame
    the gesture happened.

    Layout note: the trajectory is stored contiguously as (x, y, z) at indices
    63, 64, 65. Because 63 % 3 == 0, the `0::3` / `1::3` / `2::3` striding used
    by horizontal_flip() and rotation() in augmentation.py picks up the
    trajectory channels automatically. Do not reorder these three values.
    """
    lm_seq = lm_seq.copy()
    T = lm_seq.shape[0]

    # Reshape to (T, 21, 3) for easier indexing
    lm_3d = lm_seq.reshape(T, 21, 3)

    # Keep the wrist track before it is subtracted away — this is the swipe signal
    wrist_track = lm_3d[:, 0, :].copy()   # (T, 3)

    # Step 1: subtract wrist (landmark 0) → translation-invariant hand shape
    wrist = lm_3d[:, 0:1, :]   # (T, 1, 3)
    lm_3d = lm_3d - wrist

    # Step 2: compute max pairwise distance across the clip (x,y only).
    # Measured on wrist-relative coords, so this is hand size, not swipe extent.
    xy = lm_3d[:, :, :2].reshape(-1, 2)  # (T*21, 2)
    # Use the range (max - min) along each axis as a proxy for max pairwise dist
    span_x = xy[:, 0].max() - xy[:, 0].min()
    span_y = xy[:, 1].max() - xy[:, 1].min()
    scale  = max(float(max(span_x, span_y)), 1e-6)
    lm_3d  = lm_3d / scale

    # Step 3: wrist displacement from frame 0, in the same hand-size units
    trajectory = (wrist_track - wrist_track[0:1]) / scale   # (T, 3)

    return np.concatenate(
        [lm_3d.reshape(T, 63), trajectory], axis=1
    ).astype(np.float32)   # (T, 66)


# ──────────────────────────────────────────────────────────────────────────────
# Frame pre-processing
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_frames(
    frames_np: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Convert (T, H, W, 3) uint8 RGB numpy array to
    (T, 3, image_size, image_size) float32 tensor, ImageNet-normalised.
    """
    result = []
    for frame in frames_np:
        # HWC uint8 → CHW float32 in [0,1]
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        # Resize to image_size × image_size
        t = TF.resize(t, [image_size, image_size], antialias=True)
        # ImageNet normalise
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
        result.append(t)
    return torch.stack(result, dim=0)  # (T, 3, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────

_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}


def preprocess_all(config: Dict[str, Any]) -> None:
    """
    Process every video in data/raw/<class>/ and save .pt files to
    data/processed/<class>/<stem>.pt.

    Each .pt file contains:
        {
            "landmarks": Tensor(30, 63),      # normalised
            "frames":    Tensor(30, 3, 224, 224),  # ImageNet-normalised
            "label":     int,
        }

    Skips files that already exist (re-run safe).
    """
    data_cfg   = config['data']
    raw_dir    = Path(data_cfg['raw_dir'])
    proc_dir   = Path(data_cfg['processed_dir'])
    num_frames = data_cfg['num_frames']
    image_size = data_cfg['image_size']
    classes    = data_cfg['classes']

    hands = _init_mediapipe()
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    for class_idx, class_name in enumerate(classes):
        class_raw_dir  = raw_dir  / class_name
        class_proc_dir = proc_dir / class_name
        class_proc_dir.mkdir(parents=True, exist_ok=True)

        if not class_raw_dir.exists():
            print(f"[preprocess] Warning: {class_raw_dir} does not exist, skipping.")
            continue

        video_files = [
            p for p in class_raw_dir.iterdir()
            if p.suffix.lower() in _VIDEO_EXTENSIONS
        ]

        if not video_files:
            print(f"[preprocess] No videos found in {class_raw_dir}")
            continue

        print(f"[preprocess] {class_name}: {len(video_files)} videos")

        for video_path in sorted(video_files):
            out_path = class_proc_dir / (video_path.stem + '.pt')
            if out_path.exists():
                stats['skipped'] += 1
                continue

            # --- Extract frames ------------------------------------------------
            frames_np = _extract_frames(str(video_path), num_frames=num_frames)
            if frames_np is None:
                print(f"[preprocess] Error reading {video_path}")
                stats['errors'] += 1
                continue

            # --- Run MediaPipe on each frame ------------------------------------
            lm_seq = np.full((num_frames, 63), np.nan, dtype=np.float32)
            for t, frame in enumerate(frames_np):
                lm = _extract_landmarks_from_frame(hands, frame)
                if lm is not None:
                    lm_seq[t] = lm

            # --- Interpolate missing detections --------------------------------
            lm_seq = _interpolate_missing_landmarks(lm_seq)

            # --- Normalise landmarks -------------------------------------------
            lm_seq = _normalise_landmarks(lm_seq)

            # --- Pre-process frames --------------------------------------------
            frames_tensor = _preprocess_frames(frames_np, image_size=image_size)

            # --- Save ----------------------------------------------------------
            sample = {
                'landmarks': torch.from_numpy(lm_seq),  # (30, 63)
                'frames':    frames_tensor,              # (30, 3, 224, 224)
                'label':     class_idx,
            }
            torch.save(sample, str(out_path))
            stats['processed'] += 1

        print(f"[preprocess] {class_name}: done")

    hands.close()
    print(f"\n[preprocess] Summary: {stats}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    """
    Loads preprocessed .pt files from data/processed/.

    Parameters
    ----------
    processed_dir : str | Path
        Root of the processed data tree (contains one subdir per class).
    classes : list[str]
        Ordered list of class names (must match config).
    augmentation_pipeline : AugmentationPipeline | None
        If provided, applied online during __getitem__.
    file_list : list[Path] | None
        Pre-computed list of .pt files (for train/val/test splits).
        If None, all .pt files under processed_dir are used.
    """

    def __init__(
        self,
        processed_dir,
        classes: List[str],
        augmentation_pipeline=None,
        file_list: Optional[List[Path]] = None,
    ):
        self.classes   = classes
        self.augment   = augmentation_pipeline

        processed_dir = Path(processed_dir)
        if file_list is not None:
            self.samples = list(file_list)
        else:
            self.samples = sorted(processed_dir.rglob('*.pt'))

        if len(self.samples) == 0:
            print(f"[GestureDataset] Warning: no .pt files found under {processed_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path = self.samples[idx]
        data = torch.load(str(path), weights_only=True)

        landmarks: torch.Tensor = data['landmarks'].float()   # (30, 63)
        frames:    torch.Tensor = data['frames'].float()       # (30, 3, 224, 224)
        label:     int          = int(data['label'])

        if self.augment is not None:
            landmarks, frames, label = self.augment(landmarks, frames, label)

        # Reorder frames to (3, 30, 224, 224) for Video Swin: (C, T, H, W)
        frames_5d = frames.permute(1, 0, 2, 3)  # (3, 30, 224, 224)

        return landmarks, frames_5d, label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_labels(self) -> List[int]:
        """Return all labels (used for stratified splitting)."""
        labels = []
        for path in self.samples:
            data = torch.load(str(path), weights_only=True)
            labels.append(int(data['label']))
        return labels


def build_split_datasets(
    config: Dict[str, Any],
    augmentation_pipeline=None,
) -> Tuple[GestureDataset, GestureDataset, GestureDataset]:
    """
    Build stratified 80/10/10 train/val/test splits.

    Returns
    -------
    train_dataset, val_dataset, test_dataset
    """
    from sklearn.model_selection import train_test_split  # local import

    proc_dir = Path(config['data']['processed_dir'])
    aug_dir  = Path(config['data'].get('augmented_dir', ''))
    classes  = config['data']['classes']

    # Collect raw files — these go into val/test splits only (clean data)
    raw_files = sorted(proc_dir.rglob('*.pt'))

    # Collect augmented files — these only go into train split
    aug_files = sorted(aug_dir.rglob('*.pt')) if aug_dir.exists() else []

    print(f"[dataset] Raw: {len(raw_files)} | Augmented: {len(aug_files)}")

    # Labels for raw files
    all_labels = []
    for p in raw_files:
        d = torch.load(str(p), weights_only=True)
        all_labels.append(int(d['label']))

    # Split RAW originals first — val/test are genuinely unseen originals only
    train_raw, temp_files, _, temp_labels = train_test_split(
        raw_files, all_labels,
        test_size=0.2, random_state=42, stratify=all_labels,
    )
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5, random_state=42, stratify=temp_labels,
    )

    # Only include augmented versions of TRAINING originals — never val/test originals
    # Augmented filenames follow pattern: {original_stem}_aug{N}.pt
    # Key: use (class_dir, original_stem) to avoid cross-class stem collisions
    # e.g. swipe_left/01 and swipe_right/01 are different originals
    train_keys = {(p.parent.name, p.stem) for p in train_raw}
    valid_aug_files = [
        p for p in aug_files
        if '_aug' in p.stem and (p.parent.name, p.stem.rsplit('_aug', 1)[0]) in train_keys
    ]
    leaked = len(aug_files) - len(valid_aug_files)
    if leaked:
        print(f"[dataset] Removed {leaked} augmented files derived from val/test originals")

    train_files = list(train_raw) + valid_aug_files

    train_ds = GestureDataset(proc_dir, classes, augmentation_pipeline, file_list=train_files)
    val_ds   = GestureDataset(proc_dir, classes, augmentation_pipeline=None, file_list=val_files)
    test_ds  = GestureDataset(proc_dir, classes, augmentation_pipeline=None, file_list=test_files)
    print(f"[dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    import yaml
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    preprocess_all(cfg)
