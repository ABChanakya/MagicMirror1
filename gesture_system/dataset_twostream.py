#!/usr/bin/env python3
"""
dataset_twostream.py — landmarks from cache, frames decoded from the .mp4s.

The video branch needs pixels, but caching them is what forced an external drive
last time: 932 clips x 30 frames x 3 x 112 x 112 is ~10 GB as uint8, against
5.7 GB free. The source videos are already on disk at 628 MB, so frames are
decoded on demand in the DataLoader workers instead (~29 ms/clip, which overlaps
training and is not the bottleneck for a model this size).

Augmentation is applied consistently across both streams: a horizontal flip
mirrors the frames AND negates the landmark x components AND swaps the
left/right label. Letting those disagree would train the video branch on a
mirrored clip carrying the unmirrored label.

Two appearance strengths:

  light   mild brightness/contrast — the original setting
  strong  temporally-consistent random-resized crop, full colour jitter,
          grayscale, and cutout

`strong` exists to test one specific question: the video branch scores 22.9%
balanced on a held-out recording session (below the 33% chance line) because it
learns the session's room and clothing rather than the gesture. If aggressively
decorrelating appearance from the label recovers that, generated appearance
variants would help more. If it does not, they will not either.

Every appearance transform is drawn ONCE PER CLIP, not per frame. A crop that
jittered frame to frame would destroy the very motion the model is meant to read.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from train_landmark import aug_rotate, aug_time_warp, aug_traj_scale

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def decode_clip(path, num_frames=30, size=112, normalise=True):
    """
    Read a video -> (T, H, W, 3) float32 in [0,1], or (T,3,H,W) ImageNet-normalised
    when `normalise` is True. Augmentation runs on the [0,1] form.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = np.linspace(0, max(total - 1, 0), num_frames).astype(int) if total > 0 else None

    frames = []
    if idx is not None:
        want, got, i = set(idx.tolist()), {}, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = fr
            i += 1
        frames = [got[j] for j in idx if j in got]
    else:
        while len(frames) < num_frames:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
    cap.release()

    if not frames:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]

    out = np.empty((num_frames, size, size, 3), np.float32)
    for t, fr in enumerate(frames):
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        out[t] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return _to_tensor_layout(out) if normalise else out


def _to_tensor_layout(rgb01):
    """(T,H,W,3) in [0,1] -> (T,3,H,W) ImageNet-normalised."""
    x = (rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ── Appearance augmentation, drawn once per clip ──────────────────────────────

def _strong_appearance(rgb01, rng):
    T, H, W, _ = rgb01.shape
    out = rgb01

    # random resized crop — same window for every frame, so motion survives
    if rng.random() < 0.8:
        scale = rng.uniform(0.65, 1.0)
        ar = rng.uniform(0.9, 1.1)
        ch = int(round(H * scale)),
        ch = max(8, min(H, int(round(H * scale))))
        cw = max(8, min(W, int(round(W * scale * ar))))
        y0 = rng.integers(0, H - ch + 1)
        x0 = rng.integers(0, W - cw + 1)
        cropped = out[:, y0:y0 + ch, x0:x0 + cw, :]
        out = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
                        for f in cropped])

    # colour jitter — brightness, contrast, saturation, hue
    if rng.random() < 0.9:
        b = rng.uniform(-0.30, 0.30)
        c = rng.uniform(0.60, 1.45)
        mean = out.mean()
        out = np.clip((out - mean) * c + mean + b, 0.0, 1.0)
    if rng.random() < 0.7:
        s = rng.uniform(0.3, 1.6)
        gray = out @ np.array([0.299, 0.587, 0.114], np.float32)
        out = np.clip(gray[..., None] + (out - gray[..., None]) * s, 0.0, 1.0)
    if rng.random() < 0.5:
        shift = np.float32(rng.uniform(-0.08, 0.08))
        hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in out])
        hsv[..., 0] = (hsv[..., 0] + shift * 180.0) % 180.0
        out = np.clip(np.stack([cv2.cvtColor(f, cv2.COLOR_HSV2RGB) for f in hsv]), 0, 1)

    # full grayscale — strips colour as an identity cue entirely
    if rng.random() < 0.25:
        gray = out @ np.array([0.299, 0.587, 0.114], np.float32)
        out = np.repeat(gray[..., None], 3, axis=-1)

    # cutout — same box each frame, so it hides a region rather than flickering
    if rng.random() < 0.5:
        bh = int(H * rng.uniform(0.15, 0.35))
        bw = int(W * rng.uniform(0.15, 0.35))
        y0 = rng.integers(0, max(1, H - bh))
        x0 = rng.integers(0, max(1, W - bw))
        out = out.copy()
        out[:, y0:y0 + bh, x0:x0 + bw, :] = rng.random()

    if rng.random() < 0.4:
        out = np.clip(out + rng.normal(0, 0.03, out.shape).astype(np.float32), 0, 1)

    return out


def _light_appearance(rgb01, rng):
    if rng.random() < 0.5:
        b = rng.uniform(-0.10, 0.10)
        c = rng.uniform(0.9, 1.1)
        mean = rgb01.mean()
        rgb01 = np.clip((rgb01 - mean) * c + mean + b, 0.0, 1.0)
    return rgb01


class TwoStreamDataset(Dataset):
    """Yields (landmarks (T,174), frames (3,T,H,W), label)."""

    def __init__(self, X, y, rel_paths, video_root, flip_map,
                 train=False, num_frames=30, size=112, n_shape=168,
                 use_video=True, appearance='light', seed=0):
        self.X, self.y = X, y
        self.paths = rel_paths
        self.root = Path(video_root)
        self.flip_map = flip_map
        self.train = train
        self.NF, self.size = num_frames, size
        self.n_shape = n_shape
        self.use_video = use_video
        self.appearance = appearance
        self.seed = seed

    def __len__(self):
        return len(self.y)

    def _augment_landmarks(self, lm, label):
        # Matches the landmark-only pipeline. The two-stream path previously ran
        # only flip+noise, which handicapped the landmark branch in the
        # like-for-like comparison against video.
        if np.random.rand() < 0.5:
            lm = lm.copy()
            lm[:, 0::3] = -lm[:, 0::3]
            label = self.flip_map.get(int(label), int(label))
            flipped = True
        else:
            flipped = False
        if np.random.rand() < 0.7:
            lm = aug_time_warp(lm)
        if np.random.rand() < 0.5:
            lm = aug_rotate(lm, max_deg=8.0)
        if np.random.rand() < 0.5:
            lm = aug_traj_scale(lm)
        if np.random.rand() < 0.8:
            lm = lm + np.random.randn(*lm.shape).astype(np.float32) * 0.02
        return lm.astype(np.float32), label, flipped

    def __getitem__(self, i):
        lm = self.X[i]
        label = int(self.y[i])
        rng = np.random.default_rng(abs(hash((self.seed, i, np.random.randint(1 << 30)))) % (1 << 32))

        frames01 = None
        if self.use_video:
            frames01 = decode_clip(self.root / self.paths[i], self.NF,
                                   self.size, normalise=False)
            if frames01 is None:
                frames01 = np.zeros((self.NF, self.size, self.size, 3), np.float32)

        if self.train:
            lm, label, flipped = self._augment_landmarks(lm, label)
            if frames01 is not None:
                if flipped:
                    frames01 = frames01[:, :, ::-1, :].copy()   # mirror width
                frames01 = (_strong_appearance(frames01, rng)
                            if self.appearance == 'strong'
                            else _light_appearance(frames01, rng))

        lm_t = torch.from_numpy(np.ascontiguousarray(lm))
        if not self.use_video:
            return lm_t, torch.zeros(1), label
        fr = _to_tensor_layout(frames01)              # (T,3,H,W)
        return lm_t, torch.from_numpy(fr).permute(1, 0, 2, 3), label
