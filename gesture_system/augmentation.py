"""
Data augmentation for gesture clips.

Each augmentation function operates on:
    landmarks : torch.Tensor  (30, 63)    — normalised landmark sequence
    frames    : torch.Tensor  (30, 3, H, W) — normalised video frames

Returns the same types, possibly with a modified label.

AugmentationPipeline applies the enabled transforms probabilistically at
training time.
"""

import math
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────────────────────────────────────
# Label constants (must match config.yaml class order)
# ──────────────────────────────────────────────────────────────────────────────

# Indices come from config.yaml rather than being written out by hand: the
# class list is editable (dropping swipe_up/swipe_down renumbers "null"), and a
# stale constant here would mislabel every horizontally flipped clip in
# training without ever raising an error.

def _class_order():
    try:
        import yaml
        cfg = yaml.safe_load((Path(__file__).parent / 'config.yaml').read_text())
        return [str(c) for c in cfg['data']['classes']]
    except Exception:
        return ['swipe_left', 'swipe_right', 'swipe_up', 'swipe_down', 'null']


CLASSES = _class_order()


def _index_of(name):
    """Index of a class, or -1 when this config does not train that class."""
    return CLASSES.index(name) if name in CLASSES else -1


SWIPE_LEFT  = _index_of('swipe_left')
SWIPE_RIGHT = _index_of('swipe_right')
SWIPE_UP    = _index_of('swipe_up')
SWIPE_DOWN  = _index_of('swipe_down')
VOID        = _index_of('null')

# When we mirror horizontally, left ↔ right swaps; everything else keeps its
# label. Built by name so it stays correct whatever the configured class order.
_FLIP_NAME_MAP = {
    'swipe_left':  'swipe_right',
    'swipe_right': 'swipe_left',
}
_FLIP_LABEL_MAP = {
    _index_of(name): _index_of(_FLIP_NAME_MAP.get(name, name))
    for name in CLASSES
}
# Classes absent from this config collapse to -1; map it to itself so callers
# that still reference e.g. SWIPE_UP get a harmless no-op rather than KeyError.
_FLIP_LABEL_MAP.setdefault(-1, -1)


# ──────────────────────────────────────────────────────────────────────────────
# Individual augmentation functions
# ──────────────────────────────────────────────────────────────────────────────

def horizontal_flip(
    landmarks: torch.Tensor,
    frames: torch.Tensor,
    label: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Mirror frames left-right and negate the x-coordinates of landmarks.
    Swaps SWIPE_LEFT ↔ SWIPE_RIGHT; all other labels are unchanged.

    Assertion
    ---------
    After applying this function, if the input label was SWIPE_LEFT the output
    label must be SWIPE_RIGHT and vice-versa. Call `assert_flip_correctness()`
    to unit-test this.
    """
    new_label = _FLIP_LABEL_MAP[label]

    # Sanity-check that the label mapping is self-consistent for the two
    # direction classes (unit-testable without running MediaPipe).
    assert _FLIP_LABEL_MAP[SWIPE_LEFT]  == SWIPE_RIGHT, "Flip map: LEFT must map to RIGHT"
    assert _FLIP_LABEL_MAP[SWIPE_RIGHT] == SWIPE_LEFT,  "Flip map: RIGHT must map to LEFT"

    # Mirror frames: (T, 3, H, W) → flip along W dimension
    flipped_frames = torch.flip(frames, dims=[-1])  # last dim is W

    # Mirror landmark x-coordinates.
    # Landmarks are stored as (x0, y0, z0, x1, y1, z1, ...) — shape (T, 63).
    # After wrist-subtraction + scale normalisation x∈[-1, 1].
    # Mirroring: x_new = -x  (wrist stays at 0).
    flipped_lm = landmarks.clone()
    flipped_lm[:, 0::3] = -flipped_lm[:, 0::3]  # negate every x component

    return flipped_lm, flipped_frames, new_label


def assert_flip_correctness():
    """Unit-testable assertion that horizontal_flip swaps LEFT↔RIGHT labels."""
    dummy_lm = torch.zeros(30, 63)
    dummy_fr = torch.zeros(30, 3, 224, 224)

    _, _, out_left  = horizontal_flip(dummy_lm, dummy_fr, SWIPE_LEFT)
    _, _, out_right = horizontal_flip(dummy_lm, dummy_fr, SWIPE_RIGHT)
    _, _, out_up    = horizontal_flip(dummy_lm, dummy_fr, SWIPE_UP)
    _, _, out_void  = horizontal_flip(dummy_lm, dummy_fr, VOID)

    assert out_left  == SWIPE_RIGHT, f"Expected {SWIPE_RIGHT}, got {out_left}"
    assert out_right == SWIPE_LEFT,  f"Expected {SWIPE_LEFT},  got {out_right}"
    assert out_up    == SWIPE_UP,    f"SWIPE_UP should be invariant to flip"
    assert out_void  == VOID,        f"VOID should be invariant to flip"


def color_jitter(
    frames: torch.Tensor,
    brightness: float = 0.3,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05,
) -> torch.Tensor:
    """
    Apply identical ColorJitter to every frame in the clip.
    Operates on (T, 3, H, W) — values expected in [0, 1] before ImageNet norm,
    but also works on already-normalised tensors because we sample one
    transform and apply it uniformly.

    A consistent transform is sampled once and applied to all frames so
    that temporal coherence is preserved.
    """
    import torchvision.transforms as T

    jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    # Get a deterministic transform for this clip
    transform = jitter.forward  # callable, samples params each call
    # Apply identically to each frame (re-use same random seed per clip)
    seed = random.randint(0, 2**31 - 1)
    result_frames = []
    for t in range(frames.shape[0]):
        torch.manual_seed(seed)
        random.seed(seed)
        result_frames.append(transform(frames[t]))
    return torch.stack(result_frames, dim=0)


def temporal_jitter(
    landmarks: torch.Tensor,
    frames: torch.Tensor,
    speed_range: Tuple[float, float] = (0.8, 1.2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resample the clip at a random playback speed in [speed_range], then
    resize back to exactly 30 frames via linear interpolation.

    speed < 1 → slow-motion (fewer unique frames, padded by interpolation)
    speed > 1 → fast-motion (more source frames squeezed into 30)
    """
    T_out = landmarks.shape[0]  # 30
    speed = random.uniform(speed_range[0], speed_range[1])

    # Number of source frames we'll sample from
    # speed=1.0 → sample all 30; speed=0.8 → sample 24; speed=1.2 → sample 36
    src_len = max(2, round(T_out * speed))

    # Sample `src_len` indices from [0, T_out-1] (uniform)
    src_indices = np.linspace(0, T_out - 1, src_len)

    # Build resampled landmark sequence via linear interp → then resize to T_out
    lm_np = landmarks.numpy()  # (T, 63)
    # Interpolate from src_indices back to T_out evenly-spaced indices
    target_indices = np.linspace(0, src_len - 1, T_out)
    resampled_lm = np.zeros((T_out, lm_np.shape[1]), dtype=np.float32)
    for c in range(lm_np.shape[1]):
        # Sample from original at src_indices positions
        src_values = np.interp(src_indices, np.arange(T_out), lm_np[:, c])
        # Map back to T_out frames
        resampled_lm[:, c] = np.interp(target_indices, np.arange(src_len), src_values)

    # Resample frames using F.interpolate on the temporal dimension
    # frames: (T, 3, H, W) → (1, 3, T, H, W) for 3D interp
    frames_5d = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, 3, T, H, W)
    # First resample to src_len, then back to T_out
    frames_src = F.interpolate(frames_5d, size=(src_len, frames.shape[2], frames.shape[3]),
                                mode='trilinear', align_corners=False)
    frames_out = F.interpolate(frames_src, size=(T_out, frames.shape[2], frames.shape[3]),
                                mode='trilinear', align_corners=False)
    frames_out = frames_out.permute(0, 2, 1, 3, 4).squeeze(0)  # (T, 3, H, W)

    return torch.from_numpy(resampled_lm), frames_out


def landmark_noise(
    landmarks: torch.Tensor,
    std: float = 0.005,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise to landmark coordinates."""
    noise = torch.randn_like(landmarks) * std
    return landmarks + noise


def rotation(
    landmarks: torch.Tensor,
    frames: torch.Tensor,
    max_degrees: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate every frame by a random angle in [-max_degrees, +max_degrees].
    Landmark x,y coordinates are rotated by the same 2D rotation matrix.
    z coordinates (depth) are left unchanged as rotation is in-plane only.

    Assertion: abs(degrees) <= max_degrees is enforced.
    """
    degrees = random.uniform(-max_degrees, max_degrees)
    assert abs(degrees) <= 5.0, (
        f"rotation degrees {degrees:.4f} exceeds hard limit of 5.0. "
        f"max_degrees parameter was {max_degrees}."
    )

    # Rotate all frames identically (consistent transform)
    rotated_frames = []
    for t in range(frames.shape[0]):
        rotated_frames.append(TF.rotate(frames[t], angle=degrees, interpolation=TF.InterpolationMode.BILINEAR))
    rotated_frames = torch.stack(rotated_frames, dim=0)

    # Apply 2D rotation to landmark x, y pairs (z unchanged)
    rad = math.radians(degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    rotated_lm = landmarks.clone()
    # x components: indices 0, 3, 6, ... (every 3rd starting at 0)
    # y components: indices 1, 4, 7, ... (every 3rd starting at 1)
    x = landmarks[:, 0::3].clone()  # (T, 21)
    y = landmarks[:, 1::3].clone()  # (T, 21)

    rotated_lm[:, 0::3] = cos_a * x - sin_a * y
    rotated_lm[:, 1::3] = sin_a * x + cos_a * y
    # z (index 2::3) untouched

    return rotated_lm, rotated_frames


def grayscale(frames: torch.Tensor) -> torch.Tensor:
    """Convert all frames to grayscale (but keep 3 channels for model compatibility)."""
    # Luminance weights: R=0.299, G=0.587, B=0.114
    gray = (0.299 * frames[:, 0:1] +
            0.587 * frames[:, 1:2] +
            0.114 * frames[:, 2:3])
    return gray.expand_as(frames)


def frame_dropout(
    landmarks: torch.Tensor,
    frames: torch.Tensor,
    max_dropout: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly zero out 1..max_dropout frames:
    - Frames are replaced with black (zero) tensors.
    - Landmark values for dropped frames are linearly interpolated from
      the nearest non-dropped neighbours.
    """
    T = frames.shape[0]
    n_drop = random.randint(1, max(1, max_dropout))
    drop_indices = sorted(random.sample(range(T), min(n_drop, T)))

    augmented_frames = frames.clone()
    augmented_lm     = landmarks.clone()

    # Black out frames
    for idx in drop_indices:
        augmented_frames[idx] = 0.0

    # Interpolate landmarks for dropped frames
    all_indices = list(range(T))
    good_indices = [i for i in all_indices if i not in drop_indices]

    if len(good_indices) == 0:
        # Edge case: all frames dropped — just zero landmarks too
        augmented_lm[drop_indices] = 0.0
    else:
        for idx in drop_indices:
            # Find bounding good neighbours
            left_candidates  = [g for g in good_indices if g < idx]
            right_candidates = [g for g in good_indices if g > idx]

            if left_candidates and right_candidates:
                left_idx  = left_candidates[-1]
                right_idx = right_candidates[0]
                alpha = (idx - left_idx) / (right_idx - left_idx)
                augmented_lm[idx] = (
                    (1 - alpha) * augmented_lm[left_idx] +
                    alpha       * augmented_lm[right_idx]
                )
            elif left_candidates:
                augmented_lm[idx] = augmented_lm[left_candidates[-1]]
            else:
                augmented_lm[idx] = augmented_lm[right_candidates[0]]

    return augmented_lm, augmented_frames


# ──────────────────────────────────────────────────────────────────────────────
# AugmentationPipeline
# ──────────────────────────────────────────────────────────────────────────────

class AugmentationPipeline:
    """
    Applies enabled augmentations to a single (landmarks, frames, label) sample.

    Each augmentation is applied independently with probability 0.5, except
    horizontal_flip which is applied with probability 0.5 only for non-VOID
    classes (flipping void is fine too but doesn't change semantics).

    Usage
    -----
    pipeline = AugmentationPipeline(config['augmentation'])
    lm, fr, label = pipeline(landmarks, frames, label)
    """

    def __init__(self, aug_config: dict):
        self.cfg = aug_config

    def __call__(
        self,
        landmarks: torch.Tensor,
        frames: torch.Tensor,
        label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        # 1. Horizontal flip — p=0.5, always enabled (teaches left/right hands)
        if self.cfg.get('horizontal_flip', True) and random.random() < 0.5:
            landmarks, frames, label = horizontal_flip(landmarks, frames, label)

        # 2. Color jitter — p=0.9, strong lighting robustness
        if self.cfg.get('color_jitter', True) and random.random() < 0.9:
            frames = color_jitter(
                frames,
                brightness=self.cfg.get('color_jitter_brightness', 0.4),
                contrast=self.cfg.get('color_jitter_contrast', 0.3),
                saturation=self.cfg.get('color_jitter_saturation', 0.3),
                hue=self.cfg.get('color_jitter_hue', 0.05),
            )

        # 3. Temporal jitter — p=0.8, handles fast/slow swipers
        if self.cfg.get('temporal_jitter', True) and random.random() < 0.8:
            speed_range = tuple(self.cfg.get('temporal_jitter_range', [0.6, 1.4]))
            landmarks, frames = temporal_jitter(landmarks, frames, speed_range)

        # 4. Landmark noise — p=1.0, always add small noise (simulates tracking jitter)
        if self.cfg.get('landmark_noise_std', 0.01) > 0:
            std = self.cfg['landmark_noise_std']
            # Randomly scale noise intensity 0.5x–2x for diversity
            landmarks = landmark_noise(landmarks, std=std * random.uniform(0.5, 2.0))

        # 5. Rotation — p=0.7, small in-plane rotation
        if self.cfg.get('rotation_max_degrees', 5.0) > 0 and random.random() < 0.7:
            landmarks, frames = rotation(
                landmarks, frames,
                max_degrees=self.cfg['rotation_max_degrees'],
            )

        # 6. Frame dropout — p=0.5, simulates occlusion
        if self.cfg.get('frame_dropout', True) and random.random() < 0.5:
            landmarks, frames = frame_dropout(
                landmarks, frames,
                max_dropout=self.cfg.get('frame_dropout_max', 2),
            )

        return landmarks, frames, label
