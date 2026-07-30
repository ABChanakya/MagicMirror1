#!/usr/bin/env python3
"""
test_pipeline.py — regression tests for the landmark encoding pipeline.

Run: python test_pipeline.py

These exist because of a real bug: _normalise_landmarks used to subtract the
wrist from every frame, which set the wrist to (0,0,0) in all 30 frames and
deleted the swipe. swipe_left, swipe_right and swipe_up all encoded to
bit-identical tensors, so the landmark branch carried no direction information
at all and the model had to read direction off raw pixels instead — which is
why it never transferred to new people. test_direction_is_preserved() below
fails loudly if that regresses.
"""

import sys

import numpy as np
import torch

from dataset import _interpolate_missing_landmarks, _normalise_landmarks

PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL}  {name}" + (f"   {detail}" if detail else ""))


# ── Helpers ───────────────────────────────────────────────────────────────────

_HAND = np.random.RandomState(0).rand(21, 3).astype(np.float32) * 0.1


def make_clip(dx, dy, hand_scale=1.0, x0=0.3, y0=0.5, T=30):
    """A hand of fixed shape translating by (dx, dy) over T frames."""
    out = np.zeros((T, 21, 3), np.float32)
    for t in range(T):
        f = t / (T - 1)
        out[t] = _HAND * hand_scale + np.array([x0 + dx * f, y0 + dy * f, 0.0])
    return out.reshape(T, 63)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_output_shape():
    enc = _normalise_landmarks(make_clip(0.4, 0.0))
    check("encoding is (30, 66)", enc.shape == (30, 66), f"got {enc.shape}")


def test_direction_is_preserved():
    """The regression guard. All four directions must encode differently."""
    r = _normalise_landmarks(make_clip(+0.4, 0.0))
    l = _normalise_landmarks(make_clip(-0.4, 0.0))
    u = _normalise_landmarks(make_clip(0.0, -0.4))
    d = _normalise_landmarks(make_clip(0.0, +0.4))

    check("right != left", not np.allclose(r, l, atol=1e-4))
    check("right != up", not np.allclose(r, u, atol=1e-4))
    check("up != down", not np.allclose(u, d, atol=1e-4))

    # Signs must be unambiguous: x for left/right, y for up/down
    check("right has +x trajectory", r[-1, 63] > 0.5, f"{r[-1, 63]:+.3f}")
    check("left  has -x trajectory", l[-1, 63] < -0.5, f"{l[-1, 63]:+.3f}")
    check("up    has -y trajectory", u[-1, 64] < -0.5, f"{u[-1, 64]:+.3f}")
    check("down  has +y trajectory", d[-1, 64] > 0.5, f"{d[-1, 64]:+.3f}")


def test_scale_invariance():
    """A big hand making a proportionally big swipe encodes like a small one."""
    small = _normalise_landmarks(make_clip(0.4, 0.0, hand_scale=1.0))
    big = _normalise_landmarks(make_clip(0.8, 0.0, hand_scale=2.0))
    check("scale-invariant (2x hand + 2x swipe)",
          np.allclose(small[-1, 63], big[-1, 63], atol=1e-3),
          f"{small[-1, 63]:.3f} vs {big[-1, 63]:.3f}")


def test_position_invariance():
    """Where in the frame the gesture happens must not matter."""
    centre = _normalise_landmarks(make_clip(0.4, 0.0, x0=0.3, y0=0.5))
    corner = _normalise_landmarks(make_clip(0.4, 0.0, x0=0.05, y0=0.9))
    check("position-invariant (frame corner vs centre)",
          np.allclose(centre, corner, atol=1e-3))


def test_null_is_distinguishable():
    """A still hand must encode with near-zero trajectory."""
    still = _normalise_landmarks(make_clip(0.0, 0.0))
    swipe = _normalise_landmarks(make_clip(0.4, 0.0))
    check("still hand has ~zero trajectory",
          abs(still[-1, 63]) < 0.05 and abs(still[-1, 64]) < 0.05,
          f"({still[-1, 63]:+.3f}, {still[-1, 64]:+.3f})")
    check("still clearly separable from swipe",
          abs(swipe[-1, 63]) - abs(still[-1, 63]) > 1.0)


def test_flip_augmentation():
    """horizontal_flip must negate trajectory_x and swap LEFT<->RIGHT."""
    from augmentation import horizontal_flip, SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP

    right = torch.from_numpy(_normalise_landmarks(make_clip(0.4, 0.0)))
    dummy = torch.zeros(30, 3, 4, 4)

    flipped, _, new_label = horizontal_flip(right, dummy, SWIPE_RIGHT)
    check("flip relabels RIGHT -> LEFT", new_label == SWIPE_LEFT)
    check("flip negates trajectory x",
          np.isclose(flipped[-1, 63].item(), -right[-1, 63].item(), atol=1e-5),
          f"{right[-1, 63]:+.3f} -> {flipped[-1, 63]:+.3f}")
    check("flip leaves UP label alone",
          horizontal_flip(right, dummy, SWIPE_UP)[2] == SWIPE_UP)

    # A flipped right-swipe must match a genuine left-swipe in TRAJECTORY.
    # The shape stream is deliberately NOT equal: mirroring a right hand yields
    # a left hand, so joint x-coords invert. That is physically correct — it is
    # what makes flip a useful augmentation for the opposite hand.
    left = _normalise_landmarks(make_clip(-0.4, 0.0))
    check("flipped right matches real left trajectory",
          np.allclose(flipped.numpy()[:, 63:66], left[:, 63:66], atol=1e-4))
    check("flipped right mirrors the hand shape (not identical to real left)",
          not np.allclose(flipped.numpy()[:, :63], left[:, :63], atol=1e-4)
          and np.allclose(flipped.numpy()[:, 0:63:3], -left[:, 0:63:3], atol=1e-4))


def test_rotation_keeps_dims():
    from augmentation import rotation
    lm = torch.from_numpy(_normalise_landmarks(make_clip(0.4, 0.0)))
    out, _ = rotation(lm, torch.zeros(30, 3, 4, 4), max_degrees=3.0)
    check("rotation preserves (30, 66)", tuple(out.shape) == (30, 66), f"{tuple(out.shape)}")
    check("rotation keeps trajectory sign", out[-1, 63].item() > 0)


def test_train_inference_encoding_identical():
    """
    The important one: inference must encode a window exactly as training does.
    A duplicated normaliser in inference.py previously drifted from the training
    version, which is undetectable at runtime but destroys accuracy.
    """
    try:
        from inference import _normalise_landmark_window
    except Exception as e:                                   # pragma: no cover
        check("inference encoder importable", False, str(e)[:60])
        return

    raw = make_clip(0.4, 0.0)
    via_inference = _normalise_landmark_window(raw)
    via_training = _normalise_landmarks(_interpolate_missing_landmarks(raw.copy()))

    check("inference encoding == training encoding",
          via_inference.shape == via_training.shape
          and np.allclose(via_inference, via_training, atol=1e-6))

    # Undetected frames arrive as all-zero rows; they must be interpolated,
    # not fed to the normaliser as real coordinates.
    gappy = raw.copy()
    gappy[5] = 0.0
    gappy[17] = 0.0
    out = _normalise_landmark_window(gappy)
    check("handles dropped detections without NaN",
          out.shape == (30, 66) and np.isfinite(out).all())

    allzero = _normalise_landmark_window(np.zeros((30, 63), np.float32))
    check("all-zero window returns finite zeros",
          allzero.shape == (30, 66) and np.isfinite(allzero).all())


def test_model_accepts_66():
    from models.fusion_head import LandmarkOnlyModel
    m = LandmarkOnlyModel(landmark_d_model=128, landmark_nhead=4,
                          landmark_num_layers=3, landmark_dim_feedforward=512,
                          num_classes=5, input_dim=66)
    out = m(torch.randn(2, 30, 66))
    check("LandmarkOnlyModel forward -> (2, 5)", tuple(out.shape) == (2, 5), f"{tuple(out.shape)}")


def test_checkpoint_cross_compatible():
    """
    A landmark-only checkpoint must load into the full fusion model (and hence
    into inference.py's non-landmark path) via strict=False.
    """
    from models.fusion_head import LandmarkOnlyModel

    light = LandmarkOnlyModel(landmark_d_model=256, landmark_nhead=8,
                              landmark_num_layers=6, landmark_dim_feedforward=1024,
                              num_classes=5, input_dim=66)
    sd = light.state_dict()
    prefixes = {k.split('.')[0] for k in sd}
    check("state-dict keys match fusion submodule names",
          prefixes == {'landmark_encoder', 'lm_only_head'}, str(sorted(prefixes)))


def main():
    print(__doc__.strip().split("\n")[0])
    print()
    for fn in [
        test_output_shape,
        test_direction_is_preserved,
        test_scale_invariance,
        test_position_invariance,
        test_null_is_distinguishable,
        test_flip_augmentation,
        test_rotation_keeps_dims,
        test_train_inference_encoding_identical,
        test_model_accepts_66,
        test_checkpoint_cross_compatible,
    ]:
        print(f"{fn.__name__}:")
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} raised", False, f"{type(e).__name__}: {e}")
        print()

    n_pass, n_tot = sum(_results), len(_results)
    print(f"{'='*56}\n  {n_pass}/{n_tot} checks passed\n{'='*56}")
    return 0 if n_pass == n_tot else 1


if __name__ == '__main__':
    sys.exit(main())
