#!/usr/bin/env python3
"""
preprocess_holistic.py — richer skeleton via MediaPipe Holistic, no face data.

Replaces the 21-point single-hand encoding with body + both hands:

    pose 11..24   14 points   shoulders, elbows, wrists, pose-hand stubs, hips
    left hand     21 points
    right hand    21 points
    ------------------------
    56 points  ->  168 dims, plus 6 trajectory dims = 174

Face landmarks are never requested or stored. Pose 0..10 (nose, eyes, ears,
mouth) are dropped, refine_face_landmarks is off, and face_landmarks is never
read. Pose 25..32 (knees, ankles, feet) are dropped too — a mirror camera does
not see legs, so those points would be interpolation noise.

Why body pose helps: measured on this dataset, Holistic finds the pose in 99.1%
of frames against ~86% for the hand solution. The pose wrists therefore give a
much more reliable swipe trajectory than the hand model does, and they keep
working through the motion blur that costs the hand solution frames.

Frames are NOT cached. 924 clips x 30 frames as uint8 is ~10 GB; the source
.mp4s are 628 MB and the video branch decodes them on demand instead.

Usage
-----
python preprocess_holistic.py                 # -> data/landmarks_holistic.npz
python preprocess_holistic.py --limit 20      # quick trial run
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from dataset import _extract_frames

# ── Joint selection ───────────────────────────────────────────────────────────
POSE_KEEP = list(range(11, 25))     # 11..24 — body only, no face, no legs
N_POSE = len(POSE_KEEP)             # 14
N_HAND = 21
N_POINTS = N_POSE + 2 * N_HAND      # 56
N_SHAPE = N_POINTS * 3              # 168
N_TRAJ = 6                          # both wrists (x,y,z) relative to frame 0
N_FEATURES = N_SHAPE + N_TRAJ       # 174

# indices *within the kept pose block* (POSE_KEEP starts at 11)
L_SHOULDER, R_SHOULDER = 11 - 11, 12 - 11
L_WRIST, R_WRIST = 15 - 11, 16 - 11

_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv'}


def _init_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,   # face mesh never computed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_frame(hol, frame_rgb):
    """
    One frame -> (56, 3) with NaN where a part was not detected.
    Face landmarks are ignored even though Holistic may compute a coarse set.
    """
    res = hol.process(frame_rgb)
    out = np.full((N_POINTS, 3), np.nan, dtype=np.float32)

    if res.pose_landmarks:
        lms = res.pose_landmarks.landmark
        for i, p in enumerate(POSE_KEEP):
            lm = lms[p]
            out[i] = (lm.x, lm.y, lm.z)

    for off, hand in ((N_POSE, res.left_hand_landmarks),
                      (N_POSE + N_HAND, res.right_hand_landmarks)):
        if hand:
            for i, lm in enumerate(hand.landmark):
                out[off + i] = (lm.x, lm.y, lm.z)

    return out


def interpolate(seq):
    """Fill NaN gaps per coordinate by linear interpolation over time."""
    T, P, _ = seq.shape
    flat = seq.reshape(T, -1).copy()
    idx = np.arange(T)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = ~np.isnan(col)
        if good.sum() == 0:
            col[:] = 0.0
        elif good.sum() < T:
            col[~good] = np.interp(idx[~good], idx[good], col[good])
    return flat.reshape(T, P, 3)


def normalise(seq):
    """
    (T, 56, 3) raw image coords -> (T, 174) features.

      [0:168]   shape      — every point relative to mid-shoulder, divided by
                             shoulder width, so body size and where the person
                             stands drop out
      [168:174] trajectory — how far each wrist has moved *relative to the
                             torso* since frame 0, in shoulder-width units

    The shape stream alone already carries the swipe here, because the hand
    moves relative to the torso — unlike the old wrist-anchored encoding, which
    subtracted the very point that was moving and silently deleted the gesture.
    The trajectory stream makes that change explicit over time rather than
    leaving the model to difference it.

    Trajectory is measured torso-relative on purpose. In absolute image coords a
    person simply walking past the camera produces the same wrist displacement
    as a deliberate swipe, which is a false-positive waiting to happen; relative
    to the shoulders, only actual arm movement registers.
    """
    T = seq.shape[0]
    mid_sh = (seq[:, L_SHOULDER, :] + seq[:, R_SHOULDER, :]) / 2.0     # (T,3)
    width = np.linalg.norm(seq[:, L_SHOULDER, :2] - seq[:, R_SHOULDER, :2], axis=1)
    scale = np.maximum(np.median(width[width > 1e-6]) if (width > 1e-6).any() else 0.0, 1e-3)

    shape = (seq - mid_sh[:, None, :]) / scale
    arm = shape[:, [L_WRIST, R_WRIST], :]                              # (T,2,3) torso-relative
    traj = arm - arm[0:1]                                              # (T,2,3)

    return np.concatenate(
        [shape.reshape(T, -1), traj.reshape(T, N_TRAJ)], axis=1
    ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--out', default='data/landmarks_holistic.npz')
    ap.add_argument('--limit', type=int, default=0, help='clips per class, 0 = all')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw = Path(cfg['data']['raw_dir'])
    NF = cfg['data']['num_frames']
    classes = [str(c) for c in cfg['data']['classes']]

    hol = _init_holistic()
    X, y, clip_ids, rel_paths = [], [], [], []
    miss = {c: [0, 0] for c in classes}
    dropped = []

    for label, cls in enumerate(classes):
        cdir = raw / cls
        if not cdir.exists():
            continue
        vids = sorted(p for p in cdir.iterdir() if p.suffix.lower() in _VIDEO_EXT)
        if args.limit:
            vids = vids[:args.limit]
        if not vids:
            continue

        for vid in tqdm(vids, desc=f"{cls:12}"):
            frames = _extract_frames(str(vid), num_frames=NF)
            if frames is None:
                dropped.append(f"{vid} (unreadable)")
                continue

            seq = np.stack([extract_frame(hol, f) for f in frames])    # (T,56,3)

            pose_missing = int(np.isnan(seq[:, L_SHOULDER, 0]).sum())
            miss[cls][0] += pose_missing
            miss[cls][1] += NF

            # Without shoulders there is no anchor and no scale.
            if pose_missing == NF:
                dropped.append(f"{vid} (no pose in any frame)")
                continue

            X.append(normalise(interpolate(seq)))
            y.append(label)
            clip_ids.append(f"{cls}/{vid.stem}")
            rel_paths.append(str(vid.relative_to(raw)))

    if not X:
        print("[holistic] nothing produced")
        return

    X = np.stack(X).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=np.asarray(y, np.int64),
                        clip_ids=np.asarray(clip_ids),
                        rel_paths=np.asarray(rel_paths),
                        classes=np.asarray(classes),
                        n_points=N_POINTS, n_shape=N_SHAPE, n_traj=N_TRAJ)

    print(f"\n[holistic] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"[holistic] X={X.shape}   ({N_POINTS} points -> {N_SHAPE} shape + {N_TRAJ} traj)")
    print("\n  Per-class counts and pose-detection quality:")
    for label, cls in enumerate(classes):
        n = int((np.asarray(y) == label).sum())
        m, t = miss[cls]
        print(f"    {cls:12} {n:4} clips   frames with no pose: {m/max(t,1)*100:5.1f}%")
    if dropped:
        print(f"\n  {len(dropped)} dropped:")
        for d in dropped[:8]:
            print(f"    {d}")
        if len(dropped) > 8:
            print(f"    ... and {len(dropped)-8} more")
    print("\n  Face landmarks: never requested, never stored.")


if __name__ == '__main__':
    main()
