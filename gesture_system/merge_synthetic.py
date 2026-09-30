#!/usr/bin/env python3
"""
merge_synthetic.py — validate KiMoDo .npz motions and merge the ones that
pass direction_ok() into a copy of the real landmarks dataset, resampled to
the same fixed-length uniform sampling dataset.py's _extract_frames() uses
for real clips (np.linspace over the sequence), so train_twostream.py can
consume the result with zero changes.

Synthetic clip_ids are deliberately built without a YYYYMMDD-HHMMSS
timestamp, so train_twostream.py's split_by_session() always reads them as
session 'unknown' -> they can only land in train/val, never in the held-out
test session. The test set stays 100% real data.

Usage
-----
python merge_synthetic.py --synth-dir . --out data/landmarks_holistic_plus_synth.npz
"""

import argparse
import glob
import os

import numpy as np

from kimodo_to_landmarks import soma_to_our56, check_trajectory, direction_ok, _import_normalise


def resample(posed, n):
    idx = np.linspace(0, posed.shape[0] - 1, n).astype(int)
    return posed[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth-dir', required=True, help='dir with kimodo_{swipe_left,swipe_right,null}_*.npz')
    ap.add_argument('--base', default='data/landmarks_holistic.npz')
    ap.add_argument('--out', default='data/landmarks_holistic_plus_synth.npz')
    ap.add_argument('--n-frames', type=int, default=30)
    ap.add_argument('--max-per-class', type=int, default=0,
                     help='cap kept synthetic clips per class (0 = no cap), '
                          'random subsample with a fixed seed for a dosage sweep')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    normalise = _import_normalise()

    base = np.load(args.base, allow_pickle=True)
    classes = list(base['classes'])
    X_parts = [base['X']]
    y_parts = [base['y']]
    id_parts = [base['clip_ids']]
    rel_parts = [base['rel_paths']]

    def add(label, kept_X, kept_ids):
        label_idx = classes.index(label)
        if args.max_per_class and len(kept_X) > args.max_per_class:
            keep_idx = rng.choice(len(kept_X), args.max_per_class, replace=False)
            kept_X = [kept_X[i] for i in keep_idx]
            kept_ids = [kept_ids[i] for i in keep_idx]
        if kept_X:
            X_parts.append(np.stack(kept_X).astype(np.float32))
            y_parts.append(np.full(len(kept_X), label_idx, dtype=np.int64))
            id_parts.append(np.asarray(kept_ids))
            rel_parts.append(np.asarray([''] * len(kept_X)))
        return len(kept_X)

    n_kept, n_rejected = 0, 0

    # plain text-prompted clips: label is guessed from generated motion, so
    # direction_ok()'s sign check matters here -- it's the only signal we have.
    for label in ('swipe_left', 'swipe_right', 'null'):
        paths = sorted(glob.glob(os.path.join(args.synth_dir, f'kimodo_{label}_*.npz')))
        kept_X, kept_ids = [], []
        for p in paths:
            d = np.load(p)
            posed = d['posed_joints']  # (T, 77, 3)
            traj = check_trajectory(normalise(soma_to_our56(posed)))
            if not direction_ok(label, traj):
                n_rejected += 1
                continue
            feats = normalise(soma_to_our56(resample(posed, args.n_frames)))  # (n_frames, 174)
            kept_X.append(feats)
            stem = os.path.splitext(os.path.basename(p))[0]
            kept_ids.append(f"{label}/{stem}_synth")
        n_kept += add(label, kept_X, kept_ids)

    # keyframe-constrained clips: quality-only checking (constrained_quality_ok,
    # trusting the real source clip's label) seemed right in isolation but broke
    # train_twostream.py's flip augmentation (label==0<->1, X mirrored), which
    # requires a consistent sign convention across all swipe_left/right data --
    # letting sign-inconsistent clips through corrupted swipe_right via flipping
    # (measured: 66.9% -> 53.2% balanced accuracy). Use direction_ok() here too,
    # same as plain-prompt clips, despite having a known label.
    for label in ('swipe_left', 'swipe_right'):
        paths = sorted(glob.glob(os.path.join(args.synth_dir, f'constraint_target_{label}_*_gen.npz')))
        kept_X, kept_ids = [], []
        for p in paths:
            d = np.load(p)
            posed = d['posed_joints']
            traj = check_trajectory(normalise(soma_to_our56(posed)))
            if not direction_ok(label, traj):
                n_rejected += 1
                continue
            feats = normalise(soma_to_our56(resample(posed, args.n_frames)))
            kept_X.append(feats)
            stem = os.path.splitext(os.path.basename(p))[0]
            kept_ids.append(f"{label}/{stem}_synth")
        n_kept += add(label, kept_X, kept_ids)

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    clip_ids = np.concatenate(id_parts)
    rel_paths = np.concatenate(rel_parts)

    out = args.out
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez_compressed(out, X=X, y=y, clip_ids=clip_ids, rel_paths=rel_paths,
                         classes=base['classes'], n_points=base['n_points'],
                         n_shape=base['n_shape'], n_traj=base['n_traj'])
    print(f"kept {n_kept} synthetic clips, rejected {n_rejected}")
    print(f"wrote {out}: X={X.shape} (base was {base['X'].shape})")


if __name__ == '__main__':
    main()
