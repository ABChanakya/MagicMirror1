#!/usr/bin/env python3
"""
validate_generated.py — gate for synthetic clips before they enter training.

A generative model produces plausible video, not necessarily the motion you
asked for. A clip generated as `swipe_left` may contain a rightward move, a
twitch, or no arm motion at all — and nothing about the file says so. Training
on that is mislabelled data, which is the single most expensive class of bug in
this project's history: an encoding that silently deleted swipe direction cost
weeks of meaningless 100% accuracy.

So every generated clip is checked against the label it claims:

  1. pose detected in enough frames                 (Holistic, not the hand model)
  2. the wrist actually moves                       (rejects static / twitchy output)
  3. the movement runs in the labelled direction    (rejects inverted or wrong-axis)
  4. the motion is not ambiguous between axes       (rejects diagonal mush)

Clips failing any check are reported and, with --move-rejects, quarantined
rather than deleted, so failures can be inspected — a high reject rate is
information about the generator, not just noise to discard.

Usage
-----
python validate_generated.py --dir generated/swipe_left --label swipe_left
python validate_generated.py --dir generated --by-subdir        # label per folder
python validate_generated.py --dir generated --by-subdir --move-rejects
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml

from dataset import _extract_frames
from preprocess_holistic import (L_SHOULDER, L_WRIST, N_SHAPE, R_SHOULDER,
                                 R_WRIST, _init_holistic, extract_frame,
                                 interpolate, normalise)

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif'}

# Thresholds in shoulder-width units, matched to what real clips produce.
MIN_POSE_FRAC = 0.60     # pose must be found in at least this fraction of frames
MIN_TRAVEL = 0.45        # a real swipe moves the wrist at least this far
MIN_AXIS_RATIO = 1.6     # dominant axis must beat the other by this much


def check_clip(path, label, classes, num_frames, hol):
    frames = _extract_frames(str(path), num_frames=num_frames)
    if frames is None:
        return False, 'unreadable', {}

    seq = np.stack([extract_frame(hol, f) for f in frames])
    pose_ok = ~np.isnan(seq[:, L_SHOULDER, 0])
    frac = float(pose_ok.mean())
    if frac < MIN_POSE_FRAC:
        return False, f'pose in only {frac*100:.0f}% of frames', {'pose_frac': frac}

    enc = normalise(interpolate(seq))                 # (T, 174)
    # trajectory block: [L_x,L_y,L_z, R_x,R_y,R_z] relative to frame 0, torso-relative
    traj = enc[:, N_SHAPE:]
    lx, ly = traj[:, 0], traj[:, 1]
    rx, ry = traj[:, 3], traj[:, 4]

    # use whichever arm moved more — the generator may animate either
    travel_l = float(np.hypot(lx[-1], ly[-1]))
    travel_r = float(np.hypot(rx[-1], ry[-1]))
    if travel_r >= travel_l:
        dx, dy, travel, arm = float(rx[-1]), float(ry[-1]), travel_r, 'right'
    else:
        dx, dy, travel, arm = float(lx[-1]), float(ly[-1]), travel_l, 'left'

    info = {'pose_frac': frac, 'dx': dx, 'dy': dy, 'travel': travel, 'arm': arm}

    if label == 'null':
        # null must NOT contain a clear swipe
        if travel >= MIN_TRAVEL and abs(dx) > abs(dy) * MIN_AXIS_RATIO:
            return False, f'looks like a swipe (dx={dx:+.2f})', info
        return True, 'ok', info

    if travel < MIN_TRAVEL:
        return False, f'barely moves (travel={travel:.2f})', info

    if abs(dx) < abs(dy) * MIN_AXIS_RATIO and abs(dy) < abs(dx) * MIN_AXIS_RATIO:
        return False, f'ambiguous direction (dx={dx:+.2f}, dy={dy:+.2f})', info

    if label == 'swipe_left':
        if not (dx < 0 and abs(dx) > abs(dy)):
            return False, f'not leftward (dx={dx:+.2f}, dy={dy:+.2f})', info
    elif label == 'swipe_right':
        if not (dx > 0 and abs(dx) > abs(dy)):
            return False, f'not rightward (dx={dx:+.2f}, dy={dy:+.2f})', info
    else:
        return False, f'unhandled label {label}', info

    return True, 'ok', info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--label', help='label for every clip in --dir')
    ap.add_argument('--by-subdir', action='store_true',
                    help='take the label from each immediate subdirectory name')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--move-rejects', action='store_true',
                    help='move failures to <dir>/_rejected/ instead of leaving them')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    classes = [str(c) for c in cfg['data']['classes']]
    NF = cfg['data']['num_frames']

    root = Path(args.dir)
    jobs = []
    if args.by_subdir:
        for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('_')):
            if sub.name not in classes:
                print(f"  skipping {sub.name}/ — not a configured class")
                continue
            vids = sorted(p for p in sub.iterdir() if p.suffix.lower() in VIDEO_EXT)
            jobs += [(v, sub.name) for v in (vids[:args.limit] if args.limit else vids)]
    else:
        if not args.label:
            ap.error('--label is required unless --by-subdir is given')
        vids = sorted(p for p in root.rglob('*') if p.suffix.lower() in VIDEO_EXT)
        jobs = [(v, args.label) for v in (vids[:args.limit] if args.limit else vids)]

    if not jobs:
        print(f"no video files under {root}")
        return

    hol = _init_holistic()
    print(f"Validating {len(jobs)} generated clips\n")
    print(f"  thresholds: pose>={MIN_POSE_FRAC*100:.0f}% of frames, "
          f"travel>={MIN_TRAVEL}, axis ratio>={MIN_AXIS_RATIO}\n")

    passed, failed = [], []
    for path, label in jobs:
        ok, why, info = check_clip(path, label, classes, NF, hol)
        (passed if ok else failed).append((path, label, why, info))
        if not ok:
            print(f"  REJECT  {label:12} {path.name[:44]:46} {why}")

    print(f"\n  passed {len(passed)}/{len(jobs)}  ({len(passed)/len(jobs)*100:.0f}%)")

    if failed:
        from collections import Counter
        reasons = Counter(w.split('(')[0].strip() for _, _, w, _ in failed)
        print("\n  rejection reasons:")
        for r, n in reasons.most_common():
            print(f"    {n:4}  {r}")

    if passed:
        tr = [i['travel'] for _, _, _, i in passed]
        pf = [i['pose_frac'] for _, _, _, i in passed]
        print(f"\n  accepted clips: travel {np.mean(tr):.2f}+-{np.std(tr):.2f}, "
              f"pose detected {np.mean(pf)*100:.0f}% of frames")

    if args.move_rejects and failed:
        qdir = root / '_rejected'
        qdir.mkdir(exist_ok=True)
        for path, label, why, _ in failed:
            dest = qdir / label
            dest.mkdir(exist_ok=True)
            shutil.move(str(path), str(dest / path.name))
        print(f"\n  moved {len(failed)} rejects to {qdir}/ (not deleted)")

    rate = len(passed) / len(jobs)
    print()
    if rate < 0.5:
        print("  A pass rate this low means the generator is mostly not producing the")
        print("  motion being asked for. Fix the conditioning before generating in bulk —")
        print("  pose-guided generation keeps the label correct by construction.")
    elif rate < 0.85:
        print("  Usable, but inspect _rejected/ to see what the generator gets wrong.")
    else:
        print("  Generator is following the conditioning well.")


if __name__ == '__main__':
    main()
