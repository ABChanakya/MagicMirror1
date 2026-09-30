#!/usr/bin/env python3
"""
kimodo_export_bvh.py — convert KiMoDo .npz motion(s) to .bvh, using KiMoDo's
own official export function (kimodo.exports.bvh.save_motion_bvh) so the
skeleton hierarchy, rest pose, and bone naming all come from KiMoDo itself
rather than being re-derived by hand. Needs the real `kimodo` package, so
this runs on the cluster (like kimodo_solve_constraint.py).

standard_tpose=True exports with a standard T-pose rest pose rather than the
BONES-SEED dataset's native rest pose -- the right choice for retargeting
onto an external rig (e.g. a Mixamo character in Unity), since T-pose is the
convention most humanoid retargeting tools expect.

fps=30.0 matches our generations: 60 frames at --duration 2.0 = 30fps.

Usage
-----
python kimodo_export_bvh.py --npz kimodo_swipe_right_6.npz --out kimodo_swipe_right_6.bvh
python kimodo_export_bvh.py --glob 'constraint_target_swipe_*_gen.npz' --out-dir bvh_export/
"""

import argparse
import glob
import os

import numpy as np
import torch


def export_one(npz_path, out_path, skeleton, fps):
    d = np.load(npz_path)
    local_rot_mats = torch.tensor(d['local_rot_mats'], dtype=torch.float32)
    root_positions = torch.tensor(d['root_positions'], dtype=torch.float32)
    from kimodo.exports.bvh import save_motion_bvh
    save_motion_bvh(out_path, local_rot_mats, root_positions,
                     skeleton=skeleton, fps=fps, standard_tpose=True)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', help='a single KiMoDo .npz to convert')
    ap.add_argument('--out', help='output .bvh path (for --npz mode)')
    ap.add_argument('--glob', help='glob pattern for batch conversion, e.g. "kimodo_swipe_*.npz"')
    ap.add_argument('--list', help='text file of specific .npz paths to convert, one per line '
                     '(e.g. a pre-filtered manifest of only validated clips)')
    ap.add_argument('--out-dir', help='output directory (for --glob/--list mode)')
    ap.add_argument('--fps', type=float, default=30.0)
    args = ap.parse_args()

    from kimodo.skeleton import SOMASkeleton77
    skeleton = SOMASkeleton77()

    if args.npz:
        export_one(args.npz, args.out, skeleton, args.fps)
    elif args.glob or args.list:
        os.makedirs(args.out_dir, exist_ok=True)
        if args.list:
            with open(args.list) as f:
                paths = [line.strip() for line in f if line.strip()]
        else:
            paths = sorted(glob.glob(args.glob))
        print(f"converting {len(paths)} files")
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(args.out_dir, f"{stem}.bvh")
            try:
                export_one(p, out_path, skeleton, args.fps)
            except Exception as e:
                print(f"FAILED on {p}: {e}")
    else:
        ap.error('need --npz, --glob, or --list')


if __name__ == '__main__':
    main()
