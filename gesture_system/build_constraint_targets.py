#!/usr/bin/env python3
"""
build_constraint_targets.py — take one real clip's wrist trajectory (already
normalised: mid-shoulder-centered, shoulder-width-scaled) and rescale it onto
a template KiMoDo body, producing target wrist XYZ positions at a few
keyframes in KiMoDo's own coordinate space. This is the local (numpy-only)
half of the keyframe-constrained generation pipeline -- the cluster-side half
(kimodo_solve_constraint.py) then solves for the arm's joint rotations that
reproduce these target positions via forward kinematics, using KiMoDo's own
skeleton code so we never have to reverse-engineer its bone-frame conventions.

Coordinate handling
--------------------
- Y: real/MediaPipe data is Y-down (standard image coordinates); KiMoDo is
  Y-up. This is empirically confirmed (hips read as positive-Y relative to
  shoulder in real normalised data; KiMoDo's raw shoulder Y > raw hip Y) --
  see the fix in soma_to_our56(). Flipped here to match.
- X: no physical ground-truth check exists (left/right is arbitrary), unlike
  Y (gravity gives an unambiguous reference). --flip-x is a toggle, off by
  default; calibrate empirically per generate_and_check.py's output before
  trusting it, don't assume.
- Z (depth): left unflipped; no evidence either way, not changing without a
  reason to.

Usage
-----
python build_constraint_targets.py --real-clip-idx 5 --template kimodo_null_1.npz \
    --label swipe_right --out constraint_target_0.npz
"""

import argparse

import numpy as np

from kimodo_to_landmarks import SOMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/landmarks_holistic.npz')
    ap.add_argument('--real-clip-idx', type=int, required=True,
                     help='row index into data/landmarks_holistic.npz to use as the motion source')
    ap.add_argument('--template', required=True, help='an existing KiMoDo .npz to use as the body template')
    ap.add_argument('--label', required=True, choices=['swipe_left', 'swipe_right'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--flip-x', action='store_true', help='flip X sign (untested default -- calibrate empirically)')
    ap.add_argument('--n-keyframes', type=int, default=3)
    ap.add_argument('--hand', choices=['auto', 'left', 'right'], default='auto',
                     help="which KiMoDo arm gets the constraint. 'auto' (default) uses the "
                          "label-matching hand (right for swipe_right, left for swipe_left) -- "
                          "same as the real source clip's own hand. 'left'/'right' drives the "
                          "other arm through the SAME screen-direction motion (not mirrored), so "
                          "e.g. a left-hand swipe-right crosses the body -- the source trajectory "
                          "shape/direction is unchanged either way, only which arm plays it.")
    args = ap.parse_args()

    real = np.load(args.data, allow_pickle=True)
    x = real['X'][args.real_clip_idx]  # (30, 174)
    clip_id = str(real['clip_ids'][args.real_clip_idx])
    label_idx = list(real['classes']).index(args.label)
    assert real['y'][args.real_clip_idx] == label_idx, \
        f"clip {clip_id} is not labelled {args.label}"
    print(f"using real clip: {clip_id}")

    shape = x[:, :168].reshape(30, 56, 3)
    wrist_block = 5 if args.label == 'swipe_right' else 4  # r_wrist / l_wrist, our56 layout
    wrist_offset = shape[:, wrist_block, :].copy()  # (30, 3), already mid-shoulder-relative, shoulder-width-scaled

    wrist_offset[:, 1] *= -1  # Y: real is Y-down, KiMoDo is Y-up -- confirmed, always flip
    if args.flip_x:
        wrist_offset[:, 0] *= -1
    # Z (depth) from MediaPipe is much noisier than X/Y (inferred, not directly
    # observed) -- frame-to-frame jumps up to 1.4+ units seen on real clips, vs
    # <0.1 for X/Y on the same clips. check_trajectory() never uses Z either.
    # Don't import it into a hard constraint; template supplies Z below instead.
    wrist_offset[:, 2] = np.nan  # marked, overwritten with template Z after rescale

    tmpl = np.load(args.template)
    posed = tmpl['posed_joints']  # (T_out, 77, 3)
    local_rot_mats = tmpl['local_rot_mats']  # (T_out, 77, 3, 3)
    root_positions = tmpl['root_positions']  # (T_out, 3)
    global_root_heading = tmpl['global_root_heading']  # (T_out, 2)
    T_out = posed.shape[0]

    real_frame_idx = np.linspace(0, 29, args.n_keyframes).round().astype(int)
    out_frame_idx = np.linspace(0, T_out - 1, args.n_keyframes).round().astype(int)

    l_sh = posed[:, SOMA['l_shoulder']]
    r_sh = posed[:, SOMA['r_shoulder']]
    mid_shoulder = (l_sh + r_sh) / 2.0  # (T_out, 3)
    shoulder_width = np.linalg.norm(l_sh - r_sh, axis=1)  # (T_out,)

    natural_hand = 'right' if args.label == 'swipe_right' else 'left'
    target_hand = natural_hand if args.hand == 'auto' else args.hand
    wrist_joint = SOMA['r_wrist'] if target_hand == 'right' else SOMA['l_wrist']
    template_wrist_z = posed[:, wrist_joint, 2]  # (T_out,) -- template supplies depth, not real data

    target_xyz = np.zeros((args.n_keyframes, 3), dtype=np.float32)
    for k, (ri, oi) in enumerate(zip(real_frame_idx, out_frame_idx)):
        target_xyz[k, :2] = (mid_shoulder[oi] + wrist_offset[ri] * shoulder_width[oi])[:2]
        target_xyz[k, 2] = template_wrist_z[oi]

    np.savez(args.out,
             local_rot_mats=local_rot_mats, root_positions=root_positions,
             global_root_heading=global_root_heading, out_frame_idx=out_frame_idx,
             target_xyz=target_xyz, label=args.label, source_clip_id=clip_id,
             target_hand=target_hand,
             joint_name=('RightHand' if target_hand == 'right' else 'LeftHand'))
    print(f"target wrist positions ({args.label}, {target_hand} hand"
          f"{' -- crosses body' if target_hand != natural_hand else ''}):")
    for oi, xyz in zip(out_frame_idx, target_xyz):
        print(f"  frame {oi}: {xyz}")
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
