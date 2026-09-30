#!/usr/bin/env python3
"""
kimodo_pure_ik_retarget.py -- retarget a real clip's FULL trajectory (every
frame, not just a few keyframes) onto a KiMoDo template body via IK alone,
with NO diffusion generation involved. Where kimodo_solve_constraint.py
solves a few keyframes and hands them to kimodo_gen as a constraint (letting
the diffusion model improvise everything else, which is where the
self-intersection and dual-hand issues came from), this script solves EVERY
frame directly and writes out a complete, standalone motion -- the real
trajectory, retargeted, nothing improvised. Needs the real `kimodo` package
(skeleton FK), so runs on the cluster.

Output is a plain npz with local_rot_mats/root_positions in exactly the
shape KiMoDo's own generated npz files use, so kimodo_export_bvh.py works
on it unchanged.

Usage
-----
python kimodo_pure_ik_retarget.py --target exp_targets/real472_full.npz --out real472_pure_ik.npz
"""

import argparse
import inspect

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, help='output of build_constraint_targets.py, '
                     'built with --n-keyframes matching (or close to) the real clip length')
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--reg-weight', type=float, default=0.02)
    args = ap.parse_args()

    from kimodo.skeleton import SOMASkeleton77
    from kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle

    print("== diagnostics ==")
    print("SOMASkeleton77 constructor signature:", inspect.signature(SOMASkeleton77.__init__))
    skeleton = SOMASkeleton77()
    print("skeleton.fk signature:", inspect.signature(skeleton.fk))
    print("==")

    d = np.load(args.target, allow_pickle=True)
    label = str(d['label'])
    target_hand = str(d['target_hand']) if 'target_hand' in d.files else \
        ('right' if label == 'swipe_right' else 'left')
    local_rot_mats = d['local_rot_mats']
    root_positions_full = d['root_positions']
    out_frame_idx = d['out_frame_idx']
    target_xyz = d['target_xyz']
    K = len(out_frame_idx)
    print(f"label={label}  target_hand={target_hand}  K={K} frames")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(skeleton, 'to'):
        skeleton = skeleton.to(device)

    SOMA_SHOULDER = {'right': 40, 'left': 12}
    SOMA_ELBOW = {'right': 41, 'left': 13}
    SOMA_WRIST = {'right': 42, 'left': 14}
    shoulder_idx = SOMA_SHOULDER[target_hand]
    elbow_idx = SOMA_ELBOW[target_hand]
    wrist_idx = SOMA_WRIST[target_hand]

    rot_mats = torch.tensor(local_rot_mats[out_frame_idx], device=device, dtype=torch.float32)  # (K,77,3,3)
    root_pos = torch.tensor(root_positions_full[out_frame_idx], device=device, dtype=torch.float32)  # (K,3)
    target = torch.tensor(target_xyz, device=device, dtype=torch.float32)  # (K,3)

    axis_angle = matrix_to_axis_angle(rot_mats)  # (K,77,3)
    fixed = axis_angle.clone()
    natural = axis_angle[:, [shoulder_idx, elbow_idx]].clone().detach()
    free = natural.clone().requires_grad_(True)

    opt = torch.optim.Adam([free], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        aa = fixed.clone()
        aa[:, [shoulder_idx, elbow_idx]] = free
        mats = axis_angle_to_matrix(aa)
        fk_out = skeleton.fk(mats, root_pos)
        global_pos = fk_out[1]
        wrist_pos = global_pos[:, wrist_idx]
        err = wrist_pos - target
        pos_loss = (err ** 2).sum(-1).mean()
        reg_loss = ((free - natural) ** 2).sum(-1).mean()
        loss = pos_loss + args.reg_weight * reg_loss
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            max_err = (err ** 2).sum(-1).sqrt().max().item()
            print(f"  iter {it:4d}  pos_loss {pos_loss.item():.6f}  max positional error {max_err:.4f}")

    with torch.no_grad():
        aa = fixed.clone()
        aa[:, [shoulder_idx, elbow_idx]] = free
        mats = axis_angle_to_matrix(aa)
        fk_out = skeleton.fk(mats, root_pos)
        global_pos = fk_out[1]
        final_err = ((global_pos[:, wrist_idx] - target) ** 2).sum(-1).sqrt()
        print("final per-frame position error: max", final_err.max().item(), "mean", final_err.mean().item())

    np.savez(args.out,
             local_rot_mats=mats.detach().cpu().numpy(),
             root_positions=root_pos.detach().cpu().numpy(),
             posed_joints=global_pos.detach().cpu().numpy())
    print(f"wrote {args.out}  ({K} frames, pure IK, no diffusion)")


if __name__ == '__main__':
    main()
