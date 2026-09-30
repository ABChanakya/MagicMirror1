#!/usr/bin/env python3
"""
kimodo_solve_constraint.py -- cluster-side half of the keyframe-constrained
generation pipeline (needs the real `kimodo` package -- torch, skeleton FK --
so this can't run locally or be tested ahead of time; run on the cluster).

Takes a constraint-target npz from build_constraint_targets.py (a template
pose plus target wrist XYZ at a few keyframes) and solves for the shoulder +
elbow local rotations that place the wrist at those targets, via gradient
descent through KiMoDo's OWN forward-kinematics function -- not a
hand-derived analytic formula, so it can't get the model's internal
bone-frame conventions wrong the way a from-scratch IK solver could.
Everything else (root position, every other joint) stays fixed at the
template's own values, so only the arm moves to reach the target; the rest
of the body stays a normal, valid pose.

Position error alone lets the optimizer find any arm configuration that
reaches the target, including ones where the forearm passes through the
torso -- observed in practice, especially for cross-body targets (a left
hand reaching to the body's right side). --reg-weight adds a penalty for
straying from the template's own (already-natural) rotation at each of the
two free joints, so the solver prefers the least-contorted reachable
solution instead of any reachable one. Tune it empirically: too high and it
can't reach the target at all; too low and it stops mattering.

This is a best-effort first pass: some of the kimodo.skeleton API below
(constructor signature, fk()'s exact return values) is inferred from reading
kimodo/constraints.py on GitHub, not verified by actually running it -- that
requires the cluster. It prints diagnostics up front specifically so that if
an assumption is wrong, the error is immediately actionable rather than a
bare traceback.

Observed separately: even with only the active hand's position constrained,
the text prompt ("swipes one hand to the right") still implies a specific
hand to the model, so the OTHER, unconstrained hand sometimes animates on
its own (following the prompt) instead of staying still -- confusing when
the constrained hand is the "wrong" one for the prompt's implied hand
(build_constraint_targets.py's --hand cross-body option). --pin-other-hand
(on by default) fixes this by also naming the inactive hand in the
constraint's joint_names, holding it at the template's own resting
position -- KiMoDo's `end-effector` constraint type takes a LIST of joint
names, not just one, so both hands can be pinned in a single constraint
without solving any extra IK (the inactive arm's rotation was already left
untouched at the template's value; only the joint_names list changes).

Usage
-----
python kimodo_solve_constraint.py --target constraint_target_0.npz --out constraints_0.json
"""

import argparse
import inspect
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, help='output of build_constraint_targets.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--reg-weight', type=float, default=0.02,
                     help='penalty for deviating from the template rotation at the two free '
                          'joints, to discourage contorted/self-intersecting solutions')
    ap.add_argument('--pin-other-hand', dest='pin_other_hand', action='store_true', default=True,
                     help='also constrain the inactive hand to its template resting position, '
                          'so the text prompt cannot pull it into independent motion (default on)')
    ap.add_argument('--no-pin-other-hand', dest='pin_other_hand', action='store_false')
    args = ap.parse_args()

    from kimodo.skeleton import SOMASkeleton77
    from kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle

    print("== diagnostics (delete this block once the API is confirmed working) ==")
    print("SOMASkeleton77 constructor signature:", inspect.signature(SOMASkeleton77.__init__))
    skeleton = SOMASkeleton77()
    print("skeleton attrs (relevant subset):",
          [a for a in dir(skeleton) if a in
           ('nbjoints', 'root_idx', 'bone_index', 'fk', 'device', 'to')])
    print("skeleton.fk signature:", inspect.signature(skeleton.fk))
    print("==")

    d = np.load(args.target, allow_pickle=True)
    label = str(d['label'])
    # target_hand: which arm actually gets constrained -- may differ from the
    # label's natural hand (see build_constraint_targets.py's --hand). Older
    # target files predate this field; fall back to the natural-hand inference.
    if 'target_hand' in d.files:
        target_hand = str(d['target_hand'])
    else:
        target_hand = 'right' if label == 'swipe_right' else 'left'
    local_rot_mats = d['local_rot_mats']        # (T_out, 77, 3, 3), template
    root_positions_full = d['root_positions']   # (T_out, 3), template
    out_frame_idx = d['out_frame_idx']           # (K,)
    target_xyz = d['target_xyz']                 # (K, 3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(skeleton, 'to'):
        skeleton = skeleton.to(device)

    # SOMA-77 indices, verified earlier against sk.bone_order_names (see
    # kimodo_to_landmarks.py's SOMA dict) -- shoulder/elbow are the two joints
    # whose rotation actually moves the wrist's position via FK; the wrist's
    # own local rotation only affects hand orientation, not position, and we
    # don't have reliable real hand-orientation data to target anyway.
    SOMA_SHOULDER = {'right': 40, 'left': 12}
    SOMA_ELBOW = {'right': 41, 'left': 13}
    SOMA_WRIST = {'right': 42, 'left': 14}
    shoulder_idx = SOMA_SHOULDER[target_hand]
    elbow_idx = SOMA_ELBOW[target_hand]
    wrist_idx = SOMA_WRIST[target_hand]
    print(f"label={label}  target_hand={target_hand}  (joints: shoulder={shoulder_idx} elbow={elbow_idx} wrist={wrist_idx})")

    rot_mats = torch.tensor(local_rot_mats[out_frame_idx], device=device, dtype=torch.float32)  # (K,77,3,3)
    root_pos = torch.tensor(root_positions_full[out_frame_idx], device=device, dtype=torch.float32)  # (K,3)
    target = torch.tensor(target_xyz, device=device, dtype=torch.float32)  # (K,3)

    axis_angle = matrix_to_axis_angle(rot_mats)  # (K,77,3)
    fixed = axis_angle.clone()
    natural = axis_angle[:, [shoulder_idx, elbow_idx]].clone().detach()  # template's own rotation -- the "natural" reference
    free = natural.clone().requires_grad_(True)  # (K,2,3)

    opt = torch.optim.Adam([free], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        aa = fixed.clone()
        aa[:, [shoulder_idx, elbow_idx]] = free
        mats = axis_angle_to_matrix(aa)
        fk_out = skeleton.fk(mats, root_pos)
        global_pos = fk_out[1]  # (global_rots, global_positions, ...) per constraints.py's from_dict
        wrist_pos = global_pos[:, wrist_idx]
        err = wrist_pos - target
        pos_loss = (err ** 2).sum(-1).mean()
        reg_loss = ((free - natural) ** 2).sum(-1).mean()
        loss = pos_loss + args.reg_weight * reg_loss
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            max_err = (err ** 2).sum(-1).sqrt().max().item()
            print(f"  iter {it:4d}  pos_loss {pos_loss.item():.6f}  reg_loss {reg_loss.item():.6f}  max positional error {max_err:.4f}")

    with torch.no_grad():
        aa = fixed.clone()
        aa[:, [shoulder_idx, elbow_idx]] = free
        mats = axis_angle_to_matrix(aa)
        fk_out = skeleton.fk(mats, root_pos)
        global_pos = fk_out[1]
        final_err = ((global_pos[:, wrist_idx] - target) ** 2).sum(-1).sqrt()
        print("final per-keyframe position error:", final_err.tolist())
        if final_err.max().item() > 0.05:
            print("WARNING: error still large after optimization -- target may be out of the "
                  "arm's reach, or --iters/--lr needs adjusting. Check before using this constraint.")

    smooth_root_2d = root_pos[:, [0, 2]].detach().cpu().tolist()
    if args.pin_other_hand:
        # both hands, generic end-effector type -- the inactive arm's rotation
        # in `aa` was never touched by the optimizer, so it's still exactly
        # the template's own resting pose; naming it here is what makes
        # KiMoDo actually hold it there instead of letting the text prompt
        # pull it into motion.
        constraint = {
            "type": "end-effector",
            "joint_names": ["LeftHand", "RightHand"],
            "frame_indices": [int(i) for i in out_frame_idx],
            "local_joints_rot": aa.detach().cpu().tolist(),
            "root_positions": root_pos.detach().cpu().tolist(),
            "smooth_root_2d": smooth_root_2d,
        }
    else:
        constraint = {
            "type": "right-hand" if target_hand == "right" else "left-hand",
            "frame_indices": [int(i) for i in out_frame_idx],
            "local_joints_rot": aa.detach().cpu().tolist(),
            "root_positions": root_pos.detach().cpu().tolist(),
            "smooth_root_2d": smooth_root_2d,
        }
    with open(args.out, 'w') as f:
        json.dump([constraint], f)
    print(f"wrote {args.out}  (pin_other_hand={args.pin_other_hand})")


if __name__ == '__main__':
    main()
