#!/usr/bin/env python3
"""
kimodo_to_landmarks.py — KiMoDo 3D motion -> the project's 174-dim landmark
features, no rendering, no video model, no ControlNet.

Why this works without a virtual camera: preprocess_holistic.py's normalise()
centers every point on mid-shoulder and scales by shoulder width, which is
already camera-agnostic — it doesn't care whether the input was MediaPipe's
2D-image-plane-plus-rough-depth estimate or KiMoDo's true 3D world-space
skeleton, both reduce to "relative to torso, scaled by shoulder width." So
this script's only real job is mapping KiMoDo's 77-joint SOMA skeleton onto
the same 56-point layout (14 body + 2x21 hand) preprocess_holistic.py uses,
then calling its normalise() directly — not reimplementing it, so this never
drifts from what the real training data actually looks like.

Joint correspondence (SOMA bone-name convention: a joint is named for the
bone that STARTS there, e.g. "LeftForeArm" is the elbow, where the forearm
bone begins) — verified against sk.bone_order_names on the cluster:
  l_shoulder=LeftArm(12)   r_shoulder=RightArm(40)
  l_elbow=LeftForeArm(13)  r_elbow=RightForeArm(41)
  l_wrist=LeftHand(14)     r_wrist=RightHand(42)
  l_hip=LeftLeg(67)        r_hip=RightLeg(72)   (BVH: "LeftLeg" bone begins
                                                  at the hip joint)
The 6 pose "hand stub" points (rough finger position from the body model,
distinct from the detailed hand block) don't have a direct KiMoDo
equivalent; approximated with each finger's first knuckle.

Hand mapping: MediaPipe's 21 points/hand are wrist + 4 per finger (thumb,
index, middle, ring, pinky), uniformly. SOMA has 4 thumb joints (maps
directly) but 5 per other finger (Finger1-4 + FingerEnd) — Finger4 is
dropped, keeping Finger1/2/3/End, since finger sub-joint precision doesn't
matter much for this task (train_landmark.py / preprocess_holistic.py's own
docs: the wrist trajectory carries the swipe, not finger detail).

Validate BEFORE merging into training data — see --report below. A wave
that swings out and returns to its start (KiMoDo's default failure mode
with a plain text prompt, confirmed earlier: net displacement close to
zero) should not pass, the same way a bad Wan-Animate clip shouldn't.

Usage
-----
python kimodo_to_landmarks.py --npz kimodo_swipe_left_3431.npz --label swipe_left --report
python kimodo_to_landmarks.py --npz kimodo_swipe_left_3431.npz --label swipe_left --out synth_swipe_left.npz
"""

import argparse

import numpy as np

# SOMA-77 joint indices (0-based), from sk.bone_order_names on the cluster.
SOMA = {
    'l_shoulder': 12, 'r_shoulder': 40, 'l_elbow': 13, 'r_elbow': 41,
    'l_wrist': 14, 'r_wrist': 42, 'l_hip': 67, 'r_hip': 72,
}
# finger -> (joint1, joint2, joint3, joint_end) SOMA indices, dropping Finger4
_L_THUMB = (15, 16, 17, 18)
_L_INDEX = (19, 20, 21, 23)
_L_MIDDLE = (24, 25, 26, 28)
_L_RING = (29, 30, 31, 33)
_L_PINKY = (34, 35, 36, 38)
_R_THUMB = (43, 44, 45, 46)
_R_INDEX = (47, 48, 49, 51)
_R_MIDDLE = (52, 53, 54, 56)
_R_RING = (57, 58, 59, 61)
_R_PINKY = (62, 63, 64, 66)


def _hand21(base_wrist, thumb, index, middle, ring, pinky, joints):
    """21 SOMA joint indices in MediaPipe hand order: wrist, then 4 per
    finger (thumb, index, middle, ring, pinky)."""
    return [base_wrist, *thumb, *index, *middle, *ring, *pinky]


def soma_to_our56(posed_joints):
    """
    posed_joints: (T, 77, 3) from a KiMoDo .npz's posed_joints array.
    Returns (T, 56, 3) in preprocess_holistic.py's POSE_KEEP + 2x21-hand
    layout, ready for its normalise().
    """
    T = posed_joints.shape[0]
    out = np.full((T, 56, 3), np.nan, dtype=np.float32)

    # body block (14 points): shoulders, elbows, wrists, hand-stubs, hips —
    # matching POSE_KEEP = [11..24] order exactly.
    out[:, 0] = posed_joints[:, SOMA['l_shoulder']]
    out[:, 1] = posed_joints[:, SOMA['r_shoulder']]
    out[:, 2] = posed_joints[:, SOMA['l_elbow']]
    out[:, 3] = posed_joints[:, SOMA['r_elbow']]
    out[:, 4] = posed_joints[:, SOMA['l_wrist']]
    out[:, 5] = posed_joints[:, SOMA['r_wrist']]
    out[:, 6] = posed_joints[:, _L_PINKY[0]]     # l_pinky stub
    out[:, 7] = posed_joints[:, _R_PINKY[0]]     # r_pinky stub
    out[:, 8] = posed_joints[:, _L_INDEX[0]]     # l_index stub
    out[:, 9] = posed_joints[:, _R_INDEX[0]]     # r_index stub
    out[:, 10] = posed_joints[:, _L_THUMB[0]]    # l_thumb stub
    out[:, 11] = posed_joints[:, _R_THUMB[0]]    # r_thumb stub
    out[:, 12] = posed_joints[:, SOMA['l_hip']]
    out[:, 13] = posed_joints[:, SOMA['r_hip']]

    N_POSE = 14
    l_idx = _hand21(SOMA['l_wrist'], _L_THUMB, _L_INDEX, _L_MIDDLE, _L_RING, _L_PINKY, posed_joints)
    r_idx = _hand21(SOMA['r_wrist'], _R_THUMB, _R_INDEX, _R_MIDDLE, _R_RING, _R_PINKY, posed_joints)
    for i, j in enumerate(l_idx):
        out[:, N_POSE + i] = posed_joints[:, j]
    for i, j in enumerate(r_idx):
        out[:, N_POSE + 21 + i] = posed_joints[:, j]

    # KiMoDo is Y-up (raw shoulder Y > raw hip Y); MediaPipe/preprocess_holistic.py
    # is Y-down, standard image coordinates (real data: hip Y ~+1.9 relative to
    # mid-shoulder after normalise(), confirmed empirically). Left unflipped, every
    # synthetic clip's full-body shape features are vertically mirrored relative to
    # real clips -- invisible to the trajectory validator (which only reads |dy| and
    # the sign of dx, never body-shape geometry) but a real, systematic mismatch fed
    # straight into training. Flip here so synthetic and real share one convention.
    out[:, :, 1] *= -1

    return out


def check_trajectory(features_174, min_travel=0.45, min_axis_ratio=1.6):
    """
    Same shape of check as validate_generated.py, but read straight off the
    already-normalised torso-relative trajectory — no re-detection, no
    video, so no uncertainty from that step. traj = features[:, 168:174]
    reshaped (T,2,3) for (l_wrist, r_wrist); direction/magnitude computed
    from whichever wrist moves more.
    """
    T = features_174.shape[0]
    traj = features_174[:, 168:174].reshape(T, 2, 3)
    results = {}
    for name, wrist in (('left', traj[:, 0]), ('right', traj[:, 1])):
        dx, dy = wrist[-1, 0] - wrist[0, 0], wrist[-1, 1] - wrist[0, 1]
        travel = float(np.linalg.norm(wrist, axis=1).max())
        axis_ratio = abs(dx) / max(abs(dy), 1e-6)
        results[name] = {'dx': float(dx), 'dy': float(dy), 'travel': travel,
                         'axis_ratio': axis_ratio,
                         'passes': travel >= min_travel and axis_ratio >= min_axis_ratio}
    return results


def direction_ok(label, traj, flip=False):
    """
    Confirmed against job 3477's 20-clip batch (2026-09-21, both axis
    conventions tallied via --batch-dir): KiMoDo's +X does not read as
    screen-right the way a naive first guess assumed. Confirmed convention
    is swipe_left -> dx>0, swipe_right -> dx<0 (15% -> 65% pass rate on the
    same 20 clips, no regeneration). flip=True reverts to the original,
    now-known-wrong assumption; kept only for reference.
    """
    if label == 'null':
        return not any(r['passes'] for r in traj.values())
    want_positive = (label == 'swipe_left') != flip
    return any(r['passes'] and ((r['dx'] > 0) == want_positive) for r in traj.values())


def constrained_quality_ok(label, traj):
    """
    Quality-only check (travel + axis_ratio, no direction-sign requirement)
    for keyframe-constrained generations, where the label is already known
    from the real source clip -- not being guessed the way direction_ok()
    guesses a likely label for unlabeled plain-text-prompt generation.

    Needed because direction_ok()'s sign convention was calibrated purely
    against synthetic KiMoDo text-prompt batches and never checked against
    real data directly. When it finally was (2026-09-22), most real
    swipe_left source clips read as the "wrong" sign under that convention
    even though they're correctly human-labeled -- real swipe_left
    execution is just less sign-consistent than swipe_right in this
    dataset. Re-applying that sign check to constrained clips rejected
    genuinely good motion (3/8 pass) that was actually fine on quality
    alone (6/9 pass, same clips).
    """
    hand = 'left' if label == 'swipe_left' else 'right'
    return traj[hand]['passes']


def _import_normalise():
    """
    preprocess_holistic.normalise() is pure numpy, but the module's top-level
    `from dataset import _extract_frames` drags in mediapipe/tensorflow —
    broken locally under numpy2 (works fine on the cluster venv). Stub
    `dataset` out first since nothing here calls _extract_frames anyway.
    """
    import sys
    import types
    if 'dataset' not in sys.modules:
        stub = types.ModuleType('dataset')
        stub._extract_frames = None
        sys.modules['dataset'] = stub
    sys.path.insert(0, '.')
    from preprocess_holistic import normalise
    return normalise


def load_traj(npz_path):
    normalise = _import_normalise()
    d = np.load(npz_path)
    posed = d['posed_joints']  # (T, 77, 3)
    ours56 = soma_to_our56(posed)
    features = normalise(ours56)  # (T, 174)
    return features, check_trajectory(features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', help='a KiMoDo-generated .npz')
    ap.add_argument('--label', choices=['swipe_left', 'swipe_right', 'null'])
    ap.add_argument('--out', help='save (1,174) feature array here if it passes')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--batch-dir', help='rescan {prefix}_swipe_{left,right}_*.npz '
                     'in this dir and tally pass rate under both axis '
                     'conventions, no regeneration needed')
    ap.add_argument('--prefix', default='kimodo', help='filename prefix for '
                     '--batch-dir, e.g. "kimodo_test2" for a prompt-variant batch')
    args = ap.parse_args()

    if args.batch_dir:
        import glob
        import os
        dir_labels = ('swipe_left', 'swipe_right')
        counts = {False: {l: [0, 0] for l in dir_labels},
                  True: {l: [0, 0] for l in dir_labels}}
        null_counts = [0, 0]
        for label in dir_labels + ('null',):
            paths = sorted(glob.glob(os.path.join(args.batch_dir, f'{args.prefix}_{label}_*.npz')))
            for p in paths:
                _, traj = load_traj(p)
                if label == 'null':
                    null_counts[1] += 1
                    null_counts[0] += int(direction_ok('null', traj))
                else:
                    for flip in (False, True):
                        ok = direction_ok(label, traj, flip=flip)
                        counts[flip][label][1] += 1
                        counts[flip][label][0] += int(ok)
                if args.report:
                    tags = ' '.join(f"{h}:dx={r['dx']:+.2f}/ar={r['axis_ratio']:.1f}"
                                     f"{'*' if r['passes'] else ''}" for h, r in traj.items())
                    print(f"  {os.path.basename(p):28} {tags}")
        for flip in (False, True):
            print(f"-- {'original (known wrong)' if flip else 'confirmed'} axis convention --")
            total_pass = total_n = 0
            for label in dir_labels:
                n_pass, n = counts[flip][label]
                total_pass += n_pass
                total_n += n
                print(f"  {label:12} {n_pass}/{n} = {100*n_pass/max(n,1):.0f}%")
            print(f"  {'TOTAL':12} {total_pass}/{total_n} = {100*total_pass/max(total_n,1):.0f}%")
        if null_counts[1]:
            n_pass, n = null_counts
            print(f"-- null (no swipe detected on either wrist) --")
            print(f"  {'null':12} {n_pass}/{n} = {100*n_pass/max(n,1):.0f}%")
        return

    if not args.npz or not args.label:
        ap.error('--npz/--label required unless --batch-dir is given')

    features, traj = load_traj(args.npz)
    if args.report:
        for hand, r in traj.items():
            print(f"  {hand:6} dx={r['dx']:+.3f} dy={r['dy']:+.3f} "
                  f"travel={r['travel']:.3f} axis_ratio={r['axis_ratio']:.2f} "
                  f"{'PASS' if r['passes'] else 'reject'}")
        flipped = direction_ok(args.label, traj, flip=True)
        print(f"  (original, known-wrong convention would say: {'PASS' if flipped else 'REJECT'})")

    ok = direction_ok(args.label, traj, flip=False)
    print(f"[{args.label}] {'PASS' if ok else 'REJECT'}")
    if ok and args.out:
        np.savez(args.out, features=features[None], label=args.label)
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
