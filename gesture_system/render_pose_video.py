#!/usr/bin/env python3
"""
render_pose_video.py — draw the extracted skeleton as a driving-pose video.

WanAnimatePipeline takes `image` (who the person should look like) and
`pose_video` (what they should do). The pose video is a stick figure on black,
which is exactly what our Holistic extraction already provides — 56 points per
frame for all 932 clips.

This is the piece that makes generated clips trustworthy. Image-to-video failed
(0/3) because a single frame plus a text prompt gives the model no way to know
which gesture to perform, so it improvised — including producing a large sweep
for a clip labelled `null`. Driving generation from the real skeleton removes
that failure entirely: the motion is the recorded motion, so the label stays
correct by construction.

Limbs are coloured in the OpenPose convention, which is what pose-conditioned
video models are generally trained against.

Usage
-----
python render_pose_video.py --clip swipe_left/swipe_left_chanakya_20260816-165202_092
python render_pose_video.py --all --limit 20 --out /media/bhaskara/Volume/pose_videos
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

# indices within the kept pose block (POSE_KEEP starts at pose landmark 11)
P = {n: i - 11 for i, n in [(11, 'l_shoulder'), (12, 'r_shoulder'),
                            (13, 'l_elbow'), (14, 'r_elbow'),
                            (15, 'l_wrist'), (16, 'r_wrist'),
                            (23, 'l_hip'), (24, 'r_hip')]}
N_POSE = 14
HAND_L0, HAND_R0 = N_POSE, N_POSE + 21

# OpenPose-style limb colours (BGR)
LIMBS = [
    (P['l_shoulder'], P['r_shoulder'], (0, 85, 255)),
    (P['l_shoulder'], P['l_elbow'], (0, 170, 255)),
    (P['l_elbow'], P['l_wrist'], (0, 255, 255)),
    (P['r_shoulder'], P['r_elbow'], (85, 255, 170)),
    (P['r_elbow'], P['r_wrist'], (170, 255, 85)),
    (P['l_shoulder'], P['l_hip'], (255, 170, 0)),
    (P['r_shoulder'], P['r_hip'], (255, 85, 0)),
    (P['l_hip'], P['r_hip'], (255, 0, 0)),
]
# 21-point MediaPipe hand skeleton
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13),
              (13, 17)]


def render(seq_xy, w=704, h=480, thickness=4):
    """
    seq_xy: (T, 56, 2) normalised image coordinates -> list of BGR frames.

    Points that were never detected arrive as NaN and are simply not drawn,
    rather than being snapped to the origin, which would otherwise create a
    limb shooting to the top-left corner.
    """
    out = []
    for t in range(seq_xy.shape[0]):
        canvas = np.zeros((h, w, 3), np.uint8)
        pts = seq_xy[t]

        def px(i):
            x, y = pts[i]
            if not np.isfinite(x) or not np.isfinite(y):
                return None
            return int(np.clip(x, 0, 1) * (w - 1)), int(np.clip(y, 0, 1) * (h - 1))

        for a, b, colr in LIMBS:
            pa, pb = px(a), px(b)
            if pa and pb:
                cv2.line(canvas, pa, pb, colr, thickness, cv2.LINE_AA)
        for i in range(N_POSE):
            p = px(i)
            if p:
                cv2.circle(canvas, p, thickness + 1, (255, 255, 255), -1, cv2.LINE_AA)

        for base, colr in ((HAND_L0, (200, 200, 255)), (HAND_R0, (255, 200, 200))):
            for a, b in HAND_EDGES:
                pa, pb = px(base + a), px(base + b)
                if pa and pb:
                    cv2.line(canvas, pa, pb, colr, max(1, thickness - 2), cv2.LINE_AA)
            for i in range(21):
                p = px(base + i)
                if p:
                    cv2.circle(canvas, p, 2, colr, -1, cv2.LINE_AA)
        out.append(canvas)
    return out


def raw_landmarks(video_path, num_frames):
    """Re-extract in image coordinates. The cache stores torso-normalised values,
    which are the right input for the classifier but cannot be drawn."""
    from dataset import _extract_frames
    from preprocess_holistic import _init_holistic, extract_frame
    frames = _extract_frames(str(video_path), num_frames=num_frames)
    if frames is None:
        return None
    hol = _init_holistic()
    seq = np.stack([extract_frame(hol, f) for f in frames])   # (T,56,3) NaN where missing
    return seq[:, :, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--clip', help='<class>/<stem>')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='/media/bhaskara/Volume/pose_videos')
    ap.add_argument('--frames', type=int, default=25)
    ap.add_argument('--width', type=int, default=704)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--side-by-side', action='store_true',
                    help='also write source|skeleton, to confirm the pose tracks')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw = Path(cfg['data']['raw_dir'])
    classes = [str(c) for c in cfg['data']['classes']]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.clip:
        cls, stem = args.clip.split('/', 1)
        targets = [(raw / cls / f"{stem}.mp4", cls)]
    else:
        for c in classes:
            vids = sorted((raw / c).glob('*.mp4'))
            targets += [(v, c) for v in (vids[:args.limit] if args.limit else vids)]

    print(f"rendering {len(targets)} pose videos -> {out}\n")
    made = 0
    for src, cls in targets:
        xy = raw_landmarks(src, args.frames)
        if xy is None:
            print(f"  skip {src.name}: unreadable")
            continue
        detected = np.isfinite(xy[:, P['l_shoulder'], 0]).mean()
        frames = render(xy, args.width, args.height)

        dest = out / cls
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / f"pose_{src.stem}.mp4"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), 12,
                             (args.width, args.height))
        for f in frames:
            vw.write(f)
        vw.release()

        if args.side_by_side:
            from dataset import _extract_frames
            orig = _extract_frames(str(src), num_frames=args.frames)
            cmp_path = dest / f"cmp_{src.stem}.mp4"
            vw = cv2.VideoWriter(str(cmp_path), cv2.VideoWriter_fourcc(*'mp4v'), 12,
                                 (args.width * 2, args.height))
            for i in range(min(len(frames), len(orig))):
                o = cv2.resize(cv2.cvtColor(orig[i], cv2.COLOR_RGB2BGR),
                               (args.width, args.height))
                vw.write(np.hstack([o, frames[i]]))
            vw.release()

        made += 1
        print(f"  {cls:12} {p.name}   pose found in {detected*100:.0f}% of frames")

    print(f"\nrendered {made} pose videos")
    if args.side_by_side:
        print("check one before generating:")
        print(f"  mpv {out}/*/cmp_*.mp4")


if __name__ == '__main__':
    main()
