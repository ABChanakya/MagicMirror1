#!/usr/bin/env python3
"""
make_demo_reel.py — one presentable video showing what the pipeline extracts.

Builds a single reel: for each gesture class, the source recording beside the
56-point skeleton derived from it, with a title card naming the class and the
pose-detection rate for that clip.

The point it makes visually is the one that is hard to convey in a table: the
classifier never sees the person. It sees the stick figure. Body proportions,
clothing, lighting and background are gone before the model receives anything,
and no facial landmarks are extracted at all.

Usage
-----
python make_demo_reel.py
python make_demo_reel.py --per-class 3 --out /media/bhaskara/Volume/demo_reel.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from render_pose_video import P, raw_landmarks, render

W, H = 640, 480
FPS = 12


def title_card(text, sub, w, h, frames=18):
    out = []
    for _ in range(frames):
        c = np.zeros((h, w, 3), np.uint8)
        c[:] = (24, 22, 20)
        cv2.putText(c, text, (40, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.15, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(c, sub, (40, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 200, 255), 1, cv2.LINE_AA)
        out.append(c)
    return out


def banner(img, left_text, right_text, note=None):
    h, w = img.shape[:2]
    o = img.copy()
    cv2.rectangle(o, (0, 0), (w, 34), (18, 18, 18), -1)
    cv2.putText(o, left_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (120, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(o, right_text, (w // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (120, 200, 255), 2, cv2.LINE_AA)
    if note:
        cv2.rectangle(o, (0, h - 30), (w, h), (18, 18, 18), -1)
        cv2.putText(o, note, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--per-class', type=int, default=2)
    ap.add_argument('--frames', type=int, default=25)
    ap.add_argument('--out', default='/media/bhaskara/Volume/demo_reel.mp4')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw = Path(cfg['data']['raw_dir'])
    classes = [str(c) for c in cfg['data']['classes']]

    from dataset import _extract_frames

    reel = []
    reel += title_card("Gesture pipeline", "recording  ->  56-point skeleton", W * 2, H, 24)

    for cls in classes:
        vids = sorted((raw / cls).glob('*.mp4'))
        if not vids:
            continue
        picked, i = [], 0
        # prefer clips the pose tracker handles well — this is a demo, not a
        # failure showcase; the detection rate is printed either way
        while len(picked) < args.per_class and i < min(len(vids), 12):
            v = vids[i * 3 % len(vids)]
            xy = raw_landmarks(v, args.frames)
            if xy is not None:
                det = float(np.isfinite(xy[:, P['l_shoulder'], 0]).mean())
                if det > 0.9 or i > 8:
                    picked.append((v, xy, det))
            i += 1

        for v, xy, det in picked:
            skel = render(xy, W, H)
            orig = _extract_frames(str(v), num_frames=args.frames)
            if orig is None:
                continue
            reel += title_card(cls, f"{v.stem[:46]}", W * 2, H, 14)
            note = (f"pose tracked in {det*100:.0f}% of frames   |   "
                    f"no facial landmarks extracted")
            for t in range(min(len(orig), len(skel))):
                o = cv2.resize(cv2.cvtColor(orig[t], cv2.COLOR_RGB2BGR), (W, H))
                reel.append(banner(np.hstack([o, skel[t]]),
                                   "RECORDING", "WHAT THE MODEL SEES", note))
            print(f"  {cls:12} {v.stem[:44]:46} pose {det*100:3.0f}%")

    if not reel:
        print("nothing to render")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W * 2, H))
    for f in reel:
        vw.write(f)
    vw.release()
    secs = len(reel) / FPS
    print(f"\nwrote {out}  ({secs:.0f}s, {out.stat().st_size/1e6:.1f} MB)")
    print(f"  mpv {out}")


if __name__ == '__main__':
    main()
