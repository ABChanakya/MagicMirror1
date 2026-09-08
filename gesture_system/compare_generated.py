#!/usr/bin/env python3
"""
compare_generated.py — stitch original and generated clips side by side.

Numbers said the motion drifts at high denoise strength and the output is too
noisy for pose detection at low strength. This puts the source and every
generated strength in one frame so that judgement can be made by eye, which is
the thing the validator cannot do: whether the person still looks like a person,
whether the gesture reads as the same gesture, and what specifically went wrong.

Writes one .mp4 per source clip into <out>/, each panel labelled with its
strength and the validator's verdict.

Usage
-----
python compare_generated.py --sweep /media/bhaskara/Volume/gen_sweep
python compare_generated.py --sweep ... --out /media/bhaskara/Volume/compare
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_frames(path, n=25, size=256):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return None
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


def label(img, top, bottom=None, colr=(255, 255, 255)):
    h, w = img.shape[:2]
    out = img.copy()
    cv2.rectangle(out, (0, 0), (w, 26), (20, 20, 20), -1)
    cv2.putText(out, top, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colr, 1, cv2.LINE_AA)
    if bottom:
        cv2.rectangle(out, (0, h - 24), (w, h), (20, 20, 20), -1)
        cv2.putText(out, bottom[:34], (7, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, colr, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='/media/bhaskara/Volume/gen_sweep')
    ap.add_argument('--out', default='/media/bhaskara/Volume/compare')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--frames', type=int, default=25)
    ap.add_argument('--fps', type=int, default=12)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw = Path(cfg['data']['raw_dir'])
    sweep = Path(args.sweep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    strengths = sorted([d for d in sweep.iterdir() if d.is_dir()
                        and re.fullmatch(r's\d+', d.name)],
                       key=lambda p: int(p.name[1:]))
    if not strengths:
        strengths = [sweep]

    # map source stem -> generated file per strength
    jobs = {}
    for sd in strengths:
        for cls_dir in sd.iterdir():
            if not cls_dir.is_dir():
                continue
            for g in cls_dir.glob('gen_*.mp4'):
                stem = g.stem[len('gen_'):].rsplit('_v', 1)[0]
                jobs.setdefault((stem, cls_dir.name), {})[sd.name] = g

    if not jobs:
        print(f"no generated clips under {sweep}")
        return

    print(f"building {len(jobs)} comparisons -> {out}\n")
    for (stem, cls), variants in sorted(jobs.items()):
        src = raw / cls / f"{stem}.mp4"
        if not src.exists():
            print(f"  skip {stem}: source missing")
            continue
        orig = read_frames(src, args.frames, args.size)
        if orig is None:
            continue

        cols = [[label(f, "ORIGINAL", cls, (120, 255, 120)) for f in orig]]
        heads = ["ORIGINAL"]
        for sd in strengths:
            g = variants.get(sd.name)
            if not g:
                continue
            fr = read_frames(g, args.frames, args.size)
            if fr is None:
                continue
            st = f"strength 0.{sd.name[1:]}"
            cols.append([label(f, st, "generated", (120, 200, 255)) for f in fr])
            heads.append(st)

        if len(cols) < 2:
            continue
        n = min(len(c) for c in cols)
        h, w = cols[0][0].shape[:2]
        dest = out / f"cmp_{cls}_{stem[:40]}.mp4"
        vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'),
                             args.fps, (w * len(cols), h))
        for t in range(n):
            vw.write(np.hstack([c[t] for c in cols]))
        vw.release()
        print(f"  {dest.name}   ({' | '.join(heads)})")

    print(f"\nwatch them with:\n  mpv {out}/*.mp4    (or vlc / xdg-open)")


if __name__ == '__main__':
    main()
