#!/usr/bin/env python3
"""
review_clips.py — watch clips one at a time and record what you actually see.

Shows each clip looping, with the wrist path drawn on top and the measured
direction printed, so you can see exactly what the analysis is reading. You mark
which way the hand really moved; the tool then compares your verdicts against
both the stored label and the automatic measurement.

That comparison answers three separate questions at once:

  your verdict vs label        -> are any clips mislabelled?
  your verdict vs measurement  -> is the trajectory extraction wrong?
  label vs measurement         -> which clips to look at first

Keys
  <-  /  a      hand moved LEFT on screen
  ->  /  d      hand moved RIGHT on screen
  u             unclear / ambiguous
  n             no real hand motion
  space         replay
  s             skip, decide later
  b             back one clip
  q             quit and save

Usage
-----
python review_clips.py --suspicious          # the disputed runs first (default)
python review_clips.py --run 20260816-161128
python review_clips.py --label swipe_right --limit 40
python review_clips.py --report              # summarise saved verdicts, no video
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

VERDICTS = Path('review_verdicts.json')

# Runs where the stored label and the measured direction disagree — worth
# looking at before anything else.
SUSPICIOUS = ['20260816-161128', '20260816-164936', '20260816-160851',
              '20260816-164335', '20260816-165432', '20260825-164612',
              '20260825-164024', '20260816-170517']


def load_measurements(cache='data/landmarks_holistic.npz'):
    """clip stem -> measured horizontal travel of each wrist."""
    p = Path(cache)
    if not p.exists():
        return {}
    d = np.load(p, allow_pickle=True)
    X, cid = d['X'], d['clip_ids']
    NS = int(d['n_shape'])

    def sustained(sig):
        """
        Signed size of the longest same-direction run.

        Clips contain a wind-up: the hand pulls back one way before sweeping the
        other. Peak excursion and net displacement both pick that up and can
        report the opposite of the actual gesture — they agree with the labels
        only 68% of the time, against 91% for this measure.
        """
        v = np.diff(sig, axis=1)
        out = np.zeros(len(sig))
        for n in range(len(sig)):
            best = cur = 0.0
            cs = 0
            for x in v[n]:
                if np.sign(x) == cs or cs == 0:
                    cur += x
                    cs = np.sign(x) if x != 0 else cs
                else:
                    cur = x
                    cs = np.sign(x)
                if abs(cur) > abs(best):
                    best = cur
            out[n] = best
        return out

    Rx, Lx = sustained(X[:, :, NS + 3]), sustained(X[:, :, NS])
    out = {}
    for i, c in enumerate(cid):
        out[str(c).split('/')[-1]] = {'R': float(Rx[i]), 'L': float(Lx[i])}
    return out


def wrist_pixels(path, num_frames=30):
    """Per-frame wrist positions in pixels, for drawing the path on the video."""
    try:
        from dataset import _extract_frames
        from preprocess_holistic import (L_WRIST, R_WRIST, _init_holistic,
                                         extract_frame)
    except Exception:
        return None
    frames = _extract_frames(str(path), num_frames=num_frames)
    if frames is None:
        return None
    hol = _init_holistic()
    seq = np.stack([extract_frame(hol, f) for f in frames])
    return {'R': seq[:, R_WRIST, :2], 'L': seq[:, L_WRIST, :2]}


def draw(frame, i, total, meta, trail, t):
    h, w = frame.shape[:2]
    scale = max(1.0, 640 / w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    h, w = frame.shape[:2]

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 96), (18, 18, 18), -1)
    cv2.rectangle(panel, (0, h - 34), (w, h), (18, 18, 18), -1)
    frame = cv2.addWeighted(panel, 0.72, frame, 0.28, 0)

    def put(txt, xy, col=(240, 240, 240), sc=0.5, th=1):
        cv2.putText(frame, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    put(f"[{i+1}/{total}]  {meta['name'][:52]}", (10, 22), (255, 255, 255), 0.52)
    put(f"label: {meta['label']}", (10, 44), (120, 220, 255), 0.55, 2)
    put(f"run:   {meta['run']}", (200, 44), (170, 170, 170), 0.45)

    m = meta.get('meas')
    if m:
        act = 'RIGHT' if abs(m['R']) >= abs(m['L']) else 'LEFT'
        dx = m['R'] if act == 'RIGHT' else m['L']
        arrow = 'sweeps RIGHT ->' if dx > 0 else '<- sweeps LEFT'
        col = (120, 255, 120) if abs(dx) > 0.4 else (120, 200, 255)
        put(f"measured: {act} hand, {arrow}  (dx={dx:+.2f})", (10, 68), col, 0.5)
        put("sustained sweep, ignoring wind-up   |   green=right hand  blue=left",
            (10, 88), (150, 150, 150), 0.4)

    for key, colr in (('R', (120, 255, 120)), ('L', (255, 190, 120))):
        pts = trail.get(key)
        if pts is None:
            continue
        prev = None
        for k in range(min(t + 1, len(pts))):
            x, y = pts[k]
            if np.isnan(x):
                prev = None
                continue
            p = (int(x * w), int(y * h))
            if prev is not None:
                cv2.line(frame, prev, p, colr, 2, cv2.LINE_AA)
            prev = p
        if prev is not None:
            cv2.circle(frame, prev, 6, colr, -1, cv2.LINE_AA)

    put("<-/a LEFT   ->/d RIGHT   u unclear   n no-motion   space replay   b back   s skip   q quit",
        (10, h - 12), (200, 200, 200), 0.42)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--run')
    ap.add_argument('--label')
    ap.add_argument('--suspicious', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--per-run', type=int, default=4)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw = Path(cfg['data']['raw_dir'])
    classes = [str(c) for c in cfg['data']['classes']]
    meas = load_measurements()

    verdicts = json.loads(VERDICTS.read_text()) if VERDICTS.exists() else {}

    if args.report:
        report(verdicts, meas)
        return

    clips = []
    for cls in classes:
        if args.label and cls != args.label:
            continue
        for v in sorted((raw / cls).glob('*.mp4')):
            m = re.search(r'(\d{8}-\d{6})', v.name)
            run = m.group(1) if m else '?'
            if args.run and run != args.run:
                continue
            if args.suspicious and not args.all and run not in SUSPICIOUS:
                continue
            clips.append({'path': v, 'name': v.stem, 'label': cls, 'run': run,
                          'meas': meas.get(v.stem)})

    if not args.run and not args.all and args.per_run:
        byrun, capped = {}, []
        for c in clips:
            byrun.setdefault((c['run'], c['label']), []).append(c)
        for k in sorted(byrun):
            capped += byrun[k][:args.per_run]
        clips = capped

    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        print("no clips matched")
        return

    print(f"{len(clips)} clips queued. Window opens next — keep it focused for keys to register.")
    cv2.namedWindow('review', cv2.WINDOW_NORMAL)

    i = 0
    delay = max(1, int(1000 / args.fps))
    while 0 <= i < len(clips):
        c = clips[i]
        cap = cv2.VideoCapture(str(c['path']))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            i += 1
            continue

        trail = wrist_pixels(c['path'], num_frames=len(frames)) or {}
        key, t = None, 0
        while True:
            f = draw(frames[t % len(frames)], i, len(clips), c, trail, t % len(frames))
            cv2.imshow('review', f)
            t += 1
            k = cv2.waitKey(delay) & 0xFF
            if k in (ord('q'), 27):
                key = 'quit'; break
            if k in (81, ord('a')):
                key = 'left'; break
            if k in (83, ord('d')):
                key = 'right'; break
            if k == ord('u'):
                key = 'unclear'; break
            if k == ord('n'):
                key = 'no_motion'; break
            if k == ord('s'):
                key = 'skip'; break
            if k == ord('b'):
                key = 'back'; break
            if k == ord(' '):
                t = 0

        if key == 'quit':
            break
        if key == 'back':
            i = max(0, i - 1)
            continue
        if key != 'skip':
            verdicts[c['name']] = {'saw': key, 'label': c['label'], 'run': c['run']}
            VERDICTS.write_text(json.dumps(verdicts, indent=2))
        i += 1

    cv2.destroyAllWindows()
    print(f"\nsaved {len(verdicts)} verdicts to {VERDICTS}")
    report(verdicts, meas)


def report(verdicts, meas):
    if not verdicts:
        print("no verdicts recorded yet")
        return
    print(f"\n=== {len(verdicts)} clips reviewed ===\n")

    agree_lab = dis_lab = 0
    agree_meas = dis_meas = 0
    per_run = {}
    for name, v in verdicts.items():
        saw, lab, run = v['saw'], v['label'], v['run']
        if saw in ('unclear', 'no_motion'):
            continue
        # what the label implies on screen, under each convention
        user_rel = 'right' if lab == 'swipe_left' else 'left'      # user-relative
        cam_rel = 'left' if lab == 'swipe_left' else 'right'       # camera-relative
        d = per_run.setdefault(run, {'user': 0, 'cam': 0, 'n': 0, 'label': lab})
        d['n'] += 1
        d['user'] += (saw == user_rel)
        d['cam'] += (saw == cam_rel)

        m = meas.get(name)
        if m:
            dx = m['R'] if abs(m['R']) >= abs(m['L']) else m['L']
            meas_dir = 'right' if dx > 0 else 'left'
            agree_meas += (meas_dir == saw)
            dis_meas += (meas_dir != saw)

    print(f"{'run':18} {'label':12} {'n':>3} {'matches user-rel':>17} {'matches camera-rel':>19}")
    for run in sorted(per_run):
        d = per_run[run]
        print(f"  {run:16} {d['label']:12} {d['n']:3} "
              f"{d['user']/max(d['n'],1)*100:16.0f}% {d['cam']/max(d['n'],1)*100:18.0f}%")

    tot = agree_meas + dis_meas
    if tot:
        print(f"\n  your eyes vs automatic measurement: {agree_meas}/{tot} agree "
              f"({agree_meas/tot*100:.0f}%)")
        if agree_meas / tot < 0.85:
            print("  -> the trajectory extraction disagrees with you often; the bug is")
            print("     in the measurement, not the labels.")
        else:
            print("  -> measurement tracks what you see, so direction differences between")
            print("     runs are real and live in the labels.")


if __name__ == '__main__':
    main()
