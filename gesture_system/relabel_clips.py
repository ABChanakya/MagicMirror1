#!/usr/bin/env python3
"""
relabel_clips.py — watch each clip and confirm or correct its label.

The classifier reads `swipe_left` well (77-86% recall) and `swipe_right` badly
(21-56%), which is consistent either with a genuinely harder class or with some
clips being filed under the wrong direction. This tool settles that by hand.

Each clip plays looping with its current label and the automatically measured
sweep direction shown. You either confirm the label or say what it should be.

Keys
  SPACE       the label is correct
  <-  /  a    this is actually swipe_left
  ->  /  d    this is actually swipe_right
  n           this is actually null (no deliberate gesture)
  u           unclear, decide later
  b           back one clip
  s           skip
  q           quit and save

Verdicts are written to relabel_verdicts.json after every keypress, so quitting
mid-way loses nothing and you can resume where you left off.

Nothing is moved while reviewing. Applying corrections is a separate, explicit
step that shows a dry run first:

    python relabel_clips.py                 # review
    python relabel_clips.py --report        # summary of what you marked
    python relabel_clips.py --apply         # dry run of the file moves
    python relabel_clips.py --apply --yes   # actually move them

Usage
-----
python relabel_clips.py                      # every clip, resumes automatically
python relabel_clips.py --label swipe_right  # just the suspect class
python relabel_clips.py --run 20260816-161128
python relabel_clips.py --disagree           # only where measurement != label
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from review_clips import draw, load_measurements, wrist_pixels

VERDICTS = Path('relabel_verdicts.json')
TRAILS   = Path('data/wrist_trails.npz')


def build_trails(cfg, limit=0):
    """
    Precompute the drawn wrist path for every clip.

    Drawing the path needs image-space landmarks, and MediaPipe Holistic costs
    ~1.2 s per clip. Doing that lazily makes a 932-clip pass feel like waiting
    rather than reviewing, so it is done once up front and cached.
    """
    from tqdm import tqdm
    raw = Path(cfg['data']['raw_dir'])
    classes = [str(c) for c in cfg['data']['classes']]
    vids = []
    for c in classes:
        vids += sorted((raw / c).glob('*.mp4'))
    if limit:
        vids = vids[:limit]

    out = {}
    for v in tqdm(vids, desc='trails'):
        t = wrist_pixels(v, num_frames=30)
        if t is None:
            continue
        out[v.stem + '|R'] = t['R'].astype('float32')
        out[v.stem + '|L'] = t['L'].astype('float32')
    TRAILS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRAILS, **out)
    print(f"cached {len(out)//2} trails -> {TRAILS} "
          f"({TRAILS.stat().st_size/1e6:.1f} MB)")


def load_trails():
    if not TRAILS.exists():
        return None
    d = np.load(TRAILS)
    return {k: d[k] for k in d.files}


def load_verdicts():
    return json.loads(VERDICTS.read_text()) if VERDICTS.exists() else {}


def save_verdicts(v):
    VERDICTS.write_text(json.dumps(v, indent=2))


def measured_label(meas, classes):
    """Which class the automatic sweep measurement implies, or None."""
    if not meas:
        return None
    dx = meas['R'] if abs(meas['R']) >= abs(meas['L']) else meas['L']
    if abs(dx) < 0.45:
        return 'null' if 'null' in classes else None
    # labels are user-relative: swiping to your left moves the hand toward
    # image-right on an unmirrored camera
    return 'swipe_left' if dx > 0 else 'swipe_right'


def collect(cfg, args, meas):
    raw = Path(cfg['data']['raw_dir'])
    classes = [str(c) for c in cfg['data']['classes']]
    clips = []
    for cls in classes:
        if args.label and cls != args.label:
            continue
        for v in sorted((raw / cls).glob('*.mp4')):
            m = re.search(r'(\d{8}-\d{6})', v.name)
            run = m.group(1) if m else '?'
            if args.run and run != args.run:
                continue
            m_lab = measured_label(meas.get(v.stem), classes)
            if args.disagree and (m_lab is None or m_lab == cls):
                continue
            clips.append({'path': v, 'name': v.stem, 'label': cls, 'run': run,
                          'meas': meas.get(v.stem), 'measured_label': m_lab})
    return clips


def review(args, cfg, meas):
    classes = [str(c) for c in cfg['data']['classes']]
    trails = load_trails()
    if trails is None:
        print("no trail cache — paths will be computed live (~1.2 s per clip).")
        print("For a full pass run this first:  python relabel_clips.py --precompute")
    verdicts = load_verdicts()
    clips = collect(cfg, args, meas)

    if not args.redo:
        pending = [c for c in clips if c['name'] not in verdicts]
        done = len(clips) - len(pending)
        if done:
            print(f"resuming — {done} already reviewed, {len(pending)} to go")
        clips = pending

    if not clips:
        print("nothing left to review. --report for the summary, --redo to start over.")
        return

    print(f"{len(clips)} clips. SPACE=correct  <-=left  ->=right  n=null  u=unclear  b=back  q=quit")
    cv2.namedWindow('relabel', cv2.WINDOW_NORMAL)
    delay = max(1, int(1000 / args.fps))
    i = 0

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

        if trails is not None:
            r, l = trails.get(c['name'] + '|R'), trails.get(c['name'] + '|L')
            trail = {} if r is None else {'R': r, 'L': l}
        else:
            trail = wrist_pixels(c['path'], num_frames=len(frames)) or {}
        key, t = None, 0
        while True:
            f = draw(frames[t % len(frames)], i, len(clips), c, trail, t % len(frames))
            h, w = f.shape[:2]
            hint = f"current label: {c['label']}"
            if c['measured_label'] and c['measured_label'] != c['label']:
                hint += f"   MEASURED SAYS: {c['measured_label']}"
            cv2.rectangle(f, (0, h - 60), (w, h - 34), (18, 18, 18), -1)
            cv2.putText(f, hint, (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 220, 255) if c['measured_label'] == c['label'] else (110, 160, 255),
                        1, cv2.LINE_AA)
            cv2.imshow('relabel', f)
            t += 1
            k = cv2.waitKey(delay) & 0xFF
            if k in (ord('q'), 27):
                key = 'quit'; break
            if k == ord(' '):
                key = c['label']; break                 # confirmed as-is
            if k in (81, ord('a')):
                key = 'swipe_left'; break
            if k in (83, ord('d')):
                key = 'swipe_right'; break
            if k == ord('n'):
                key = 'null'; break
            if k == ord('u'):
                key = 'unclear'; break
            if k == ord('b'):
                key = 'back'; break
            if k == ord('s'):
                key = 'skip'; break

        if key == 'quit':
            break
        if key == 'back':
            i = max(0, i - 1)
            continue
        if key != 'skip':
            verdicts[c['name']] = {'was': c['label'], 'should_be': key,
                                   'run': c['run'],
                                   'measured': c['measured_label']}
            save_verdicts(verdicts)
        i += 1

    cv2.destroyAllWindows()
    print(f"\nsaved {len(verdicts)} verdicts -> {VERDICTS}")
    report(cfg)


def report(cfg):
    classes = [str(c) for c in cfg['data']['classes']]
    v = load_verdicts()
    if not v:
        print("no verdicts yet")
        return
    ok = [k for k, d in v.items() if d['should_be'] == d['was']]
    bad = {k: d for k, d in v.items() if d['should_be'] not in (d['was'], 'unclear')}
    unclear = [k for k, d in v.items() if d['should_be'] == 'unclear']

    print(f"\n{len(v)} reviewed:  {len(ok)} correct, {len(bad)} MISLABELLED, {len(unclear)} unclear")
    if bad:
        print("\n  corrections by direction:")
        for (a, b), n in Counter((d['was'], d['should_be']) for d in bad.values()).most_common():
            print(f"    {a:12} -> {b:12}  {n}")
        print("\n  which runs they came from:")
        for run, n in Counter(d['run'] for d in bad.values()).most_common(8):
            tot = sum(1 for d in v.values() if d['run'] == run)
            print(f"    {run}  {n}/{tot} wrong")

    agree = sum(1 for d in v.values()
                if d.get('measured') and d['measured'] == d['should_be'])
    hasm = sum(1 for d in v.values() if d.get('measured'))
    if hasm:
        print(f"\n  automatic measurement matched your verdict: {agree}/{hasm} "
              f"({agree/hasm*100:.0f}%)")
        print("  -> if this is high, the measurement can flag the rest without you watching them")


def apply(cfg, yes):
    raw = Path(cfg['data']['raw_dir'])
    v = load_verdicts()
    moves = [(k, d) for k, d in v.items()
             if d['should_be'] not in (d['was'], 'unclear')]
    if not moves:
        print("nothing to move")
        return
    print(f"{'MOVING' if yes else 'DRY RUN'} — {len(moves)} clips\n")
    for name, d in moves:
        src = raw / d['was'] / f"{name}.mp4"
        dst = raw / d['should_be'] / f"{name}.mp4"
        js, jd = src.with_suffix('.json'), dst.with_suffix('.json')
        if not src.exists():
            print(f"  MISSING {src}")
            continue
        print(f"  {d['was']:12} -> {d['should_be']:12}  {name[:46]}")
        if yes:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            if js.exists():
                shutil.move(str(js), str(jd))   # keep the sidecar with its clip
    if yes:
        print("\nmoved. The landmark cache is now stale — regenerate before training:")
        print("  python preprocess_holistic.py")
    else:
        print("\nre-run with --apply --yes to actually move them")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--label')
    ap.add_argument('--run')
    ap.add_argument('--disagree', action='store_true',
                    help='only clips where the measurement disagrees with the label')
    ap.add_argument('--redo', action='store_true', help='review already-done clips again')
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--yes', action='store_true')
    ap.add_argument('--precompute', action='store_true',
                    help='cache wrist paths for every clip (~20 min once, then '
                         'review runs with no per-clip delay)')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.precompute:
        build_trails(cfg); return
    if args.report:
        report(cfg); return
    if args.apply:
        apply(cfg, args.yes); return
    review(args, cfg, load_measurements())


if __name__ == '__main__':
    main()
