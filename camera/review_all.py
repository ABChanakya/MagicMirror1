#!/usr/bin/env python3
"""
review_all.py — check the gesture label and mask faces in a single pass.

Doing these separately means watching all 932 clips twice. Here each clip opens
once with everything visible: the current label, the measured sweep direction,
the wrist path, and the automatically detected face boxes already drawn. Fix
whatever is wrong, press SPACE, move on.

Mouse
  drag                 draw a face box
  right-click a box    delete it

Keys — label (each of these confirms the clip — boxes included — and moves
               straight to the next one, same as pressing SPACE)
  SPACE   the label is already correct
  <- / a  this is actually swipe_left
  -> / d  this is actually swipe_right
  n       this is actually null
  ?       mark the label unclear

Keys — faces
  f       fetch the automatic detection for this clip (off by default — it
          misses people, so boxes are drawn by hand and carried forward)
  u       undo the last box
  c       clear all boxes
  x       no identifiable face in this clip (explicit, so it is not just forgotten)

Keys — navigation
  , / .   step one frame back / forward
  p       play the clip through
  b       back one clip
  q       quit and save

Everything is written to review_all.json after each clip, so quitting loses
nothing and the next run resumes where you stopped.

Nothing is modified while reviewing. The two outputs are applied separately:

    python review_all.py                      # review
    python review_all.py --report             # summary
    python review_all.py --render --yes       # write face-masked copies
    python review_all.py --apply-labels --yes # move mislabelled clips
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
STORE = HERE / 'review_all.json'
VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv'}
GS = HERE.parent / 'gesture_system'


# ── persistence ───────────────────────────────────────────────────────────────

def load_store():
    return json.loads(STORE.read_text()) if STORE.exists() else {}


def save_store(s):
    STORE.write_text(json.dumps(s, indent=2))


# ── clip IO ───────────────────────────────────────────────────────────────────

def read_clip(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def auto_face_boxes(frames, conf=0.10, pad=0.30):
    """
    Union of every detection, split into groups by horizontal position so two
    people stay two boxes. Applied to every frame, not only detected ones — a
    box that covers only the detected frames leaves the rest exposed, and
    anonymisation is judged on its worst frame.
    """
    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                   min_detection_confidence=conf)
    found = []
    for f in frames:
        r = fd.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if r.detections:
            for d in r.detections:
                b = d.location_data.relative_bounding_box
                found.append([b.xmin, b.ymin, b.xmin + b.width, b.ymin + b.height])
    fd.close()
    if not found:
        return []

    a = np.array(found, np.float32)
    a = a[np.argsort(a[:, 0])]
    groups, cur = [], [a[0]]
    for row in a[1:]:
        if row[0] - cur[-1][0] > 0.20:
            groups.append(np.array(cur)); cur = [row]
        else:
            cur.append(row)
    groups.append(np.array(cur))

    out = []
    for g in groups:
        x1, y1, x2, y2 = g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()
        w, h = x2 - x1, y2 - y1
        out.append([float(np.clip(x1 - w * pad, 0, 1)), float(np.clip(y1 - h * pad, 0, 1)),
                    float(np.clip(x2 + w * pad, 0, 1)), float(np.clip(y2 + h * pad, 0, 1))])
    return out


# ── gesture measurement (reused from the landmark cache) ──────────────────────

def load_measurements():
    sys.path.insert(0, str(GS))
    try:
        from review_clips import load_measurements as lm
        cwd = Path.cwd()
        import os
        os.chdir(GS)
        try:
            return lm()
        finally:
            os.chdir(cwd)
    except Exception as e:
        print(f"  (no landmark measurements available: {str(e)[:60]})")
        return {}


def measured_label(meas, classes):
    if not meas:
        return None
    dx = meas['R'] if abs(meas['R']) >= abs(meas['L']) else meas['L']
    if abs(dx) < 0.45:
        return 'null' if 'null' in classes else None
    # labels are user-relative: swiping to your left moves the hand toward
    # image-right on an unmirrored camera
    return 'swipe_left' if dx > 0 else 'swipe_right'


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_ui(frame, boxes, drag, st):
    h, w = frame.shape[:2]
    scale = max(1.0, 720 / w)
    out = cv2.resize(frame, (int(w * scale), int(h * scale)))
    h, w = out.shape[:2]

    for i, b in enumerate(boxes):
        p1, p2 = (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h))
        cv2.rectangle(out, p1, p2, (0, 0, 0), -1)
        cv2.rectangle(out, p1, p2, (0, 200, 255), 2)
        cv2.putText(out, str(i + 1), (p1[0] + 5, p1[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2, cv2.LINE_AA)
    if drag:
        cv2.rectangle(out, drag[0], drag[1], (0, 255, 120), 2)

    trail = st.get('trail') or {}
    for k, colr in (('R', (120, 255, 120)), ('L', (255, 190, 120))):
        pts = trail.get(k)
        if pts is None:
            continue
        prev = None
        for j in range(min(st['t'] + 1, len(pts))):
            x, y = pts[j]
            if not np.isfinite(x):
                prev = None; continue
            p = (int(x * w), int(y * h))
            if prev:
                cv2.line(out, prev, p, colr, 2, cv2.LINE_AA)
            prev = p
        if prev:
            cv2.circle(out, prev, 6, colr, -1, cv2.LINE_AA)

    panel = out.copy()
    cv2.rectangle(panel, (0, 0), (w, 92), (18, 18, 18), -1)
    cv2.rectangle(panel, (0, h - 46), (w, h), (18, 18, 18), -1)
    out = cv2.addWeighted(panel, .75, out, .25, 0)

    def put(t, xy, c=(235, 235, 235), s=.5, th=1):
        cv2.putText(out, t, xy, cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)

    put(st['title'][:74], (10, 20), (255, 255, 255), .5)
    lab = st['label']
    changed = lab != st['orig_label']
    put(f"LABEL: {lab}" + ("  (changed)" if changed else ""), (10, 44),
        (120, 255, 160) if changed else (120, 220, 255), .6, 2)
    if st['measured'] and st['measured'] != lab:
        put(f"measurement says: {st['measured']}", (300, 44), (110, 160, 255), .5)
    face_txt = ("NO FACE marked" if st['no_face']
                else f"{len(boxes)} face box(es)" if boxes else "NO BOX YET")
    put(f"{face_txt}    frame {st['t']+1}/{st['n']}", (10, 68),
        (120, 255, 120) if (boxes or st['no_face']) else (110, 160, 255), .5)

    put("SPACE=correct as-is | <- left  -> right  n null  ? unclear  (each confirms + advances) | "
        "drag=box  right-click=del  u undo  c clear  f auto  x no-face",
        (10, h - 28), (200, 200, 200), .42)
    put(", . frame   p play   b back   q quit", (10, h - 10), (170, 170, 170), .42)
    return out


# ── review loop ───────────────────────────────────────────────────────────────

def review(args):
    raw = HERE / args.src
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    store = load_store()
    meas = load_measurements()

    clips = []
    for c in classes:
        for v in sorted(p for p in (raw / c).iterdir() if p.suffix.lower() in VIDEO_EXT):
            if args.label and c != args.label:
                continue
            clips.append(v)
    if not args.redo:
        clips = [v for v in clips if v.stem not in store]
    if not clips:
        print("nothing left. --redo to revisit, --report for the summary.")
        return

    start_i = 0
    if args.start:
        # 1-indexed, matching the "[i+1/total]" counter shown on screen, so
        # "clip 490" means the 490th one in the same order the reviewer used.
        start_i = max(0, min(args.start - 1, len(clips) - 1))
        print(f"jumping to clip {start_i + 1}/{len(clips)}: {clips[start_i].stem}")

    print(f"{len(clips)} clips to review.  SPACE=confirm  arrows=fix label  drag=face box")
    win = 'review'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    st = {'boxes': [], 'drag': None, 'start': None, 'shape': (1, 1)}

    def on_mouse(ev, x, y, flags, _):
        H, W = st['shape']
        if ev == cv2.EVENT_LBUTTONDOWN:
            st['start'] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st['start']:
            st['drag'] = (st['start'], (x, y))
        elif ev == cv2.EVENT_LBUTTONUP and st['start']:
            x0, y0 = st['start']
            b = [min(x0, x) / W, min(y0, y) / H, max(x0, x) / W, max(y0, y) / H]
            if b[2] - b[0] > .01 and b[3] - b[1] > .01:
                st['boxes'].append(b)
            st['start'] = st['drag'] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            nx, ny = x / W, y / H
            for i in range(len(st['boxes']) - 1, -1, -1):
                b = st['boxes'][i]
                if b[0] <= nx <= b[2] and b[1] <= ny <= b[3]:
                    st['boxes'].pop(i); break

    cv2.setMouseCallback(win, on_mouse)

    carried = []          # boxes carried from the previous clip
    i = start_i
    while 0 <= i < len(clips):
        v = clips[i]
        frames, _ = read_clip(v)
        if not frames:
            i += 1; continue

        prev = store.get(v.stem) if args.redo else None
        if prev:
            st['boxes'] = [list(b) for b in prev['boxes']]
        elif args.auto:
            st['boxes'] = auto_face_boxes(frames, args.conf)
        else:
            # Carry the previous clip's boxes forward. Consecutive clips in a
            # run are the same person standing in the same place, so drawing
            # once per run and confirming the rest is far faster than drawing
            # 932 boxes — and unlike auto-detection it never silently misses
            # someone. `c` clears, `f` fetches the automatic guess.
            st['boxes'] = [list(b) for b in carried]
        label = prev['label'] if prev else v.parent.name
        no_face = bool(prev['no_face']) if prev else False
        t, playing, key = 0, False, None

        while True:
            disp = draw_ui(frames[t], st['boxes'], st['drag'], {
                'title': f"[{i+1}/{len(clips)}] {v.parent.name}/{v.stem}",
                'label': label, 'orig_label': v.parent.name,
                'measured': measured_label(meas.get(v.stem), classes),
                'no_face': no_face, 't': t, 'n': len(frames), 'trail': None})
            st['shape'] = disp.shape[:2]
            cv2.imshow(win, disp)
            k = cv2.waitKey(40 if playing else 20) & 0xFF
            if playing:
                t = (t + 1) % len(frames)
                if t == 0:
                    playing = False
            if k == 255:
                continue
            if k in (ord('q'), 27):  key = 'quit'; break
            if k == ord(' '):        key = 'next'; break            # confirm as-is
            if k == ord('b'):        key = 'back'; break
            if k in (81, ord('a')):  label = 'swipe_left';  key = 'next'; break
            if k in (83, ord('d')):  label = 'swipe_right'; key = 'next'; break
            if k == ord('n'):        label = 'null';        key = 'next'; break
            if k == ord('?'):        label = 'unclear';     key = 'next'; break
            if k == ord('f'):        st['boxes'] += auto_face_boxes(frames, args.conf)
            if k == ord('u') and st['boxes']: st['boxes'].pop()
            if k == ord('c'):        st['boxes'] = []
            if k == ord('x'):        no_face = True; st['boxes'] = []
            if k == ord(','):        t = (t - 1) % len(frames)
            if k == ord('.'):        t = (t + 1) % len(frames)
            if k == ord('p'):        playing = not playing

        if key == 'quit':
            break
        if key == 'back':
            i = max(0, i - 1); continue

        if st['boxes']:
            carried = [list(b) for b in st['boxes']]
        store[v.stem] = {'class': v.parent.name, 'label': label,
                         'boxes': [list(map(float, b)) for b in st['boxes']],
                         'no_face': no_face}
        save_store(store)
        i += 1

    cv2.destroyAllWindows()
    report(args)


# ── outputs ───────────────────────────────────────────────────────────────────

def report(args):
    s = load_store()
    if not s:
        print("nothing reviewed yet"); return
    wrong = {k: r for k, r in s.items() if r['label'] not in (r['class'], 'unclear')}
    unclear = [k for k, r in s.items() if r['label'] == 'unclear']
    unmasked = [k for k, r in s.items() if not r['boxes'] and not r['no_face']]

    print(f"\n{len(s)} clips reviewed")
    print(f"  labels:  {len(s)-len(wrong)-len(unclear)} confirmed, "
          f"{len(wrong)} corrected, {len(unclear)} unclear")
    print(f"  faces:   {sum(1 for r in s.values() if r['boxes'])} masked, "
          f"{sum(1 for r in s.values() if r['no_face'])} marked no-face, "
          f"{len(unmasked)} NEITHER")
    if wrong:
        print("\n  label corrections:")
        for (a, b), n in Counter((r['class'], r['label']) for r in wrong.values()).most_common():
            print(f"    {a:12} -> {b:12}  {n}")
        runs = Counter(re.search(r'(\d{8}-\d{6})', k).group(1)
                       for k in wrong if re.search(r'(\d{8}-\d{6})', k))
        if runs:
            print("\n  by recording run:")
            for run, n in runs.most_common(8):
                tot = sum(1 for k in s if run in k)
                print(f"    {run}  {n}/{tot}")
    if unmasked:
        print(f"\n  {len(unmasked)} clips have no box and no 'no-face' mark — "
              f"render will refuse these")


def render(args):
    raw, out = HERE / args.src, HERE / args.out
    s = load_store()
    if not s:
        print("nothing reviewed yet"); return
    print(f"{'RENDERING' if args.yes else 'DRY RUN'} face-masked copies\n")
    done = skipped = 0
    for stem, r in sorted(s.items()):
        src = raw / r['class'] / f"{stem}.mp4"
        if not src.exists():
            continue
        if not r['boxes'] and not r['no_face']:
            skipped += 1
            print(f"  REFUSED (no box, not marked x)  {stem[:54]}")
            continue
        if args.yes:
            frames, fps = read_clip(src)
            if not frames:
                continue
            h, w = frames[0].shape[:2]
            # the corrected label decides the destination folder
            dst = out / (r['label'] if r['label'] != 'unclear' else r['class']) / f"{stem}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for f in frames:
                for b in r['boxes']:
                    cv2.rectangle(f, (int(b[0] * w), int(b[1] * h)),
                                  (int(b[2] * w), int(b[3] * h)), (0, 0, 0), -1)
                vw.write(f)
            vw.release()
            sc = src.with_suffix('.json')
            if sc.exists():
                shutil.copy2(sc, dst.with_suffix('.json'))
        done += 1
    print(f"\n  {done} clips {'written to ' + str(out) if args.yes else 'would be written'}")
    if skipped:
        print(f"  {skipped} refused — give them a box or press x on them")
    if not args.yes:
        print("\n  re-run with --render --yes")


def apply_labels(args):
    raw = HERE / args.src
    s = load_store()
    moves = [(k, r) for k, r in s.items() if r['label'] not in (r['class'], 'unclear')]
    if not moves:
        print("no label corrections to apply"); return
    print(f"{'MOVING' if args.yes else 'DRY RUN'} — {len(moves)} clips\n")
    for stem, r in sorted(moves):
        src = raw / r['class'] / f"{stem}.mp4"
        dst = raw / r['label'] / f"{stem}.mp4"
        if not src.exists():
            continue
        print(f"  {r['class']:12} -> {r['label']:12}  {stem[:48]}")
        if args.yes:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            js = src.with_suffix('.json')
            if js.exists():
                shutil.move(str(js), str(dst.with_suffix('.json')))
    if args.yes:
        print("\n  moved. The landmark cache is now stale:")
        print("    cd ../gesture_system && python preprocess_holistic.py")
    else:
        print("\n  re-run with --apply-labels --yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='gesture_training_data')
    ap.add_argument('--out', default='gesture_training_data_anon')
    ap.add_argument('--label', help='review only this class')
    ap.add_argument('--conf', type=float, default=0.10)
    ap.add_argument('--auto', action='store_true',
                    help='seed each clip with automatic detection. Off by default: '
                         'it misses people, and a missed face defeats the purpose. '
                         'Press f on any clip to fetch it on demand.')
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--start', type=int,
                    help='jump straight to this clip number (1-indexed, matching '
                         'the on-screen [i/total] counter). Implies --redo, since '
                         'an already-reviewed clip is normally skipped.')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--apply-labels', action='store_true')
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()
    if args.start:
        args.redo = True

    if args.report:        report(args)
    elif args.render:      render(args)
    elif args.apply_labels: apply_labels(args)
    else:                  review(args)


if __name__ == '__main__':
    main()
