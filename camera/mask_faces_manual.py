#!/usr/bin/env python3
"""
mask_faces_manual.py — draw the black boxes by hand where detection is not enough.

Automatic detection is a starting point, not a guarantee. On this dataset it
finds a face in ~90% of frames at conf=0.10, and picks up a second person only
sometimes — so anyone standing further back, turned away, or partly out of frame
can be missed entirely. Since anonymisation is judged on its worst frame, a
manual pass is the only way to be sure.

Workflow: every clip opens with the automatic boxes already drawn. Adjust what
is wrong, add what was missed, confirm, move on. Boxes are stored per clip in
face_boxes.json and applied to every frame of that clip.

Mouse
  drag              draw a new box
  right-click box   delete that box

Keys
  a       add the automatic detections for this clip
  u       undo the last box
  c       clear every box
  , / .   step back / forward a frame (check the box holds for the whole clip)
  p       play the clip through once
  SPACE   confirm this clip and go to the next
  x       mark this clip as needing NO box (nobody identifiable)
  b       back one clip
  q       quit and save

    python mask_faces_manual.py                 # review every clip
    python mask_faces_manual.py --no-face       # only clips detection found nothing in
    python mask_faces_manual.py --render        # dry run of the blacked output
    python mask_faces_manual.py --render --yes  # write the anonymised copies
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

BOXES = Path(__file__).parent / 'face_boxes.json'
VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv'}


def load_boxes():
    return json.loads(BOXES.read_text()) if BOXES.exists() else {}


def save_boxes(b):
    BOXES.write_text(json.dumps(b, indent=2))


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


def auto_boxes(frames, conf=0.10):
    """Union of every automatic detection, padded — the same rule as the batch tool."""
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
    # cluster crudely by horizontal position so two people stay two boxes
    order = np.argsort(a[:, 0])
    a = a[order]
    groups, cur = [], [a[0]]
    for row in a[1:]:
        if row[0] - cur[-1][0] > 0.20:      # a new person, not the same face drifting
            groups.append(np.array(cur)); cur = [row]
        else:
            cur.append(row)
    groups.append(np.array(cur))

    out = []
    for g in groups:
        x1, y1 = g[:, 0].min(), g[:, 1].min()
        x2, y2 = g[:, 2].max(), g[:, 3].max()
        w, h = x2 - x1, y2 - y1
        out.append([float(np.clip(x1 - w * .30, 0, 1)), float(np.clip(y1 - h * .30, 0, 1)),
                    float(np.clip(x2 + w * .30, 0, 1)), float(np.clip(y2 + h * .30, 0, 1))])
    return out


def draw_ui(frame, boxes, drag, meta):
    h, w = frame.shape[:2]
    out = frame.copy()
    for i, b in enumerate(boxes):
        p1 = (int(b[0] * w), int(b[1] * h)); p2 = (int(b[2] * w), int(b[3] * h))
        cv2.rectangle(out, p1, p2, (0, 0, 0), -1)
        cv2.rectangle(out, p1, p2, (0, 200, 255), 2)
        cv2.putText(out, str(i + 1), (p1[0] + 5, p1[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    if drag:
        cv2.rectangle(out, drag[0], drag[1], (0, 255, 120), 2)

    bar = out.copy()
    cv2.rectangle(bar, (0, 0), (w, 58), (18, 18, 18), -1)
    cv2.rectangle(bar, (0, h - 30), (w, h), (18, 18, 18), -1)
    out = cv2.addWeighted(bar, .75, out, .25, 0)
    cv2.putText(out, meta['title'][:70], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (255, 255, 255), 1, cv2.LINE_AA)
    col = (120, 255, 120) if boxes else (110, 160, 255)
    cv2.putText(out, f"{len(boxes)} box(es)   frame {meta['t']+1}/{meta['n']}"
                     f"   {'NO-BOX marked' if meta['nobox'] else ''}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1, cv2.LINE_AA)
    cv2.putText(out, "drag=add  right-click=delete  a=auto  u=undo  c=clear  ,/.=frame  "
                     "p=play  SPACE=next  x=no-box  b=back  q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def review(args):
    here = Path(__file__).parent
    raw = here / args.src
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    store = load_boxes()

    clips = []
    for c in classes:
        for v in sorted(p for p in (raw / c).iterdir() if p.suffix.lower() in VIDEO_EXT):
            clips.append(v)
    if not args.redo:
        clips = [v for v in clips if v.stem not in store]
    if not clips:
        print("every clip already has boxes. --redo to revisit, --render to write output.")
        return

    print(f"{len(clips)} clips to mask.")
    win = 'mask'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {'boxes': [], 'drag': None, 'start': None, 'shape': (1, 1)}

    def on_mouse(ev, x, y, flags, _):
        H, W = state['shape']
        nx, ny = x / W, y / H
        if ev == cv2.EVENT_LBUTTONDOWN:
            state['start'] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and state['start']:
            state['drag'] = (state['start'], (x, y))
        elif ev == cv2.EVENT_LBUTTONUP and state['start']:
            x0, y0 = state['start']
            b = [min(x0, x) / W, min(y0, y) / H, max(x0, x) / W, max(y0, y) / H]
            if (b[2] - b[0]) > .01 and (b[3] - b[1]) > .01:
                state['boxes'].append(b)
            state['start'] = None; state['drag'] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            for i in range(len(state['boxes']) - 1, -1, -1):
                b = state['boxes'][i]
                if b[0] <= nx <= b[2] and b[1] <= ny <= b[3]:
                    state['boxes'].pop(i); break

    cv2.setMouseCallback(win, on_mouse)
    i = 0
    while 0 <= i < len(clips):
        v = clips[i]
        frames, _ = read_clip(v)
        if not frames:
            i += 1; continue
        state['shape'] = frames[0].shape[:2]
        rec = store.get(v.stem)
        if rec is not None and args.redo:
            state['boxes'] = [list(b) for b in rec['boxes']]
        else:
            state['boxes'] = auto_boxes(frames, args.conf)
        nobox = False
        t = 0; playing = False
        key = None

        while True:
            meta = {'title': f"[{i+1}/{len(clips)}] {v.parent.name}/{v.stem}",
                    't': t, 'n': len(frames), 'nobox': nobox}
            cv2.imshow(win, draw_ui(frames[t], state['boxes'], state['drag'], meta))
            k = cv2.waitKey(40 if playing else 20) & 0xFF
            if playing:
                t = (t + 1) % len(frames)
                if t == 0:
                    playing = False
            if k == 255:
                continue
            if k in (ord('q'), 27): key = 'quit'; break
            if k == ord(' '):       key = 'next'; break
            if k == ord('x'):       nobox = True; state['boxes'] = []; key = 'next'; break
            if k == ord('b'):       key = 'back'; break
            if k == ord('a'):       state['boxes'] += auto_boxes(frames, args.conf)
            if k == ord('u') and state['boxes']: state['boxes'].pop()
            if k == ord('c'):       state['boxes'] = []
            if k == ord(','):       t = (t - 1) % len(frames)
            if k == ord('.'):       t = (t + 1) % len(frames)
            if k == ord('p'):       playing = not playing

        if key == 'quit':
            break
        if key == 'back':
            i = max(0, i - 1); continue
        store[v.stem] = {'class': v.parent.name,
                         'boxes': [list(map(float, b)) for b in state['boxes']],
                         'no_face': bool(nobox)}
        save_boxes(store)
        i += 1

    cv2.destroyAllWindows()
    done = len(store)
    empty = sum(1 for r in store.values() if not r['boxes'] and not r['no_face'])
    print(f"\nsaved {done} clips -> {BOXES}")
    if empty:
        print(f"  {empty} have NO box and were not marked 'x' — revisit those before sharing")
    print("  render with:  python mask_faces_manual.py --render --yes")


def render(args):
    here = Path(__file__).parent
    raw, out = here / args.src, here / args.out
    store = load_boxes()
    if not store:
        print("no boxes saved yet — run the tool without --render first")
        return
    todo = [(k, r) for k, r in store.items()]
    print(f"{'RENDERING' if args.yes else 'DRY RUN'} — {len(todo)} clips\n")
    written = risky = 0
    for stem, rec in sorted(todo):
        src = raw / rec['class'] / f"{stem}.mp4"
        if not src.exists():
            continue
        if not rec['boxes'] and not rec['no_face']:
            risky += 1
            print(f"  SKIP (no box, not marked x)  {stem[:52]}")
            continue
        if args.yes:
            frames, fps = read_clip(src)
            if not frames:
                continue
            h, w = frames[0].shape[:2]
            dst = out / rec['class'] / f"{stem}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for f in frames:
                for b in rec['boxes']:
                    cv2.rectangle(f, (int(b[0] * w), int(b[1] * h)),
                                  (int(b[2] * w), int(b[3] * h)), (0, 0, 0), -1)
                vw.write(f)
            vw.release()
            sc = src.with_suffix('.json')
            if sc.exists():
                shutil.copy2(sc, dst.with_suffix('.json'))
        written += 1
    print(f"\n  {written} clips {'written to ' + str(out) if args.yes else 'would be written'}")
    if risky:
        print(f"  {risky} skipped for having no box and no explicit 'no face' mark")
    if not args.yes:
        print("\n  re-run with --render --yes to write them")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='gesture_training_data')
    ap.add_argument('--out', default='gesture_training_data_anon')
    ap.add_argument('--conf', type=float, default=0.10)
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()
    render(args) if args.render else review(args)


if __name__ == '__main__':
    main()
