#!/usr/bin/env python3
"""
anonymize_faces.py — black out faces in recorded gesture clips.

Uses MediaPipe Face Detection (BlazeFace, model_selection=1 for the longer-range
variant). It ships with mediapipe, so nothing extra to install.

Why the default is a union box rather than per-frame boxes:

    Detection succeeds on ~72% of frames in this dataset. Blacking only the
    frames where a face was found leaves the other 28% untouched, and a single
    visible frame defeats the whole exercise — anonymisation has to be judged on
    its worst frame, not its average. So every detection across the clip is
    merged into one padded region that is blacked in every frame. The person
    stands roughly still in these recordings, so that region stays tight.

    --mode track is available for footage where the subject moves a lot, and
    fills detection gaps by interpolating between neighbouring frames instead.

Originals are never modified. Output goes to a separate directory.

IMPORTANT — do not train on the anonymised copies without checking first.
MediaPipe Pose partly locates a person via their face, so blacking it can hurt
pose detection, which is what the gesture pipeline depends on. --verify measures
that on a sample and prints the before/after rates.

Usage
-----
python anonymize_faces.py --verify                 # measure the cost first
python anonymize_faces.py --limit 5                # small trial
python anonymize_faces.py                          # all clips
python anonymize_faces.py --mode track --pad 0.35
"""

import argparse
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv'}


def detect_boxes(frames, conf=0.10):
    """Per-frame face boxes in normalised xyxy, or None where none was found."""
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                   min_detection_confidence=conf)
    out = []
    for f in frames:
        res = fd.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if not res.detections:
            out.append(None)
            continue
        boxes = []
        for d in res.detections:
            b = d.location_data.relative_bounding_box
            boxes.append([b.xmin, b.ymin, b.xmin + b.width, b.ymin + b.height])
        out.append(np.array(boxes, dtype=np.float32))
    fd.close()
    return out


def union_box(per_frame, pad):
    """One padded box covering every detection in the clip."""
    found = [b for b in per_frame if b is not None]
    if not found:
        return None
    allb = np.concatenate(found, axis=0)
    x1, y1 = allb[:, 0].min(), allb[:, 1].min()
    x2, y2 = allb[:, 2].max(), allb[:, 3].max()
    w, h = x2 - x1, y2 - y1
    return np.clip([x1 - w * pad, y1 - h * pad, x2 + w * pad, y2 + h * pad], 0, 1)


def filled_track(per_frame, pad):
    """Per-frame boxes with detection gaps filled from the nearest neighbours."""
    n = len(per_frame)
    single = [None if b is None else
              [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]
              for b in per_frame]
    idx = [i for i, b in enumerate(single) if b is not None]
    if not idx:
        return [None] * n
    out = []
    for i in range(n):
        if single[i] is not None:
            box = single[i]
        else:
            prev = max([j for j in idx if j < i], default=None)
            nxt = min([j for j in idx if j > i], default=None)
            if prev is None:
                box = single[nxt]
            elif nxt is None:
                box = single[prev]
            else:  # widen across the gap rather than guessing a position
                a, b = single[prev], single[nxt]
                box = [min(a[0], b[0]), min(a[1], b[1]),
                       max(a[2], b[2]), max(a[3], b[3])]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        out.append(np.clip([x1 - w * pad, y1 - h * pad,
                            x2 + w * pad, y2 + h * pad], 0, 1))
    return out


def apply_box(frame, box):
    if box is None:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (int(x1 * w), int(y1 * h)),
                  (int(x2 * w), int(y2 * h)), (0, 0, 0), -1)
    return frame


def process(src, dst, mode, pad, conf):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None

    per = detect_boxes(frames, conf)
    n_detected = sum(1 for b in per if b is not None)

    if mode == 'union':
        box = union_box(per, pad)
        boxes = [box] * len(frames)
    else:
        boxes = filled_track(per, pad)

    h, w = frames[0].shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f, b in zip(frames, boxes):
        vw.write(apply_box(f, b))
    vw.release()
    return n_detected, len(frames), boxes[0] is not None


def verify(raw, classes, n_per_class, pad, conf, mode):
    """Does blacking the face cost us pose detection? Measure, do not assume."""
    from tempfile import TemporaryDirectory
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'gesture_system'))
    from dataset import _extract_frames
    from preprocess_holistic import L_SHOULDER, _init_holistic, extract_frame

    hol = _init_holistic()
    before = after = total = 0
    with TemporaryDirectory() as td:
        for cls in classes:
            for v in sorted((raw / cls).glob('*.mp4'))[:n_per_class]:
                fr = _extract_frames(str(v), num_frames=30)
                if fr is None:
                    continue
                seq = np.stack([extract_frame(hol, f) for f in fr])
                before += int(np.isfinite(seq[:, L_SHOULDER, 0]).sum())

                out = Path(td) / f"{v.stem}.mp4"
                process(v, out, mode, pad, conf)
                fr2 = _extract_frames(str(out), num_frames=30)
                seq2 = np.stack([extract_frame(hol, f) for f in fr2])
                after += int(np.isfinite(seq2[:, L_SHOULDER, 0]).sum())
                total += 30

    print(f"\n  pose detected BEFORE blacking: {before}/{total} ({before/total*100:.1f}%)")
    print(f"  pose detected AFTER  blacking: {after}/{total} ({after/total*100:.1f}%)")
    drop = (before - after) / max(total, 1) * 100
    print(f"  cost: {drop:+.1f} percentage points")
    if drop > 5:
        print("\n  -> Blacking the face measurably hurts pose tracking. Train on the")
        print("     ORIGINALS and use the anonymised copies only for sharing.")
    else:
        print("\n  -> Negligible. The anonymised copies are safe to train on too.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='gesture_training_data')
    ap.add_argument('--out', default='gesture_training_data_anon')
    ap.add_argument('--mode', default='union', choices=['union', 'track'])
    ap.add_argument('--pad', type=float, default=0.30,
                    help='fraction of box size added on each side')
    ap.add_argument('--conf', type=float, default=0.10,
                    help='0.10 not 0.3: at 0.3 the detector found faces in 72%% of '
                         'frames and never more than one person; at 0.10 it finds '
                         '90%% and picks up second faces it was previously missing')
    ap.add_argument('--limit', type=int, default=0, help='clips per class')
    ap.add_argument('--verify', action='store_true',
                    help='measure the pose-detection cost, write nothing')
    args = ap.parse_args()

    here = Path(__file__).parent
    raw = here / args.src
    out = here / args.out
    if not raw.exists():
        print(f"no such directory: {raw}")
        return
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())

    if args.verify:
        print(f"Measuring pose-detection cost of face blacking "
              f"(mode={args.mode}, pad={args.pad})")
        verify(raw, classes, args.limit or 4, args.pad, args.conf, args.mode)
        return

    print(f"{args.src} -> {args.out}   mode={args.mode} pad={args.pad}\n")
    tot = noface = 0
    for cls in classes:
        vids = sorted(p for p in (raw / cls).iterdir() if p.suffix.lower() in VIDEO_EXT)
        if args.limit:
            vids = vids[:args.limit]
        det_sum = frm_sum = 0
        for v in vids:
            r = process(v, out / cls / v.name, args.mode, args.pad, args.conf)
            if r is None:
                continue
            d, f, had = r
            det_sum += d
            frm_sum += f
            tot += 1
            if not had:
                noface += 1
            sc = v.with_suffix('.json')          # keep the sidecar alongside
            if sc.exists():
                shutil.copy2(sc, out / cls / sc.name)
        if frm_sum:
            print(f"  {cls:12} {len(vids):4} clips   face found in "
                  f"{det_sum/frm_sum*100:5.1f}% of frames")

    print(f"\n  {tot} clips written to {out}")
    if noface:
        print(f"  {noface} clips had NO face detected in any frame — nothing was")
        print("  blacked in those. Check them by hand before sharing:")
        print(f"    python anonymize_faces.py --verify   # or review them visually")
    print("\n  Originals are untouched. Verify a few before relying on this:")
    print(f"    mpv {out}/*/*.mp4")


if __name__ == '__main__':
    main()
