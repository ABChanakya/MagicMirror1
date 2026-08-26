#!/usr/bin/env python3
"""
merge_incoming.py — safely merge clips received from another machine.

Both machines number clips 01.mp4, 02.mp4, ... per class, so rsyncing straight
onto gesture_training_data/ would overwrite same-named clips that are actually
different recordings. Transfers therefore land in a staging directory and get
merged with this script, which decides per file:

  identical   — same SHA-256 as an existing clip anywhere in that class: skipped
  new name    — name is free: copied as-is
  collision   — name taken by different content: copied under the next free number

Nothing in gesture_training_data/ is ever overwritten or deleted.

Usage
-----
python merge_incoming.py --dry-run     # show what would happen (do this first)
python merge_incoming.py               # perform the merge
"""

import argparse
import hashlib
import shutil
from pathlib import Path

BASE = Path(__file__).parent
DEST = BASE / "gesture_training_data"
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "null"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def next_free_number(class_dir: Path) -> int:
    used = set()
    for p in class_dir.glob("*.mp4"):
        try:
            used.add(int(p.stem))
        except ValueError:
            pass
    n = 1
    while n in used:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="incoming_mac/gesture_training_data",
                    help="staging directory the transfer landed in")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_root = (BASE / args.source).resolve()
    if not src_root.exists():
        print(f"Nothing to merge — {src_root} does not exist.")
        print("Has the transfer finished? Expected layout:")
        print(f"  {src_root}/<gesture>/NN.mp4")
        return 1

    mode = "DRY RUN — nothing will be written" if args.dry_run else "MERGING"
    print(f"{mode}\n  from {src_root}\n  into {DEST}\n")

    totals = {"identical": 0, "copied": 0, "renumbered": 0}

    for gesture in GESTURES:
        src_dir = src_root / gesture
        dst_dir = DEST / gesture
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Hash everything already present so an identical clip under a
        # different name is still recognised as a duplicate.
        existing = {}
        for p in sorted(dst_dir.glob("*.mp4")):
            existing[sha256(p)] = p.name

        incoming = sorted(src_dir.glob("*.mp4"))
        if not incoming:
            continue

        stats = {"identical": 0, "copied": 0, "renumbered": 0}
        notes = []

        for src in incoming:
            digest = sha256(src)

            if digest in existing:
                stats["identical"] += 1
                continue

            target = dst_dir / src.name
            if target.exists():
                n = next_free_number(dst_dir)
                target = dst_dir / f"{n:02d}.mp4"
                stats["renumbered"] += 1
                notes.append(f"      {src.name} -> {target.name} (name taken by different clip)")
            else:
                stats["copied"] += 1

            if not args.dry_run:
                shutil.copy2(src, target)
            # Reserve the name/hash so later files in this batch see it
            existing[digest] = target.name
            if args.dry_run:
                # emulate occupancy so dry-run numbering matches the real run
                target.touch(exist_ok=True)
                if target.stat().st_size == 0:
                    target.unlink()

        print(f"  {gesture:12} {len(incoming):4} incoming -> "
              f"{stats['copied']} new, {stats['renumbered']} renumbered, "
              f"{stats['identical']} already present")
        for line in notes[:5]:
            print(line)
        if len(notes) > 5:
            print(f"      ... and {len(notes) - 5} more renumbered")

        for k in totals:
            totals[k] += stats[k]

    added = totals["copied"] + totals["renumbered"]
    print(f"\n  {added} clips added ({totals['renumbered']} renumbered to avoid "
          f"collisions), {totals['identical']} skipped as duplicates")

    if not args.dry_run and added:
        print("\n  New per-class totals:")
        for gesture in GESTURES:
            d = DEST / gesture
            if d.exists():
                print(f"    {gesture:12} {len(list(d.glob('*.mp4'))):4}")
        print("\n  Next: cd ../gesture_system && python preprocess_landmarks.py && python train_landmark.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
