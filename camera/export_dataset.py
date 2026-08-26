#!/usr/bin/env python3
"""
export_dataset.py — package or sync gesture_training_data/ for the training machine

Two ways to move data off the laptop:

    # 1. Package into a single checksummed archive you can copy anywhere
    python3 export_dataset.py pack
    python3 export_dataset.py pack --since 2026-08-01     # only recent clips

    # 2. Push straight to the university box over ssh (incremental, resumable)
    python3 export_dataset.py sync user@gpu-box:/home/user/MagicMirror1/camera/

And on the receiving end:

    python3 export_dataset.py unpack gestures_2026-08-03.tar.gz --into /path/to/camera
    python3 export_dataset.py check          # verify an archive's manifest

`sync` uses rsync, so an interrupted transfer resumes where it left off and
re-running only sends new clips — which is what you want after each collection
session rather than re-uploading hundreds of megabytes.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "gesture_training_data"
MANIFEST_NAME = "MANIFEST.json"

GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "null"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_files(since=None):
    """All clips (+ sidecars), optionally only those modified since a date."""
    cutoff = None
    if since:
        cutoff = datetime.strptime(since, "%Y-%m-%d").timestamp()

    files = []
    for gesture in GESTURES:
        d = DATA_DIR / gesture
        if not d.exists():
            continue
        for path in sorted(d.iterdir()):
            if path.suffix.lower() not in (".mp4", ".json"):
                continue
            if cutoff and path.stat().st_mtime < cutoff:
                continue
            files.append(path)
    return files


def build_manifest(files):
    entries = []
    counts = {g: 0 for g in GESTURES}
    for path in files:
        rel = path.relative_to(DATA_DIR)
        entries.append({
            "path": str(rel),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        if path.suffix.lower() == ".mp4" and rel.parts[0] in counts:
            counts[rel.parts[0]] += 1
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "clip_counts": counts,
        "total_clips": sum(counts.values()),
        "files": entries,
    }


def cmd_pack(args):
    files = collect_files(args.since)
    clips = [f for f in files if f.suffix.lower() == ".mp4"]
    if not clips:
        print("Nothing to pack — no clips found"
              + (f" modified since {args.since}." if args.since else "."))
        return 1

    print(f"Hashing {len(files)} files...")
    manifest = build_manifest(files)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = Path(args.out) if args.out else BASE_DIR / f"gestures_{stamp}.tar.gz"
    manifest_path = DATA_DIR / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Writing {out} ...")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(manifest_path, arcname=f"gesture_training_data/{MANIFEST_NAME}")
        for path in files:
            tar.add(path, arcname=f"gesture_training_data/{path.relative_to(DATA_DIR)}")

    size_mb = out.stat().st_size / 1e6
    digest = sha256(out)
    (out.parent / (out.name + ".sha256")).write_text(f"{digest}  {out.name}\n")

    print(f"\n✅ {out.name}  ({size_mb:.1f} MB, {manifest['total_clips']} clips)")
    for g, n in manifest["clip_counts"].items():
        print(f"   {g:12} {n}")
    print(f"\n   sha256: {digest}")
    print("\nCopy it over with:")
    print(f"   scp {out} user@gpu-box:~/")
    print("Then on that machine:")
    print(f"   python3 export_dataset.py unpack {out.name} "
          f"--into /path/to/MagicMirror1/camera")
    return 0


def cmd_sync(args):
    if shutil.which("rsync") is None:
        print("rsync not found on this machine.")
        return 1
    if not DATA_DIR.exists():
        print(f"No data to sync — {DATA_DIR} does not exist.")
        return 1

    dest = args.destination.rstrip("/") + "/"
    cmd = [
        "rsync", "-avz", "--partial", "--progress",
        "--include=*/", "--include=*.mp4", "--include=*.json", "--exclude=*",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd += [str(DATA_DIR) + "/", dest + "gesture_training_data/"]

    print("Running: " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("\n✅ Sync complete. On the training machine:")
        print("   cd gesture_system && python3 preprocess_landmarks.py")
    return result.returncode


def cmd_unpack(args):
    archive = Path(args.archive)
    if not archive.exists():
        print(f"No such archive: {archive}")
        return 1
    into = Path(args.into).resolve()
    into.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive.name} into {into} ...")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            # Refuse paths that would escape the destination.
            target = (into / member.name).resolve()
            if not str(target).startswith(str(into)):
                print(f"  ⚠️  skipping unsafe path {member.name}")
                continue
            tar.extract(member, path=into)

    manifest_path = into / "gesture_training_data" / MANIFEST_NAME
    if manifest_path.exists():
        return verify_manifest(manifest_path)
    print("✅ Extracted (archive had no manifest to verify).")
    return 0


def verify_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    missing, corrupt = [], []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.exists():
            missing.append(entry["path"])
        elif sha256(path) != entry["sha256"]:
            corrupt.append(entry["path"])

    print(f"\nManifest: {manifest['total_clips']} clips from {manifest['host']} "
          f"({manifest['created_at']})")
    for g, n in manifest["clip_counts"].items():
        print(f"   {g:12} {n}")
    if missing or corrupt:
        print(f"\n❌ {len(missing)} missing, {len(corrupt)} corrupt")
        for p in (missing + corrupt)[:20]:
            print(f"   {p}")
        return 1
    print("\n✅ All files present and checksums match.")
    return 0


def cmd_check(args):
    manifest_path = Path(args.manifest) if args.manifest else DATA_DIR / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}. Run `pack` first.")
        return 1
    return verify_manifest(manifest_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pack", help="build a checksummed .tar.gz")
    p.add_argument("--out", help="archive path (default: camera/gestures_<date>.tar.gz)")
    p.add_argument("--since", help="only clips modified on/after YYYY-MM-DD")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("sync", help="rsync the dataset to a remote machine")
    p.add_argument("destination",
                   help="e.g. user@gpu-box:/home/user/MagicMirror1/camera/")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("unpack", help="extract an archive and verify checksums")
    p.add_argument("archive")
    p.add_argument("--into", default=".", help="the camera/ directory to extract into")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("check", help="verify a manifest against files on disk")
    p.add_argument("--manifest")
    p.set_defaults(func=cmd_check)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
