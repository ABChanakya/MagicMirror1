#!/usr/bin/env python3
"""
collect_gesture_data.py — record gesture clips for the landmark training pipeline

Cross-platform (macOS / Linux). Writes clips to gesture_training_data/<class>/,
which is exactly what gesture_system/preprocess_landmarks.py reads.

Usage
-----
    python3 collect_gesture_data.py cameras              # pick a camera, by eye
    python3 collect_gesture_data.py                      # interactive menu
    python3 collect_gesture_data.py --subject chanakya   # tag clips with a subject
    python3 collect_gesture_data.py batch swipe_left 25  # record 25 clips
    python3 collect_gesture_data.py batch null 25 --auto # ...without the ENTER prompt
    python3 collect_gesture_data.py status               # per-class counts
    python3 collect_gesture_data.py verify               # re-check every clip on disk

Run `cameras` first on any machine with more than one camera. It shows each
feed and saves the index you confirm to .camera_choice.json. Device names are
deliberately not used: OpenCV's macOS backend enumerates cameras in a different
order than AVFoundation lists them, so a name looked up by index can label the
wrong feed — and the indices shuffle when a USB camera reconnects. Re-run
`cameras` after replugging the webcam.

Each take waits for you to press ENTER in the video window before its lead-in
countdown starts, so you record when you are set rather than when a timer says
so. --auto restores the old hands-free behaviour for bulk null-class capture.

Design notes
------------
* Recorded frames are CLEAN. Overlays (skeleton, countdown, borders) are drawn
  on a display copy only — baking them into the file would corrupt the
  landmarks that preprocess_landmarks.py extracts later.
* Frames are recorded UNMIRRORED, matching camera/main.py (MIRROR_FLIP=false)
  and gesture_system/inference.py. Mirroring the recording would invert
  swipe_left / swipe_right relative to inference. --mirror flips display and
  recording together if you ever need it.
* Every clip gets a sidecar .json with subject, session, host, resolution,
  measured fps and the fraction of frames MediaPipe found a hand in. Filenames
  carry subject + timestamp, so datasets from several machines merge without
  collisions.
* Each clip is validated after writing (re-opened, frames counted, hand
  coverage checked). Bad takes are deleted and re-recorded rather than silently
  poisoning the dataset.
"""

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "gesture_training_data"

# Which camera index the operator confirmed by eye (see pick_camera).
CAMERA_CHOICE_FILE = BASE_DIR / ".camera_choice.json"

_ENTER_KEYS = (13, 10)      # CR on macOS, LF on Linux
_SPACE_KEY = 32

# Must match gesture_system/config.yaml -> data.classes
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "null"]

# Per-class collection targets (the "null" class needs fewer).
TARGETS = {
    "swipe_left": 150,
    "swipe_right": 150,
    "swipe_up": 150,
    "swipe_down": 150,
    "null": 100,
}

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

RECORD_SECONDS = 3        # wall-clock length of one clip
MAX_RECORD_SECONDS = 2.5    # hard stop, so a slow camera cannot record forever
MIN_FRAMES = 24             # below this the clip is rejected (config wants 30 samples)
MIN_HAND_RATIO = 0.60       # fraction of frames MediaPipe must find a hand in
COUNTDOWN_SECONDS = 0      # lead-in before each clip in batch mode

# The "null" class is background/stillness — a hand often isn't there at all,
# so the hand-coverage gate does not apply to it.
NO_HAND_GATE = {"null"}

# Consecutive failed reads before we call the camera gone. read_frame() already
# retries internally for ~50 ms, so this is about a second of silence.
DEAD_READ_LIMIT = 20

_VIDEO_EXT = ".mp4"


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe
# ──────────────────────────────────────────────────────────────────────────────

_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils
_hands = None


def get_hands():
    """Lazily build the MediaPipe Hands solution (it is slow to construct)."""
    global _hands
    if _hands is None:
        _hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _hands


# ──────────────────────────────────────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────────────────────────────────────

class CameraError(RuntimeError):
    pass


def _backends():
    """Capture backends to try, best first, for this platform."""
    if platform.system() == "Darwin":
        return [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (cv2.CAP_ANY, "default")]
    if platform.system() == "Linux":
        return [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "default")]
    return [(cv2.CAP_ANY, "default")]


def probe_indices(limit=6):
    """
    Indices that open and deliver a frame.

    Stops at the first gap — indices are contiguous, and probing past the end
    makes OpenCV print a wall of "out device of bound" errors we cannot silence.
    """
    found = []
    backend = _backends()[0][0]
    for i in range(limit):
        cap = cv2.VideoCapture(i, backend)
        ok = False
        if cap.isOpened():
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
        cap.release()
        if not ok:
            break
        found.append(i)
    return found


def pick_camera(mirror=False):
    """
    Show each camera in turn and let the operator confirm one by eye.

    Device *names* are not usable for this on macOS: OpenCV's AVFoundation
    backend enumerates in a different order than AVCaptureDevice reports, so a
    name looked up by index can label the wrong feed — and the indices
    themselves get reshuffled when a USB camera reconnects. Looking at the
    picture is the only check that cannot be wrong, so we ask once and remember
    the answer.

    Returns the chosen index, or None if cancelled.
    """
    indices = probe_indices()
    if not indices:
        raise CameraError("No working cameras found.\n" + macos_permission_hint())

    print(f"\nFound {len(indices)} camera(s). A preview window will open for each.")
    print("  ENTER = use this one    SPACE = next    q = cancel\n")

    window = "Pick a camera"
    for pos, idx in enumerate(indices, 1):
        try:
            cap, info = open_camera(idx)
        except CameraError:
            continue
        try:
            while True:
                frame = read_frame(cap)
                if frame is None:
                    break
                display = cv2.flip(frame, 1) if mirror else frame.copy()
                draw_hud(display, [
                    (f"CAMERA INDEX {idx}   ({pos}/{len(indices)})", 0.9, (255, 150, 0)),
                    (f"{info['width']}x{info['height']} @ "
                     f"{info['fps_measured']:.0f} fps", 0.7, (0, 255, 0)),
                    ("ENTER = use this   SPACE = next   q = cancel",
                     0.6, (200, 200, 200)),
                ], border=(255, 255, 255))
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in _ENTER_KEYS:
                    save_camera_choice(idx)
                    print(f"Saved: camera index {idx} "
                          f"({info['width']}x{info['height']}). "
                          f"Re-run `cameras` if you replug the webcam.\n")
                    return idx
                if key == _SPACE_KEY:
                    break
                if key == ord('q'):
                    print("Cancelled — nothing saved.\n")
                    return None
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print("Ran out of cameras without a choice. Run `cameras` again.\n")
    return None


def save_camera_choice(index):
    try:
        CAMERA_CHOICE_FILE.write_text(json.dumps({"index": index}) + "\n")
    except OSError as e:
        print(f"(could not save camera choice: {e})")


def load_camera_choice():
    try:
        return int(json.loads(CAMERA_CHOICE_FILE.read_text())["index"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def resolve_camera(spec):
    """
    Turn a --camera value into an index.

    An explicit number wins. Otherwise fall back to the index saved by
    `cameras`, and finally to 0 — with a nudge, because on a two-camera laptop
    index 0 is a guess and guessing is what sends clips to the wrong webcam.
    """
    if spec is not None and str(spec).strip() != "":
        spec = str(spec).strip()
        if spec.lstrip("+-").isdigit():
            return int(spec)
        raise CameraError(
            f"--camera takes an index, not a name (got '{spec}').\n"
            "Camera names cannot be mapped to OpenCV indices reliably on this "
            "platform.\nRun `cameras` to pick one by eye instead.")

    saved = load_camera_choice()
    if saved is not None:
        return saved

    print("\nNo camera chosen yet — using index 0. If that is the wrong camera, "
          "run:\n  ./collect.sh cameras\n")
    return 0


def open_camera(index=0, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=TARGET_FPS):
    """
    Open a camera and return (cap, info).

    Tries each platform backend in turn. Reports the resolution/fps the driver
    actually granted rather than the ones requested — VideoWriter must be built
    from the real frame size or it writes an unreadable file.
    """
    last_err = None
    for backend, name in _backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            last_err = f"{name} could not open camera index {index}"
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # A small buffer keeps the preview close to real time.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass

        # Warm up: the first frames off a Mac camera are often black, and the
        # driver may need a moment to settle on the requested format.
        ok = False
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok or frame is None:
            cap.release()
            last_err = f"{name} opened camera {index} but returned no frames"
            continue

        h, w = frame.shape[:2]
        info = {
            "backend": name,
            "index": index,
            "width": w,
            "height": h,
            "fps_reported": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "fps_measured": measure_fps(cap),
        }
        return cap, info

    raise CameraError(
        f"Could not open camera index {index}. Last error: {last_err}\n"
        + macos_permission_hint()
    )


def measure_fps(cap, frames=20):
    """Time a short burst of reads to get the true capture rate."""
    t0 = time.monotonic()
    got = 0
    for _ in range(frames):
        ok, _f = cap.read()
        if ok:
            got += 1
    dt = time.monotonic() - t0
    if got < 2 or dt <= 0:
        return float(TARGET_FPS)
    return round(got / dt, 2)


def macos_permission_hint():
    if platform.system() != "Darwin":
        return ("Check the device exists (ls /dev/video*) and that no other "
                "process is holding it.")
    return (
        "On macOS the app running Python needs camera access:\n"
        "  System Settings -> Privacy & Security -> Camera\n"
        "  -> enable the app you launched this from (Terminal, iTerm, VS Code).\n"
        "You must fully quit and reopen that app after granting access.\n"
        "Also close Photo Booth / FaceTime / Zoom — they can hold the camera."
    )


def camera_lost_hint():
    """Why a camera that was working stops mid-session."""
    return (
        "A USB webcam that drops off the bus does not come back on its own.\n"
        "  1. Check the cable / try another port.\n"
        "  2. Re-run `./collect.sh cameras` — indices shuffle on reconnect.\n"
        "Clips already recorded in this session are safe on disk."
    )


def read_frame(cap, retries=5):
    """Read one frame, tolerating the occasional dropped read."""
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.01)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Recording
# ──────────────────────────────────────────────────────────────────────────────

def session_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def clip_path(gesture, subject, session, seq):
    name = f"{gesture}_{subject}_{session}_{seq:03d}{_VIDEO_EXT}"
    return DATA_DIR / gesture / name


def existing_count(gesture):
    return len(list((DATA_DIR / gesture).glob(f"*{_VIDEO_EXT}")))


def draw_hud(display, lines, colour=(0, 255, 0), border=None):
    """Draw text lines and an optional border on the DISPLAY copy only."""
    y = 30
    for text, scale, col in lines:
        cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, col, 2, cv2.LINE_AA)
        y += int(34 * max(scale, 0.7))
    if border is not None:
        h, w = display.shape[:2]
        cv2.rectangle(display, (4, 4), (w - 5, h - 5), border, 4)
    return display


def record_clip(cap, info, gesture, out_path, mirror=False, window="Collect"):
    """
    Record one clip.

    Returns a metadata dict on success, None if the operator aborted with 'q'.
    Writes clean frames; overlays go on a display copy. Hand presence is
    measured live so a bad take can be caught immediately.
    """
    hands = get_hands()
    fps = info["fps_measured"] or TARGET_FPS
    size = (info["width"], info["height"])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        raise CameraError(f"Could not open VideoWriter for {out_path}")

    frames_written = 0
    hand_frames = 0
    aborted = False
    dead_reads = 0
    lost = False
    start = time.monotonic()

    try:
        while True:
            elapsed = time.monotonic() - start
            # Stop on time, but never before MIN_FRAMES are on disk — a camera
            # that drops to 15 fps would otherwise produce clips that are
            # rejected for length through no fault of the operator.
            if elapsed >= RECORD_SECONDS and frames_written >= MIN_FRAMES:
                break
            if elapsed >= MAX_RECORD_SECONDS:
                break

            frame = read_frame(cap)
            if frame is None:
                # A camera that has gone away never recovers inside a take. Say
                # so now: writing zero frames leaves a header-only file that
                # passes the size>0 check and then fails to re-open, which
                # reads as "bad take" and gets retried forever.
                dead_reads += 1
                if dead_reads >= DEAD_READ_LIMIT:
                    raise CameraError(
                        "Camera stopped delivering frames mid-take.\n"
                        + camera_lost_hint())
                continue
            dead_reads = 0
            if mirror:
                frame = cv2.flip(frame, 1)

            if (frame.shape[1], frame.shape[0]) != size:
                raise CameraError(
                    f"Camera changed resolution mid-take: expected "
                    f"{size[0]}x{size[1]}, got {frame.shape[1]}x{frame.shape[0]}.\n"
                    "VideoWriter silently drops frames that do not match, so "
                    "the clip would be empty. Restart the collector.")

            # Clean frame to disk, before anything is drawn on it.
            writer.write(frame)
            frames_written += 1

            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                hand_frames += 1

            display = frame.copy()
            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks:
                    _mp_draw.draw_landmarks(display, lm, _mp_hands.HAND_CONNECTIONS)
            draw_hud(display, [
                (f"REC {gesture}", 0.9, (0, 0, 255)),
                (f"{RECORD_SECONDS - elapsed:.1f}s  frames {frames_written}", 0.7,
                 (0, 0, 255)),
            ], border=(0, 0, 255))
            cv2.imshow(window, display)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                aborted = True
                break
    except CameraError:
        lost = True
        raise
    finally:
        writer.release()
        if lost:
            # Release first, then remove: a header-only stub is exactly what
            # made the old code retry the same dead camera 300 times.
            out_path.unlink(missing_ok=True)

    if aborted:
        out_path.unlink(missing_ok=True)
        return None

    if frames_written == 0:
        out_path.unlink(missing_ok=True)
        raise CameraError("Camera returned no frames for this take.\n"
                          + camera_lost_hint())

    hand_ratio = hand_frames / frames_written if frames_written else 0.0
    return {
        "gesture": gesture,
        "file": out_path.name,
        "frames_written": frames_written,
        "hand_ratio": round(hand_ratio, 3),
        "fps_write": fps,
        "width": size[0],
        "height": size[1],
        "mirrored": mirror,
        "duration_s": round(time.monotonic() - start, 2),
    }


def validate_clip(path, gesture, meta):
    """
    Re-open a written clip and decide whether to keep it.

    Returns (ok: bool, reason: str). Catches the failure modes that otherwise
    show up only at training time: unreadable files, short clips, and takes
    where the hand was never in frame.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False, "file missing or empty"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, "written file cannot be re-opened"
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    if not ok or first is None:
        return False, "written file has no readable frames"
    if count < MIN_FRAMES:
        return False, f"only {count} frames (need {MIN_FRAMES})"
    if gesture not in NO_HAND_GATE and meta["hand_ratio"] < MIN_HAND_RATIO:
        return False, (f"hand seen in only {meta['hand_ratio']*100:.0f}% of frames "
                       f"(need {MIN_HAND_RATIO*100:.0f}%)")
    return True, "ok"


def write_sidecar(path, meta, subject, session, info):
    meta = dict(meta)
    meta.update({
        "subject": subject,
        "session": session,
        "host": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()}",
        "camera_backend": info["backend"],
        "camera_index": info["index"],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": 1,
    })
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def arm(cap, info, gesture, seq, total, mirror, window):
    """
    Hold on a live preview until the operator presses ENTER (or SPACE).

    Returns True to go ahead with the take, False if they quit. This is the
    difference between recording when you are ready and recording whatever the
    countdown happened to catch — with the camera live the whole time, so the
    take starts instantly rather than waiting on a driver warm-up.
    """
    hands = get_hands()
    while True:
        frame = read_frame(cap)
        if frame is None:
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        display = frame.copy()
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                _mp_draw.draw_landmarks(display, lm, _mp_hands.HAND_CONNECTIONS)
            hint, hint_col = "hand detected", (0, 255, 0)
        else:
            hint, hint_col = "NO HAND IN FRAME", (0, 165, 255)
        draw_hud(display, [
            (f"{gesture.upper()}  [{seq}/{total}]", 0.9, (255, 150, 0)),
            ("READY - press ENTER to record", 0.8, (255, 255, 255)),
            (hint, 0.7, hint_col),
            ("ENTER = go   q = quit", 0.6, (200, 200, 200)),
        ], border=(255, 255, 255))
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key in _ENTER_KEYS or key == _SPACE_KEY:
            return True


def countdown(cap, info, gesture, seq, total, seconds, mirror, window):
    """Live preview countdown before a take. Returns False if aborted."""
    start = time.monotonic()
    hands = get_hands()
    while True:
        left = seconds - (time.monotonic() - start)
        if left <= 0:
            return True
        frame = read_frame(cap)
        if frame is None:
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        display = frame.copy()
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                _mp_draw.draw_landmarks(display, lm, _mp_hands.HAND_CONNECTIONS)
            hint = "hand detected"
            hint_col = (0, 255, 0)
        else:
            hint = "NO HAND IN FRAME"
            hint_col = (0, 165, 255)
        draw_hud(display, [
            (f"{gesture.upper()}  [{seq}/{total}]", 0.9, (255, 150, 0)),
            (f"starting in {left:.1f}s", 0.8, (0, 255, 0)),
            (hint, 0.7, hint_col),
            ("q = quit   s = skip wait", 0.6, (200, 200, 200)),
        ])
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key == ord('s'):
            return True


def record_batch(gesture, count, subject, camera_index=0, mirror=False,
                 wait_for_key=True):
    """Record `count` clips of one gesture. Every take is kept."""
    if gesture not in GESTURES:
        print(f"Unknown gesture '{gesture}'. Choose from: {', '.join(GESTURES)}")
        return 0

    (DATA_DIR / gesture).mkdir(parents=True, exist_ok=True)
    session = session_id()
    window = "Gesture collection"

    try:
        cap, info = open_camera(camera_index)
    except CameraError as e:
        print(f"\n❌ {e}\n")
        return 0
    except KeyboardInterrupt:
        # Ctrl-C while the camera is warming up: back out quietly, not with a
        # traceback out of the menu.
        print("\nCancelled.")
        return 0

    print(f"\nCamera: {info['backend']} index {info['index']}  "
          f"{info['width']}x{info['height']} @ {info['fps_measured']} fps "
          f"(driver reports {info['fps_reported']})")
    print(f"Recording {count} × {gesture} — {RECORD_SECONDS}s each, "
          f"{COUNTDOWN_SECONDS}s lead-in.")
    print("Directions are from the CAMERA's point of view "
          f"(mirror {'ON' if mirror else 'OFF'}).")
    if wait_for_key:
        print("Press ENTER in the video window to start each take, q to stop.\n")
    else:
        print("Hands-free: takes roll automatically. Press q in the video "
              "window to stop.\n")

    saved = 0
    seq = existing_count(gesture)
    try:
        for i in range(1, count + 1):
            if wait_for_key and not arm(cap, info, gesture, i, count, mirror, window):
                print("\nStopped by operator.")
                break
            if not countdown(cap, info, gesture, i, count,
                             COUNTDOWN_SECONDS, mirror, window):
                print("\nStopped by operator.")
                break

            seq += 1
            path = clip_path(gesture, subject, session, seq)
            meta = record_clip(cap, info, gesture, path, mirror, window)
            if meta is None:
                print("\nStopped by operator.")
                return saved

            # Every take is kept. Quality is not judged here — run `verify`
            # over the dataset later if you want the clips re-checked.
            write_sidecar(path, meta, subject, session, info)
            saved += 1
            print(f"  ✅ [{i}/{count}] {path.name}  "
                  f"{meta['frames_written']} frames, "
                  f"hand {meta['hand_ratio']*100:.0f}%")
    except CameraError as e:
        print(f"\n\u274c {e}\n")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSaved {saved}/{count} clips of {gesture}.")
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Status / verification
# ──────────────────────────────────────────────────────────────────────────────

def show_status():
    print("\n" + "=" * 64)
    print("DATA COLLECTION STATUS")
    print("=" * 64)
    total = target_total = 0
    for gesture in GESTURES:
        d = DATA_DIR / gesture
        count = len(list(d.glob(f"*{_VIDEO_EXT}"))) if d.exists() else 0
        target = TARGETS.get(gesture, 100)
        pct = 100 * count / target if target else 0
        mark = "✅" if count >= target else "⚠️ " if count >= target / 2 else "❌"
        print(f" {mark} {gesture:12} {count:4}/{target:<4} ({pct:3.0f}%)")
        total += count
        target_total += target
    print("-" * 64)
    print(f"    {'TOTAL':12} {total:4}/{target_total:<4} "
          f"({100*total/target_total if target_total else 0:3.0f}%)")

    subjects = {}
    for j in DATA_DIR.rglob("*.json"):
        try:
            subject = json.loads(j.read_text()).get("subject", "?")
        except (json.JSONDecodeError, OSError):
            continue
        subjects[subject] = subjects.get(subject, 0) + 1
    if subjects:
        print("\n Clips per subject: "
              + ", ".join(f"{k}={v}" for k, v in sorted(subjects.items())))
    print()
    return total


def verify_all(delete_bad=False):
    """Re-open every clip on disk and report the ones training would choke on."""
    print("\nVerifying every clip on disk (this reads each file)...\n")
    hands = get_hands()
    bad = []
    checked = 0

    for gesture in GESTURES:
        d = DATA_DIR / gesture
        if not d.exists():
            continue
        for path in sorted(d.glob(f"*{_VIDEO_EXT}")):
            checked += 1
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                bad.append((path, "unreadable")); cap.release(); continue
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < MIN_FRAMES:
                bad.append((path, f"{count} frames")); cap.release(); continue

            # Sample 10 frames across the clip and check hand coverage.
            hits = 0
            idxs = np.linspace(0, count - 1, 10, dtype=int)
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
                    hits += 1
            cap.release()
            ratio = hits / len(idxs)
            if gesture not in NO_HAND_GATE and ratio < MIN_HAND_RATIO:
                bad.append((path, f"hand in {ratio*100:.0f}% of sampled frames"))

    print(f"Checked {checked} clips — {len(bad)} problem(s).")
    for path, reason in bad:
        print(f"  ❌ {path.relative_to(DATA_DIR)}: {reason}")
        if delete_bad:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            print("     deleted")
    if bad and not delete_bad:
        print("\nRe-run with --delete-bad to remove them.")
    print()
    return bad


# ──────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ──────────────────────────────────────────────────────────────────────────────

def _raise_range():
    raise ValueError("out of range")


def ask(prompt, cast=str, default=None):
    """
    Prompt for one value.

    Drains anything already sitting in stdin first: pasting a block of commands
    leaves later lines queued, and they would otherwise be swallowed as menu
    answers. Returns None if the operator hits Ctrl-C or Ctrl-D.
    """
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (ImportError, termios.error, OSError):
        pass

    while True:
        try:
            sys.stdout.flush()
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            continue
        try:
            return cast(raw)
        except (ValueError, IndexError):
            print(f"  '{raw}' is not valid — try again (Ctrl-C to go back).")


def interactive(subject, camera_index, mirror, wait_for_key=True):
    while True:
        print("\n" + "=" * 64)
        print("GESTURE DATA COLLECTION")
        print("=" * 64)
        print(f"  subject: {subject}   mirror: {'on' if mirror else 'off'}   "
              f"start: {'ENTER key' if wait_for_key else 'automatic'}")
        print(f"  camera:  index {camera_index}")
        print("\n  1) Record a batch of one gesture")
        print("  2) Record a round of every gesture")
        print("  3) Status")
        print("  4) Verify clips on disk")
        print("  5) Change camera")
        print("  6) Quit")

        choice = ask("\n> ")
        if choice is None:
            show_status()
            return

        if choice == "1":
            for i, g in enumerate(GESTURES, 1):
                print(f"  {i}) {g}  ({existing_count(g)} so far)")
            idx = ask("gesture > ", lambda s: GESTURES[int(s) - 1]
                      if 1 <= int(s) <= len(GESTURES) else _raise_range())
            if idx is None:
                continue
            n = ask("how many clips > ", int)
            if n is None:
                continue
            record_batch(idx, n, subject, camera_index, mirror,
                         wait_for_key=wait_for_key)

        elif choice == "2":
            n = ask("clips per gesture > ", int)
            if n is None:
                continue
            for g in GESTURES:
                per = max(1, n // 2) if g == "null" else n
                if record_batch(g, per, subject, camera_index, mirror,
                                wait_for_key=wait_for_key) == 0:
                    break

        elif choice == "3":
            show_status()

        elif choice == "4":
            verify_all()

        elif choice == "5":
            chosen = pick_camera(mirror)
            if chosen is not None:
                camera_index = chosen

        elif choice in ("6", "q", "quit", "exit"):
            show_status()
            return

        else:
            print("Invalid choice.")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="menu",
                    choices=["menu", "batch", "status", "verify", "cameras"])
    ap.add_argument("gesture", nargs="?", help="for `batch`: which gesture")
    ap.add_argument("count", nargs="?", type=int, help="for `batch`: how many clips")
    ap.add_argument("--subject", default=os.getenv("GESTURE_SUBJECT", "unknown"),
                    help="who is performing the gesture (goes in the filename)")
    ap.add_argument("--camera", default=os.getenv("GESTURE_CAMERA", ""),
                    help="camera index; defaults to the one you confirmed with "
                         "the `cameras` command")
    ap.add_argument("--auto", action="store_true",
                    help="hands-free: roll takes automatically instead of "
                         "waiting for ENTER before each one")
    ap.add_argument("--mirror", action="store_true",
                    help="flip display AND recording horizontally "
                         "(off by default, to match inference)")
    ap.add_argument("--delete-bad", action="store_true",
                    help="for `verify`: delete clips that fail")
    args = ap.parse_args()

    for g in GESTURES:
        (DATA_DIR / g).mkdir(parents=True, exist_ok=True)

    if args.command == "status":
        show_status()
        return
    if args.command == "verify":
        verify_all(delete_bad=args.delete_bad)
        return
    if args.command == "cameras":
        pick_camera(args.mirror)
        return

    camera_index = resolve_camera(args.camera)
    wait_for_key = not args.auto

    if args.command == "batch":
        if not args.gesture or not args.count:
            ap.error("batch needs a gesture and a count, e.g. batch swipe_left 25")
        record_batch(args.gesture, args.count, args.subject, camera_index,
                     args.mirror, wait_for_key=wait_for_key)
        show_status()
    else:
        interactive(args.subject, camera_index, args.mirror, wait_for_key)


if __name__ == "__main__":
    try:
        main()
    except CameraError as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)
