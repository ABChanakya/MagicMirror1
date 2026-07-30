#!/usr/bin/env python3
"""
record_batch.py — Record multiple swipes per gesture in one session

Usage:
    python3 record_batch.py

Records multiple samples of EACH gesture before moving to next.
Much faster than running the script multiple times!

──────────────────────────────────────────────────────────────
MACBOOK AIR M5 SETUP (run once before recording)
──────────────────────────────────────────────────────────────
1. Install dependencies:
       pip install opencv-python mediapipe

2. Allow camera access:
   System Settings → Privacy & Security → Camera → enable Terminal

3. Run the script:
       cd Magicmirror3/camera
       python3 record_batch.py

4. If camera doesn't open, try a different index:
   Change CAMERA_INDEX below from 0 to 1 or 2.

5. After recording, copy the gesture_training_data/ folder back
   to the Linux machine and re-run:
       python preaugment.py
       python train.py  (in gesture_system/)
──────────────────────────────────────────────────────────────
"""

import platform
import cv2
import time
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "gesture_training_data"
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "null"]
RECORD_DURATION = 1  # seconds per gesture (1 sec = 30 frames @ 30fps)

# Camera index — 0 works on most machines; try 1 or 2 if camera doesn't open
CAMERA_INDEX = 0

# Ensure directories exist
for gesture in GESTURES:
    (DATA_DIR / gesture).mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("🎥 BATCH GESTURE RECORDING")
print("="*60)
print("\nHow to use:")
print("  1. For each gesture, specify how many samples to record")
print("  2. Press SPACE before each swipe (1 sec per swipe)")
print("  3. Automatically counts down between swipes")
print("  4. Moves to next gesture when done")
print("\nGestures:")
print("  • swipe_left:  Right edge → left edge (aim for 100)")
print("  • swipe_right: Left edge → right edge (aim for 100)")
print("  • swipe_up:    Bottom → top (aim for 100)")
print("  • swipe_down:  Top → bottom (aim for 100)")
print("  • null:        Keep hand still/background (aim for 50)")
print("\nPress 'q' to quit\n")

# Open camera — use AVFoundation on Mac, V4L2 on Linux
if platform.system() == "Darwin":
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"\n❌ Could not open camera {CAMERA_INDEX}.")
    print("   Try changing CAMERA_INDEX at the top of this file to 1 or 2.")
    exit(1)

total_recorded = 0

for gesture in GESTURES:
    gesture_dir = DATA_DIR / gesture
    current_count = len(list(gesture_dir.glob("*.mp4")))

    print(f"\n{'='*60}")
    print(f"Gesture: {gesture.upper()}")
    print(f"{'='*60}")
    print(f"Currently have: {current_count} samples")

    # Ask how many to record
    while True:
        try:
            num_samples = int(input(f"How many more samples to record? (0-100): ").strip())
            if 0 <= num_samples <= 100:
                break
            print("Enter a number between 0 and 100")
        except ValueError:
            print("Enter a valid number")

    if num_samples == 0:
        print("Skipping this gesture")
        continue

    # Record samples
    for sample_num in range(num_samples):
        next_num = current_count + sample_num + 1
        output_path = gesture_dir / f"{next_num:02d}.mp4"

        print(f"\n[{sample_num + 1}/{num_samples}] {gesture.upper()}")
        print(f"Output: {output_path.name}")
        print(f"Position ready, then press SPACE to start recording...")

        # Wait for SPACE to start
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                continue

            # Instructions
            cv2.putText(frame, f"Sample {sample_num + 1}/{num_samples}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to record", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, gesture.upper(), (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)
            cv2.imshow("Batch Recording", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                waiting = False
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("\n\nAborted!")
                exit(0)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (640, 480))

        # Record for RECORD_DURATION seconds
        print(f"🔴 RECORDING...")
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < RECORD_DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            # Draw recording indicator
            elapsed = time.time() - start_time
            remaining = RECORD_DURATION - elapsed
            cv2.rectangle(frame, (5, 5), (635, 475), (0, 0, 255), 3)
            cv2.putText(frame, f"RECORDING: {remaining:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"Sample {sample_num + 1}/{num_samples}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            out.write(frame)
            cv2.imshow("Batch Recording", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                if output_path.exists():
                    output_path.unlink()
                print("\n\nAborted!")
                exit(0)

        out.release()
        print(f"✅ Saved {frame_count} frames")
        total_recorded += 1

        # Countdown to next sample (if not the last one)
        if sample_num < num_samples - 1:
            for countdown in range(3, 0, -1):
                print(f"   Next sample in {countdown}...", end='\r')
                time.sleep(1)
            print("   Ready!             ")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("✨ RECORDING COMPLETE!")
print("="*60)
print(f"\n📊 Total videos recorded this session: {total_recorded}")

# Check totals
print(f"\n📊 Sample count per gesture:")
targets = {"swipe_left": 100, "swipe_right": 100, "swipe_up": 100, "swipe_down": 100, "null": 50}
total = 0
all_ready = True
for gesture in GESTURES:
    gesture_dir = DATA_DIR / gesture
    count = len(list(gesture_dir.glob("*.mp4")))
    target = targets[gesture]
    pct = (count / target) * 100 if target > 0 else 0
    status = "✅" if count >= target else "⚠️ " if count >= target * 0.5 else "❌"
    print(f"   {status} {gesture:15} {count:3}/{target} samples ({pct:3.0f}%)")
    total += count
    if count < target:
        all_ready = False

total_target = sum(targets.values())
print(f"\n   Total: {total}/{total_target} samples")

if all_ready:
    print("\n✅ Ready to extract landmarks and train!")
else:
    print(f"\n⚠️  Need {total_target - total} more samples")

print("\nNext steps:")
print("  python3 collect_gesture_data.py")
print("  (Choose option 2: Extract landmarks)")
