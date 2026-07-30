#!/usr/bin/env python3
"""
record_auto.py — Simple automatic gesture recording

Usage:
    python3 record_auto.py

Just press SPACE to start recording each gesture!
Records 3 seconds per gesture, automatically moves to next one.
"""

import cv2
import time
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "gesture_training_data"
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down"]
RECORD_DURATION = 1  # seconds per gesture

# Ensure directories exist
for gesture in GESTURES:
    (DATA_DIR / gesture).mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("🎥 GESTURE RECORDING — AUTO MODE")
print("="*60)
print("\nHow to use:")
print("  1. Position your hand in the camera")
print("  2. Press SPACE to start recording")
print("  3. Do the gesture quickly (1 second)")
print("  4. Automatically moves to next gesture")
print("  5. Repeat for all 4 gestures")
print("\nGestures:")
print("  • swipe_left:  Right edge → left edge")
print("  • swipe_right: Left edge → right edge")
print("  • swipe_up:    Bottom → top")
print("  • swipe_down:  Top → bottom")
print("\nPress 'q' to quit\n")

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

for gesture_idx, gesture in enumerate(GESTURES):
    # Get next sample number
    gesture_dir = DATA_DIR / gesture
    next_num = len(list(gesture_dir.glob("*.mp4"))) + 1
    output_path = gesture_dir / f"{next_num:02d}.mp4"

    print(f"\n{'='*60}")
    print(f"Gesture {gesture_idx + 1}/4: {gesture.upper()}")
    print(f"{'='*60}")
    print(f"Output: {output_path.name}")
    print(f"\nPosition ready, then press SPACE to start recording...")

    # Wait for SPACE to start
    waiting = True
    while waiting:
        ret, frame = cap.read()
        if not ret:
            continue

        # Instructions
        cv2.putText(frame, "Press SPACE to record", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, gesture.upper(), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)
        cv2.imshow("Recording Setup", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            waiting = False
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("\nAborted!")
            exit(0)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 15.0, (640, 480))

    # Record for RECORD_DURATION seconds
    print(f"\n🔴 RECORDING {gesture.upper()}...")
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
        cv2.putText(frame, f"RECORDING: {remaining:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, gesture.upper(), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

        out.write(frame)
        cv2.imshow("Recording Setup", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            if output_path.exists():
                output_path.unlink()
            print("\nAborted!")
            exit(0)

    out.release()
    print(f"✅ Saved {frame_count} frames to {output_path.name}")

    # Show what's next
    if gesture_idx < len(GESTURES) - 1:
        next_gesture = GESTURES[gesture_idx + 1]
        print(f"\nNext: {next_gesture}")
        print("Repositioning in 3 seconds...")
        time.sleep(3)
    else:
        print("\n🎉 ALL GESTURES RECORDED!")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("✨ Recording complete!")
print("="*60)
print("\nNow extract landmarks:")
print("  python3 collect_gesture_data.py")
print("  (Choose option 2)")
