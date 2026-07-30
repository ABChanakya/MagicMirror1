#!/usr/bin/env python3
"""
collect_gesture_data.py — Record hand gesture videos and extract MediaPipe landmarks

Usage:
    python3 collect_gesture_data.py

This will:
1. Show real-time webcam with hand detection
2. Record videos for each gesture (swipe_left, swipe_right, swipe_up, swipe_down)
3. Extract MediaPipe landmarks from videos
4. Save landmark sequences to gesture_training_data/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from pathlib import Path
from collections import deque

# Setup
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "gesture_training_data"
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down"]

# Ensure directories exist
for gesture in GESTURES:
    (DATA_DIR / gesture).mkdir(parents=True, exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils


def get_landmark_sequence(video_path, max_frames=60):
    """
    Extract hand landmarks from a video file.
    Returns: list of (landmarks_array) or None if hand not detected consistently
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Cannot open {video_path}")
        return None

    sequences = []
    frame_count = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark
            # Convert to numpy array: [x, y, z] × 21 joints = 63 values
            lm_array = np.array([
                [lm.x, lm.y, lm.z] for lm in landmarks
            ], dtype=np.float32).flatten()  # Shape: (63,)
            sequences.append(lm_array)
            detected_frames += 1

    cap.release()

    if len(sequences) < 10:
        print(f"  ⚠️  Only {detected_frames}/{frame_count} frames with hand detected")
        return None

    return np.array(sequences, dtype=np.float32)  # Shape: (num_frames, 63)


def record_gesture(gesture_name, sample_num):
    """
    Record a single gesture video from webcam.
    """
    print(f"\n📹 Recording {gesture_name} (sample {sample_num})")
    print(f"   Show your {gesture_name} gesture...")
    print(f"   Press 's' to START, 'q' to QUIT, or SPACE when done")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Video writer
    output_path = DATA_DIR / gesture_name / f"{sample_num:02d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (640, 480))

    recording = False
    frame_count = 0
    instruction_text = "Press 's' to START recording"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw hand skeleton
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw instructions
        cv2.putText(frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if recording:
            cv2.putText(frame, f"RECORDING: {frame_count} frames", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (5, 5), (635, 475), (0, 0, 255), 3)

        cv2.imshow(f"Recording: {gesture_name}", frame)

        if recording:
            out.write(frame)
            frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
            frame_count = 0
            instruction_text = "Recording... Press SPACE when done"
        elif key == ord(' ') and recording:
            break
        elif key == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            if output_path.exists():
                output_path.unlink()
            return False

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if frame_count < 30:
        print(f"  ❌ Too few frames ({frame_count}), discarding")
        output_path.unlink()
        return False

    print(f"  ✅ Saved {frame_count} frames to {output_path.name}")
    return True


def extract_all_landmarks():
    """
    Extract landmarks from all recorded videos.
    """
    print("\n" + "="*60)
    print("🔍 EXTRACTING LANDMARKS FROM ALL VIDEOS")
    print("="*60)

    for gesture in GESTURES:
        gesture_dir = DATA_DIR / gesture
        video_files = sorted(gesture_dir.glob("*.mp4"))

        if not video_files:
            print(f"\n{gesture}: No videos yet")
            continue

        print(f"\n{gesture}:")
        for video_path in video_files:
            print(f"  Processing {video_path.name}...", end=" ")
            sequences = get_landmark_sequence(video_path)

            if sequences is not None:
                npy_path = video_path.with_suffix(".npy")
                np.save(str(npy_path), sequences)
                print(f"✅ Saved {sequences.shape[0]} frames")
            else:
                print(f"❌ Skipped (no hand detected)")


def show_statistics():
    """
    Show how many samples we have per gesture.
    """
    print("\n" + "="*60)
    print("📊 DATA COLLECTION STATUS")
    print("="*60)

    total = 0
    for gesture in GESTURES:
        gesture_dir = DATA_DIR / gesture
        npy_files = list(gesture_dir.glob("*.npy"))
        count = len(npy_files)
        total += count
        status = "✅" if count >= 25 else "⚠️ " if count >= 10 else "❌"
        print(f"{status} {gesture:15} {count:3}/25 samples")

    print(f"\nTotal: {total}/100 samples collected")
    if total >= 100:
        print("✅ Ready to train in Colab!")
    elif total >= 80:
        print("⚠️  Almost ready (aim for 100)")
    else:
        print(f"❌ Need {100 - total} more samples")


def main():
    print("\n" + "="*60)
    print("🎮 GESTURE DATA COLLECTION")
    print("="*60)
    print("\nOptions:")
    print("  1) Record new gesture samples")
    print("  2) Extract landmarks from videos")
    print("  3) Show collection status")
    print("  4) Exit")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            print("\nWhich gesture?")
            for i, gesture in enumerate(GESTURES, 1):
                print(f"  {i}) {gesture}")
            gesture_choice = input("Enter (1-4): ").strip()

            try:
                gesture_idx = int(gesture_choice) - 1
                if 0 <= gesture_idx < len(GESTURES):
                    gesture = GESTURES[gesture_idx]
                    sample_num = len(list((DATA_DIR / gesture).glob("*.mp4"))) + 1
                    record_gesture(gesture, sample_num)
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid input")

        elif choice == "2":
            extract_all_landmarks()

        elif choice == "3":
            show_statistics()

        elif choice == "4":
            print("Goodbye! 👋")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
