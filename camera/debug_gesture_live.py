#!/usr/bin/env python3
"""
debug_gesture_live.py — Detailed debugging for gesture detection

Shows:
- Hand detection (MediaPipe)
- Motion features being computed
- Model predictions per frame (not just high-confidence)
- Why gestures aren't triggering
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
import time

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "gesture_model.keras"

GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "null"]
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 5
MAX_FRAMES = 30
SMOOTHING_WINDOW = 20
CONFIDENCE_THRESHOLD = 0.98


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize by wrist origin and scale."""
    frames = len(sequence)
    if frames == 0:
        return sequence
    reshaped = sequence.reshape(frames, NUM_LANDMARKS, 3)
    wrist_origin = reshaped[0, 0]
    middle_tip = reshaped[0, 12]
    centered = reshaped - wrist_origin
    scale = float(np.linalg.norm(middle_tip - wrist_origin))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (centered / scale).astype(np.float32).reshape(frames, -1)


def extract_motion_features(landmarks_seq: np.ndarray) -> np.ndarray:
    """Convert raw landmarks to motion features."""
    frames = len(landmarks_seq)
    normalized = normalize_sequence(landmarks_seq).reshape(frames, NUM_LANDMARKS, 3)
    relative_position = normalized - normalized[:1]
    velocity = np.zeros_like(normalized)
    velocity[1:] = normalized[1:] - normalized[:-1]
    acceleration = np.zeros_like(normalized)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    speed = np.linalg.norm(velocity, axis=-1)
    direction = np.arctan2(velocity[..., 1], velocity[..., 0])
    acceleration_magnitude = np.linalg.norm(acceleration, axis=-1)
    features = np.stack([
        relative_position[..., 0],
        relative_position[..., 1],
        speed,
        direction,
        acceleration_magnitude,
    ], axis=-1)
    return features.astype(np.float32)


print(f"Loading model: {MODEL_PATH}")
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)

model = tf.keras.models.load_model(str(MODEL_PATH))
print("✅ Model loaded\n")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

print("🎥 Starting webcam... Press Q to quit\n")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

landmark_buffer = deque(maxlen=MAX_FRAMES)
prediction_queue = deque(maxlen=SMOOTHING_WINDOW)
last_swipe_time = 0.0

frame_count = 0
hands_detected = 0
buffer_full_count = 0
high_conf_count = 0
swipe_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # === HAND DETECTION ===
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hands_detected += 1
            hand = results.multi_hand_landmarks[0]

            # Extract landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand.landmark
            ], dtype=np.float32).flatten()

            landmark_buffer.append(landmarks)

            # Get hand position
            wrist = hand.landmark[0]
            hand_x = int(wrist.x * w)
            hand_y = int(wrist.y * h)

            # Draw hand marker
            cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 0), -1)

            # === MODEL PREDICTION ===
            if len(landmark_buffer) == MAX_FRAMES:
                buffer_full_count += 1

                # Extract motion features
                padded = np.zeros((MAX_FRAMES, 63), dtype=np.float32)
                for i, lm in enumerate(landmark_buffer):
                    padded[i] = lm

                motion_features = extract_motion_features(padded)
                input_tensor = np.expand_dims(motion_features, axis=0)
                predictions = model.predict(input_tensor, verbose=0)[0]

                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
                gesture_name = GESTURES[predicted_class]

                # Show all predictions
                cv2.putText(frame, f"Predictions (buffer full):", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                for i, (gesture, conf) in enumerate(zip(GESTURES, predictions)):
                    color = (0, 200, 0) if i == predicted_class else (100, 100, 100)
                    cv2.putText(frame, f"  {gesture:12} {conf*100:5.1f}%", (10, 85 + i*20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Check confidence
                if confidence >= CONFIDENCE_THRESHOLD:
                    high_conf_count += 1
                    prediction_queue.append(predicted_class)

                    # Check if all recent predictions agree
                    if len(prediction_queue) == SMOOTHING_WINDOW:
                        if len(set(prediction_queue)) == 1:
                            # All frames agree!
                            now = time.monotonic()
                            if now - last_swipe_time > 0.8:
                                swipe_count += 1
                                last_swipe_time = now
                                print(f"✅ SWIPE #{swipe_count}: {gesture_name.upper()} ({confidence*100:.1f}%)")
                                prediction_queue.clear()
                        else:
                            # Show disagreement
                            unique_gestures = set(GESTURES[i] for i in prediction_queue)
                            cv2.putText(frame, f"⚠️  Inconsistent: {unique_gestures}", (10, 250),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                            prediction_queue.clear()

        else:
            cv2.putText(frame, "❌ No hand detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            landmark_buffer.clear()
            prediction_queue.clear()

        # === STATUS BAR ===
        cv2.putText(frame, f"Frame: {frame_count} | Hands: {hands_detected} | Buffer_Full: {buffer_full_count} | HighConf: {high_conf_count} | Swipes: {swipe_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(frame, f"Buffer: {len(landmark_buffer)}/{MAX_FRAMES} | Queue: {len(prediction_queue)}/{SMOOTHING_WINDOW}",
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "Press Q to quit | Press L to lower confidence threshold", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow("Gesture Debug", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            CONFIDENCE_THRESHOLD = max(0.5, CONFIDENCE_THRESHOLD - 0.05)
            print(f"Lowered confidence threshold to {CONFIDENCE_THRESHOLD*100:.0f}%")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

print(f"\n📊 DEBUG SUMMARY")
print(f"Total frames: {frame_count}")
print(f"Hands detected in: {hands_detected} frames ({hands_detected*100/frame_count:.1f}%)")
print(f"Buffer full: {buffer_full_count} times")
print(f"High confidence: {high_conf_count} times")
print(f"Swipes detected: {swipe_count}")
print(f"\nIssue diagnosis:")
if hands_detected < frame_count * 0.5:
    print("  ⚠️  Hand detection is poor - try different lighting or camera angle")
elif buffer_full_count < hands_detected * 0.5:
    print("  ⚠️  Hand tracking is inconsistent - jumpy or lost hands")
elif high_conf_count < buffer_full_count * 0.5:
    print("  ⚠️  Model predictions lack confidence - try lowering threshold with 'L'")
elif swipe_count == 0 and high_conf_count > 0:
    print("  ⚠️  Predictions don't stay consistent for 20 frames - gestures too fast/variable")
else:
    print("  ✅ System working - perform larger/slower gestures")
