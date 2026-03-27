"""
gesture_detector.py
Detects hand gestures using MediaPipe Hands.

Recognised gestures:
  swipe_left, swipe_right, swipe_up, swipe_down  — hand movement across frames
  fingers_1, fingers_2, fingers_3, fingers_4     — extended finger count (held stable)
"""

import logging
import time
from collections import deque

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class GestureDetector:
    def __init__(
        self,
        swipe_threshold: float = 0.18,   # fraction of frame width/height
        finger_hold_seconds: float = 0.9, # how long finger count must be stable
        gesture_cooldown: float = 2.5,    # seconds between same gesture fires
    ):
        self.swipe_threshold = swipe_threshold
        self.finger_hold_seconds = finger_hold_seconds
        self.gesture_cooldown = gesture_cooldown

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        # Swipe tracking
        self._positions: deque = deque(maxlen=12)  # (x, y, timestamp)

        # Finger-count stability tracking
        self._finger_count_history: deque = deque(maxlen=20)
        self._finger_count_start: float | None = None
        self._last_finger_count: int | None = None

        # Cooldown per gesture name
        self._last_fired: dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def process_frame(self, rgb_frame: np.ndarray) -> str | None:
        """
        Process one RGB frame. Returns a gesture name string if one fires,
        otherwise None.
        """
        result = self.hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            self._positions.clear()
            self._finger_count_history.clear()
            self._finger_count_start = None
            self._last_finger_count = None
            return None

        landmarks = result.multi_hand_landmarks[0].landmark
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        cx, cy = wrist.x, wrist.y
        now = time.monotonic()

        self._positions.append((cx, cy, now))

        # Check swipe first (takes priority over finger count)
        swipe = self._detect_swipe()
        if swipe and self._can_fire(swipe, now):
            self._last_fired[swipe] = now
            self._positions.clear()
            return swipe

        # Check finger count
        count = self._count_extended_fingers(landmarks)
        finger_gesture = self._detect_finger_hold(count, now)
        if finger_gesture and self._can_fire(finger_gesture, now):
            self._last_fired[finger_gesture] = now
            return finger_gesture

        return None

    def close(self):
        self.hands.close()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _detect_swipe(self) -> str | None:
        if len(self._positions) < 8:
            return None
        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        adx, ady = abs(dx), abs(dy)

        if adx < self.swipe_threshold and ady < self.swipe_threshold:
            return None

        if adx > ady:
            return "swipe_left" if dx < 0 else "swipe_right"
        else:
            return "swipe_up" if dy < 0 else "swipe_down"

    def _count_extended_fingers(self, landmarks) -> int:
        """Count extended fingers (excluding thumb)."""
        tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP,
        ]
        pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP,
        ]
        count = 0
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                count += 1
        return count

    def _detect_finger_hold(self, count: int, now: float) -> str | None:
        if count not in (1, 2, 3, 4):
            self._finger_count_start = None
            self._last_finger_count = None
            self._finger_count_history.clear()
            return None

        if count != self._last_finger_count:
            self._last_finger_count = count
            self._finger_count_start = now
            self._finger_count_history.clear()

        self._finger_count_history.append(count)

        held = now - (self._finger_count_start or now)
        if held >= self.finger_hold_seconds and len(self._finger_count_history) >= 10:
            # Reset so it doesn't fire continuously
            self._finger_count_start = now + self.finger_hold_seconds
            return f"fingers_{count}"

        return None

    def _can_fire(self, name: str, now: float) -> bool:
        last = self._last_fired.get(name, 0)
        return (now - last) >= self.gesture_cooldown
