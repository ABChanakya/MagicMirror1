"""
gesture_detector.py
Finger-count gesture detection using MediaPipe Hands.

Recognised gestures: fingers_1 … fingers_5
hold_progress (0.0–1.0) is updated every frame and exposed for the UI progress bar.
"""

import logging
import os
import time
from collections import deque

import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

logger = logging.getLogger(__name__)

FINGER_EXTENSION_THRESHOLD = 0.02  # normalised units; tip must be this much above PIP


class GestureDetector:
    def __init__(
        self,
        finger_hold_seconds: float | None = None,
    ):
        if finger_hold_seconds is None:
            finger_hold_seconds = float(os.getenv("FINGER_HOLD_SECONDS", "0.5"))

        self.finger_hold_seconds    = finger_hold_seconds
        self.finger_debounce_frames = int(os.getenv("FINGER_DEBOUNCE_FRAMES", "3"))

        self.enabled            = True
        self.backend            = "mediapipe"
        self.gesture_model_name = ""
        self.yolo_runtime       = ""
        self.yolo_model         = None

        self.mp_hands = None
        self.hands    = None
        self._init_mediapipe()

        logger.info(
            "GestureDetector: hold=%.2fs  debounce=%d frames",
            self.finger_hold_seconds, self.finger_debounce_frames,
        )

        # Temporal smoothing — majority vote over last 5 raw counts
        self._raw_count_buf: deque = deque(maxlen=5)

        # Hold tracking
        self._finger_count_start: float | None = None
        self._last_stable_count:  int   | None = None
        self._hold_frames = 0

        # Public attributes read by main.py every frame
        self.last_landmarks    = None
        self.last_finger_state: int | None = None   # stable finger count (0-5)
        self.hold_progress: float = 0.0             # 0.0–1.0

    # ── Backend init ───────────────────────────────────────────────────────

    def _init_mediapipe(self):
        if mp is None or not hasattr(mp, "solutions"):
            logger.warning("MediaPipe unavailable — gesture detection disabled.")
            self.enabled = False
            return

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=float(os.getenv("MP_HAND_DET_CONF",   "0.3")),
            min_tracking_confidence= float(os.getenv("MP_HAND_TRACK_CONF", "0.3")),
        )
        logger.info("Gesture backend: mediapipe")

    # ── Public API ─────────────────────────────────────────────────────────

    def process_frame(self, rgb_frame: np.ndarray) -> str | None:
        """
        Process one RGB frame.
        Updates self.last_landmarks, self.last_finger_state, self.hold_progress.
        Returns "fingers_N" when the hold threshold is first reached, else None.
        main.py owns cooldown / state-machine logic.
        """
        if not self.enabled:
            return None

        result = self.hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            self._reset()
            return None

        landmarks  = result.multi_hand_landmarks[0].landmark
        handedness = result.multi_handedness[0].classification[0].label  # "Left" or "Right"
        self.last_landmarks = landmarks
        now = time.monotonic()

        raw_count = self._count_extended_fingers(landmarks, handedness)
        self._raw_count_buf.append(raw_count)

        stable_count = self._majority_count()
        self.last_finger_state = stable_count if stable_count >= 0 else None

        return self._detect_finger_hold(stable_count, now)

    def close(self):
        if self.hands is not None:
            self.hands.close()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _reset(self):
        self.last_landmarks        = None
        self.last_finger_state     = None
        self.hold_progress         = 0.0
        self._finger_count_start   = None
        self._last_stable_count    = None
        self._hold_frames          = 0
        self._raw_count_buf.clear()

    def _majority_count(self) -> int:
        """Return the count that appears >= 3 times in the last 5 frames, else -1."""
        buf = list(self._raw_count_buf)
        if len(buf) < 3:
            return -1
        for candidate in set(buf):
            if buf.count(candidate) >= 3:
                return candidate
        return -1

    def _count_extended_fingers(self, landmarks, handedness: str) -> int:
        H = self.mp_hands.HandLandmark

        # ── Four fingers: tip above PIP AND tip above MCP ─────────────────
        finger_info = [
            (H.INDEX_FINGER_TIP,  H.INDEX_FINGER_PIP,  H.INDEX_FINGER_MCP),
            (H.MIDDLE_FINGER_TIP, H.MIDDLE_FINGER_PIP, H.MIDDLE_FINGER_MCP),
            (H.RING_FINGER_TIP,   H.RING_FINGER_PIP,   H.RING_FINGER_MCP),
            (H.PINKY_TIP,         H.PINKY_PIP,         H.PINKY_MCP),
        ]

        count = 0
        for tip_idx, pip_idx, mcp_idx in finger_info:
            tip_y = landmarks[tip_idx].y
            pip_y = landmarks[pip_idx].y
            mcp_y = landmarks[mcp_idx].y
            if (pip_y - tip_y) > FINGER_EXTENSION_THRESHOLD and tip_y < mcp_y:
                count += 1

        # ── Thumb: X-axis, handedness-aware ──────────────────────────────
        thumb_tip = landmarks[H.THUMB_TIP]
        thumb_cmc = landmarks[H.THUMB_CMC]

        if handedness == "Right":
            thumb_extended = thumb_tip.x < (thumb_cmc.x - 0.04)
        else:
            thumb_extended = thumb_tip.x > (thumb_cmc.x + 0.04)

        if thumb_extended:
            count += 1

        return count

    def _detect_finger_hold(self, stable_count: int, now: float) -> str | None:
        """
        Fire "fingers_N" once when the stable count has been held for
        finger_hold_seconds and debounce_frames frames.
        Updates self.hold_progress every call.
        """
        if stable_count < 0:
            self.hold_progress = 0.0
            return None

        if stable_count != self._last_stable_count:
            # Pose changed — restart window
            self._last_stable_count  = stable_count
            self._finger_count_start = now
            self._hold_frames        = 0
            self.hold_progress       = 0.0
            return None

        self._hold_frames += 1
        held = now - (self._finger_count_start or now)
        self.hold_progress = min(1.0, held / self.finger_hold_seconds)

        frames_ok = self._hold_frames >= self.finger_debounce_frames
        time_ok   = held >= self.finger_hold_seconds

        if time_ok and frames_ok and stable_count in (1, 2, 3, 4, 5):
            # Advance start so it doesn't re-fire every frame
            self._finger_count_start = now + self.finger_hold_seconds
            self._hold_frames        = 0
            return f"fingers_{stable_count}"

        return None
