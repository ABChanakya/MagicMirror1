"""
Rule-based swipe detector using MediaPipe hand landmarks.

Works for any person — no training data, no model weights.
Detects: swipe_left, swipe_right, swipe_up, swipe_down, (null = no gesture)

Algorithm:
  1. Track wrist (landmark 0) across a sliding window of frames
  2. Normalize displacement by hand scale (wrist→middle-MCP distance)
     so hand size doesn't matter
  3. If normalized displacement exceeds threshold AND motion is
     sufficiently directional → fire gesture
"""

from collections import deque

import mediapipe as mp
import numpy as np


# ── Tunable constants ──────────────────────────────────────────────────────────
WINDOW_FRAMES      = 18    # frames over which to measure displacement
SWIPE_THRESHOLD    = 0.6   # min normalized displacement to count as a swipe
DIRECTION_RATIO    = 2.5   # dominant axis must be this much larger than other
COOLDOWN_FRAMES    = 30    # frames to ignore after a gesture fires
# ──────────────────────────────────────────────────────────────────────────────

_mp_hands = mp.solutions.hands


class GestureDetectorRuleBased:
    """Drop-in replacement for the deleted gesture_detector.py."""

    def __init__(self):
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        # Ring buffer of (norm_x, norm_y) wrist positions
        self._wrist_buf: deque = deque(maxlen=WINDOW_FRAMES)
        self._cooldown   = 0
        self.last_landmarks = None
        self.hand_x: float = 0.5
        self.hand_y: float = 0.5

    # ── Public API (same as old GestureDetector) ───────────────────────────────

    def process_frame(self, rgb_frame) -> tuple:
        """
        Args:
            rgb_frame: H×W×3 uint8 numpy array (RGB)
        Returns:
            (gesture_event_dict | None, hand_present: bool)
        """
        result = self._hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            self.last_landmarks = None
            self._wrist_buf.clear()   # reset on hand loss
            if self._cooldown:
                self._cooldown -= 1
            return None, False

        lm = result.multi_hand_landmarks[0]
        self.last_landmarks = lm

        # Wrist and middle-MCP in image-normalised coords [0,1]
        wrist  = lm.landmark[0]
        mid_mcp = lm.landmark[9]

        self.hand_x = wrist.x
        self.hand_y = wrist.y

        # Hand scale = distance wrist → middle MCP (person-invariant normaliser)
        scale = np.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y)
        scale = max(scale, 0.01)   # guard against degenerate detections

        self._wrist_buf.append((wrist.x / scale, wrist.y / scale))

        if self._cooldown:
            self._cooldown -= 1
            return None, True

        if len(self._wrist_buf) < WINDOW_FRAMES:
            return None, True

        gesture = self._classify()
        if gesture:
            self._cooldown = COOLDOWN_FRAMES
            self._wrist_buf.clear()
            return {"type": "gesture", "name": gesture}, True

        return None, True

    # ── Internal ───────────────────────────────────────────────────────────────

    def _classify(self) -> str | None:
        positions = np.array(self._wrist_buf)   # (WINDOW_FRAMES, 2)

        # Use first→last displacement for robustness against mid-path wobble
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]

        abs_dx, abs_dy = abs(dx), abs(dy)
        magnitude = np.hypot(dx, dy)

        if magnitude < SWIPE_THRESHOLD:
            return None

        # Reject ambiguous diagonal motions
        if abs_dx == 0 or abs_dy == 0:
            return None
        if max(abs_dx, abs_dy) / min(abs_dx, abs_dy) < DIRECTION_RATIO:
            return None

        if abs_dx > abs_dy:
            return "swipe_right" if dx > 0 else "swipe_left"
        else:
            # Y axis is flipped in image coords (0=top)
            return "swipe_down" if dy > 0 else "swipe_up"
