"""Gesture detection engine for the Neural Gesture Sculptor project."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

import numpy as np


@dataclass
class HandLandmarks:
    """Container for one tracked hand."""

    label: str
    landmarks_norm: np.ndarray  # (21, 3) normalized MediaPipe coordinates
    landmarks_px: np.ndarray  # (21, 2) pixel coordinates


@dataclass
class GestureState:
    """Current gesture state emitted by GestureEngine."""

    name: str = "IDLE"
    circle_triggered: bool = False
    pinch_active: bool = False
    pinch_point: Optional[Tuple[float, float]] = None
    scale_delta: float = 1.0
    rotate_delta: Tuple[float, float] = (0.0, 0.0)  # (yaw_delta, pitch_delta)
    hands_count: int = 0


class GestureEngine:
    """Interprets hand landmarks and emits high-level gesture states."""

    CIRCLE_MIN_FRAMES = 28
    CIRCLE_TRAIL_SIZE = 72
    CIRCLE_COOLDOWN_SECONDS = 1.0

    PINCH_THRESHOLD_PX = 40.0
    PINCH_REQUIRED_FRAMES = 3

    ROTATION_DEAD_ZONE_DEG = 5.0
    MAX_NO_HAND_FRAMES_FOR_CIRCLE = 4

    def __init__(self) -> None:
        self.index_history: Deque[np.ndarray] = deque(maxlen=self.CIRCLE_TRAIL_SIZE)
        self._last_circle_time = 0.0
        self._no_hand_frames = 0

        self._pinch_frames = 0
        self._last_two_hands_distance: Optional[float] = None
        self._last_rotate_angles: Optional[Tuple[float, float]] = None

    def update(
        self,
        hands: Sequence[HandLandmarks],
        frame_size: Tuple[int, int],
        now_s: float,
    ) -> GestureState:
        """Compute current gesture from detected hands."""

        state = GestureState(hands_count=len(hands))

        if not hands:
            self._no_hand_frames += 1
            self._pinch_frames = 0
            self._last_two_hands_distance = None
            self._last_rotate_angles = None
            if self._no_hand_frames > self.MAX_NO_HAND_FRAMES_FOR_CIRCLE:
                self.index_history.clear()
            return state
        self._no_hand_frames = 0

        primary = hands[0]
        index_point = primary.landmarks_px[8].astype(np.float32)
        if self.index_history:
            jump = float(np.linalg.norm(index_point - self.index_history[-1]))
            # Ignore sudden tracking jumps that pollute the circle path.
            if jump < 180.0:
                self.index_history.append(index_point)
        else:
            self.index_history.append(index_point)

        if self._detect_circle(now_s):
            state.name = "CIRCLE"
            state.circle_triggered = True
            self._pinch_frames = 0
            self._last_two_hands_distance = None
            self._last_rotate_angles = None
            return state

        if len(hands) == 2:
            state.name = "TWO_HANDS"
            state.scale_delta = self._detect_two_hands_scale(hands[0], hands[1], frame_size)
            self._pinch_frames = 0
            self._last_rotate_angles = None
            return state
        self._last_two_hands_distance = None

        pinch_active, pinch_point = self._detect_pinch(primary)
        if pinch_active:
            state.name = "PINCH"
            state.pinch_active = True
            state.pinch_point = pinch_point
            self._last_rotate_angles = None
            return state

        rotate_delta = self._detect_rotate(primary)
        if rotate_delta != (0.0, 0.0):
            state.name = "ROTATE"
            state.rotate_delta = rotate_delta

        return state

    def _detect_circle(self, now_s: float) -> bool:
        if len(self.index_history) < self.CIRCLE_MIN_FRAMES:
            return False
        if now_s - self._last_circle_time < self.CIRCLE_COOLDOWN_SECONDS:
            return False

        points = np.array(self.index_history, dtype=np.float32)
        center = points.mean(axis=0)
        radii = np.linalg.norm(points - center, axis=1)
        mean_radius = float(np.mean(radii))

        if mean_radius < 18.0:
            return False

        path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if path_length <= 1e-6:
            return False

        return_distance = float(np.linalg.norm(points[0] - points[-1]))
        diameter = float(
            np.max(np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2))
        )
        radius_spread = float(np.std(radii) / (mean_radius + 1e-6))

        # Require enough angular coverage to avoid random scribbles.
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        bins = np.floor(((angles + np.pi) / (2.0 * np.pi)) * 12.0).astype(np.int32)
        bins = np.clip(bins, 0, 11)
        covered_bins = int(np.unique(bins).size)

        if covered_bins < 8:
            return False
        if path_length < 4.2 * mean_radius:
            return False
        if return_distance > 0.65 * mean_radius:
            return False
        if diameter < 1.5 * mean_radius:
            return False
        if radius_spread > 0.65:
            return False

        self._last_circle_time = now_s
        self.index_history.clear()
        return True

    def _detect_pinch(self, hand: HandLandmarks) -> Tuple[bool, Optional[Tuple[float, float]]]:
        thumb_tip = hand.landmarks_px[4]
        index_tip = hand.landmarks_px[8]
        distance = float(np.linalg.norm(thumb_tip - index_tip))

        if distance < self.PINCH_THRESHOLD_PX:
            self._pinch_frames += 1
        else:
            self._pinch_frames = 0

        if self._pinch_frames >= self.PINCH_REQUIRED_FRAMES:
            pinch_midpoint = (thumb_tip + index_tip) / 2.0
            return True, (float(pinch_midpoint[0]), float(pinch_midpoint[1]))
        return False, None

    def _detect_two_hands_scale(
        self,
        hand_a: HandLandmarks,
        hand_b: HandLandmarks,
        frame_size: Tuple[int, int],
    ) -> float:
        center_a = hand_a.landmarks_px[9]
        center_b = hand_b.landmarks_px[9]
        current_distance = float(np.linalg.norm(center_a - center_b))

        if self._last_two_hands_distance is None:
            self._last_two_hands_distance = current_distance
            return 1.0

        width, height = frame_size
        diagonal = float(np.hypot(width, height))
        normalized_delta = (current_distance - self._last_two_hands_distance) / max(diagonal, 1.0)
        self._last_two_hands_distance = current_distance

        return float(1.0 + np.clip(normalized_delta * 4.0, -0.08, 0.08))

    def _detect_rotate(self, hand: HandLandmarks) -> Tuple[float, float]:
        if not self._is_hand_open(hand):
            self._last_rotate_angles = None
            return 0.0, 0.0

        wrist = hand.landmarks_px[0]
        middle_tip = hand.landmarks_px[12]
        vector = middle_tip - wrist

        yaw_deg = float(np.degrees(np.arctan2(vector[0], 200.0)))
        pitch_deg = float(np.degrees(np.arctan2(-vector[1], 200.0)))
        current_angles = (yaw_deg, pitch_deg)

        if self._last_rotate_angles is None:
            self._last_rotate_angles = current_angles
            return 0.0, 0.0

        dyaw = float(np.clip(current_angles[0] - self._last_rotate_angles[0], -12.0, 12.0))
        dpitch = float(np.clip(current_angles[1] - self._last_rotate_angles[1], -12.0, 12.0))
        self._last_rotate_angles = current_angles

        if abs(dyaw) < self.ROTATION_DEAD_ZONE_DEG:
            dyaw = 0.0
        if abs(dpitch) < self.ROTATION_DEAD_ZONE_DEG:
            dpitch = 0.0
        return dyaw, dpitch

    @staticmethod
    def _is_hand_open(hand: HandLandmarks) -> bool:
        tips_pips = ((8, 6), (12, 10), (16, 14), (20, 18))
        points = hand.landmarks_px
        extended_count = sum(1 for tip, pip in tips_pips if points[tip, 1] < points[pip, 1])
        return extended_count >= 3
