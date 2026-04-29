"""Unit tests for GestureEngine."""

from __future__ import annotations

import math
import unittest

import numpy as np

from gesture_engine import GestureEngine, HandLandmarks


def make_hand(
    index_xy: tuple[float, float] = (100.0, 100.0),
    thumb_xy: tuple[float, float] = (130.0, 100.0),
    palm_xy: tuple[float, float] = (100.0, 140.0),
) -> HandLandmarks:
    norm = np.zeros((21, 3), dtype=np.float32)
    px = np.zeros((21, 2), dtype=np.float32)
    px[:, 0] = 100.0
    px[:, 1] = 100.0

    # Mandatory points used by gesture engine.
    px[0] = palm_xy
    px[4] = thumb_xy
    px[8] = index_xy
    px[9] = palm_xy
    px[12] = (index_xy[0], index_xy[1] - 20.0)
    px[6] = (index_xy[0], index_xy[1] + 12.0)
    px[10] = (index_xy[0], index_xy[1] + 12.0)
    px[14] = (index_xy[0], index_xy[1] + 12.0)
    px[18] = (index_xy[0], index_xy[1] + 12.0)

    return HandLandmarks(label="Right", landmarks_norm=norm, landmarks_px=px)


class GestureEngineTests(unittest.TestCase):
    def test_pinch_after_three_frames(self) -> None:
        engine = GestureEngine()
        hand = make_hand(index_xy=(120, 120), thumb_xy=(125, 122))
        now = 0.0

        for _ in range(2):
            state = engine.update([hand], (1280, 720), now)
            self.assertFalse(state.pinch_active)
            now += 1 / 30

        state = engine.update([hand], (1280, 720), now)
        self.assertTrue(state.pinch_active)
        self.assertEqual(state.name, "PINCH")
        self.assertIsNotNone(state.pinch_point)

    def test_two_hands_scale_delta(self) -> None:
        engine = GestureEngine()
        hand_a = make_hand(palm_xy=(200, 200))
        hand_b = make_hand(palm_xy=(400, 200))
        engine.update([hand_a, hand_b], (1280, 720), 0.0)  # baseline

        hand_b_farther = make_hand(palm_xy=(520, 200))
        state = engine.update([hand_a, hand_b_farther], (1280, 720), 1 / 30)
        self.assertEqual(state.name, "TWO_HANDS")
        self.assertGreater(state.scale_delta, 1.0)

    def test_circle_detection(self) -> None:
        engine = GestureEngine()
        cx, cy, r = 300.0, 200.0, 60.0
        now = 0.0
        triggered = False
        for i in range(40):
            angle = (2.0 * math.pi * i) / 39.0
            index_xy = (cx + r * math.cos(angle), cy + r * math.sin(angle))
            hand = make_hand(index_xy=index_xy, thumb_xy=(index_xy[0] + 40, index_xy[1]))
            state = engine.update([hand], (1280, 720), now)
            triggered = triggered or state.circle_triggered
            now += 1 / 30

        self.assertTrue(triggered)

    def test_circle_history_survives_short_tracking_drop(self) -> None:
        engine = GestureEngine()
        hand = make_hand(index_xy=(220, 180), thumb_xy=(270, 180))
        now = 0.0

        for _ in range(12):
            engine.update([hand], (1280, 720), now)
            now += 1 / 30
        history_before_gap = len(engine.index_history)

        for _ in range(engine.MAX_NO_HAND_FRAMES_FOR_CIRCLE):
            engine.update([], (1280, 720), now)
            now += 1 / 30

        self.assertEqual(len(engine.index_history), history_before_gap)

    def test_circle_history_resets_after_long_tracking_drop(self) -> None:
        engine = GestureEngine()
        hand = make_hand(index_xy=(220, 180), thumb_xy=(270, 180))
        now = 0.0

        for _ in range(12):
            engine.update([hand], (1280, 720), now)
            now += 1 / 30
        self.assertGreater(len(engine.index_history), 0)

        for _ in range(engine.MAX_NO_HAND_FRAMES_FOR_CIRCLE + 1):
            engine.update([], (1280, 720), now)
            now += 1 / 30

        self.assertEqual(len(engine.index_history), 0)

    def test_rotation_dead_zone_and_activation(self) -> None:
        engine = GestureEngine()
        now = 0.0

        # Open hand, no pinch.
        hand_a = make_hand(index_xy=(150, 140), thumb_xy=(260, 140), palm_xy=(140, 210))
        hand_a.landmarks_px[12] = (150, 120)
        state = engine.update([hand_a], (1280, 720), now)
        self.assertEqual(state.name, "IDLE")
        now += 1 / 30

        # Small angle delta stays in dead-zone.
        hand_b = make_hand(index_xy=(150, 140), thumb_xy=(260, 140), palm_xy=(140, 210))
        hand_b.landmarks_px[12] = (162, 120)
        state = engine.update([hand_b], (1280, 720), now)
        self.assertEqual(state.name, "IDLE")
        now += 1 / 30

        # Larger delta should trigger rotation.
        hand_c = make_hand(index_xy=(150, 140), thumb_xy=(260, 140), palm_xy=(140, 210))
        hand_c.landmarks_px[12] = (220, 120)
        state = engine.update([hand_c], (1280, 720), now)
        self.assertEqual(state.name, "ROTATE")
        self.assertNotEqual(state.rotate_delta, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
