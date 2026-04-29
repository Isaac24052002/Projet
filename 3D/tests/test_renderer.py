"""Unit tests for renderer helper layout logic."""

from __future__ import annotations

import unittest

from renderer import compute_hud_layout


class RendererLayoutTests(unittest.TestCase):
    def test_hud_layout_stays_inside_small_frame(self) -> None:
        x1, y1, x2, y2, debug_x = compute_hud_layout(320, 240)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)
        self.assertLessEqual(x2, 320)
        self.assertLessEqual(y2, 240)
        self.assertGreaterEqual(debug_x, 0)

    def test_hud_layout_caps_panel_width_on_large_frame(self) -> None:
        x1, y1, x2, y2, debug_x = compute_hud_layout(1920, 1080)
        self.assertEqual((x1, y1), (14, 14))
        self.assertEqual(x2, 760)
        self.assertEqual(y2, 154)
        self.assertEqual(debug_x, 1690)


if __name__ == "__main__":
    unittest.main()
