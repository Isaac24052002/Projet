"""Unit tests for camera opening fallback logic."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

from config import AppConfig

if "mediapipe" not in sys.modules:
    sys.modules["mediapipe"] = types.SimpleNamespace()

main = importlib.import_module("main")


class _FakeCapture:
    def __init__(self, opened: bool) -> None:
        self._opened = opened
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True


class OpenCameraTests(unittest.TestCase):
    def _config(self) -> AppConfig:
        return AppConfig(
            width=1280,
            height=720,
            target_fps=30,
            detection_width=640,
            show_debug_landmarks=False,
            pinch_alpha=0.4,
            scale_alpha=0.55,
            rotate_alpha=0.65,
            low_fps_threshold=24.0,
            high_fps_threshold=32.0,
            auto_tune_interval_s=1.0,
            camera_index=0,
        )

    def test_open_camera_fallbacks_to_second_backend(self) -> None:
        first = _FakeCapture(opened=False)
        second = _FakeCapture(opened=True)
        created: list[tuple[int, int]] = []

        def fake_videocapture(index: int, backend: int) -> _FakeCapture:
            created.append((index, backend))
            return first if len(created) == 1 else second

        with patch.object(main.os, "name", "posix"), patch.object(main.cv2, "VideoCapture", side_effect=fake_videocapture):
            cap = main.open_camera(self._config())

        self.assertIs(cap, second)
        self.assertTrue(first.released)
        self.assertEqual(len(created), 2)

    def test_open_camera_raises_when_no_backend_works(self) -> None:
        failed = _FakeCapture(opened=False)

        def fake_videocapture(index: int, backend: int) -> _FakeCapture:
            del index, backend
            return failed

        with patch.object(main.os, "name", "posix"), patch.object(main.cv2, "VideoCapture", side_effect=fake_videocapture):
            with self.assertRaises(RuntimeError):
                main.open_camera(self._config())


if __name__ == "__main__":
    unittest.main()
