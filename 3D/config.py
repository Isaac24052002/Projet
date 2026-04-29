"""Application runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_TARGET_FPS = 30
DEFAULT_DETECTION_WIDTH = 640
DETECTION_WIDTH_MIN = 384
DETECTION_WIDTH_MAX = 960

DEFAULT_SHOW_DEBUG = False
DEFAULT_PINCH_ALPHA = 0.40
DEFAULT_SCALE_ALPHA = 0.55
DEFAULT_ROTATE_ALPHA = 0.65
DEFAULT_LOW_FPS = 24.0
DEFAULT_HIGH_FPS = 32.0
DEFAULT_AUTO_TUNE_INTERVAL_S = 1.0
DEFAULT_CAMERA_INDEX = 0


@dataclass(frozen=True)
class AppConfig:
    """Static runtime config loaded once at startup."""

    width: int
    height: int
    target_fps: int
    detection_width: int
    show_debug_landmarks: bool
    pinch_alpha: float
    scale_alpha: float
    rotate_alpha: float
    low_fps_threshold: float
    high_fps_threshold: float
    auto_tune_interval_s: float
    camera_index: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        width = _read_env_int("NGS_WIDTH", DEFAULT_WIDTH, min_value=320, max_value=3840)
        height = _read_env_int("NGS_HEIGHT", DEFAULT_HEIGHT, min_value=240, max_value=2160)
        target_fps = _read_env_int("NGS_FPS", DEFAULT_TARGET_FPS, min_value=15, max_value=120)
        detection_width = _read_env_int(
            "NGS_DET_WIDTH",
            DEFAULT_DETECTION_WIDTH,
            min_value=DETECTION_WIDTH_MIN,
            max_value=min(width, DETECTION_WIDTH_MAX),
        )

        show_debug = _read_env_bool("NGS_DEBUG_LANDMARKS", DEFAULT_SHOW_DEBUG)
        pinch_alpha = _read_env_float("NGS_PINCH_ALPHA", DEFAULT_PINCH_ALPHA, min_value=0.0, max_value=1.0)
        scale_alpha = _read_env_float("NGS_SCALE_ALPHA", DEFAULT_SCALE_ALPHA, min_value=0.0, max_value=1.0)
        rotate_alpha = _read_env_float("NGS_ROTATE_ALPHA", DEFAULT_ROTATE_ALPHA, min_value=0.0, max_value=1.0)
        low_fps = _read_env_float("NGS_LOW_FPS", DEFAULT_LOW_FPS, min_value=5.0, max_value=119.0)
        high_fps = _read_env_float("NGS_HIGH_FPS", DEFAULT_HIGH_FPS, min_value=6.0, max_value=120.0)
        if high_fps <= low_fps:
            high_fps = low_fps + 2.0
        auto_tune_interval_s = _read_env_float(
            "NGS_AUTO_TUNE_S",
            DEFAULT_AUTO_TUNE_INTERVAL_S,
            min_value=0.2,
            max_value=10.0,
        )
        camera_index = _read_env_int("NGS_CAMERA_INDEX", DEFAULT_CAMERA_INDEX, min_value=0, max_value=10)

        return cls(
            width=width,
            height=height,
            target_fps=target_fps,
            detection_width=detection_width,
            show_debug_landmarks=show_debug,
            pinch_alpha=pinch_alpha,
            scale_alpha=scale_alpha,
            rotate_alpha=rotate_alpha,
            low_fps_threshold=low_fps,
            high_fps_threshold=high_fps,
            auto_tune_interval_s=auto_tune_interval_s,
            camera_index=camera_index,
        )


def _read_env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return int(np.clip(value, min_value, max_value))


def _read_env_float(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return float(np.clip(value, min_value, max_value))


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
