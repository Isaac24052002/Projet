"""Entry point for Neural Gesture Sculptor."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import cv2

# Avoid matplotlib cache warnings triggered by MediaPipe imports in locked dirs.
_MPL_CACHE_DIR = Path(".mplconfig")
_MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR.resolve()))

import mediapipe as mp
import numpy as np

from config import AppConfig, DETECTION_WIDTH_MAX, DETECTION_WIDTH_MIN
from gesture_engine import GestureEngine, GestureState, HandLandmarks
from neural_object import NeuralNetObject
from renderer import Renderer


TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class RuntimeState:
    """Mutable state used while the capture loop is running."""

    fps: float
    last_time_s: float
    last_timestamp_ms: int
    detector_width: int
    last_perf_tune_s: float
    show_debug_landmarks: bool
    dragging: bool
    drag_offset: np.ndarray


class HandDetector(Protocol):
    """Common protocol implemented by both MediaPipe backends."""

    backend_name: str

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> "HandDetector":
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        ...


def extract_hands_legacy(
    mp_results: object,
    frame_width: int,
    frame_height: int,
) -> List[HandLandmarks]:
    """Convert legacy MediaPipe Solutions result objects into typed HandLandmarks."""
    hands: List[HandLandmarks] = []
    hand_landmarks_list = getattr(mp_results, "multi_hand_landmarks", None)
    handedness_list = getattr(mp_results, "multi_handedness", None)
    if hand_landmarks_list is None:
        return hands

    for idx, hand_lm in enumerate(hand_landmarks_list):
        label = "Unknown"
        if handedness_list and idx < len(handedness_list):
            label = handedness_list[idx].classification[0].label

        norm = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
        px = np.stack(
            [norm[:, 0] * frame_width, norm[:, 1] * frame_height],
            axis=1,
        ).astype(np.float32)
        hands.append(HandLandmarks(label=label, landmarks_norm=norm, landmarks_px=px))
    return hands


def extract_hands_tasks(
    mp_result: object,
    frame_width: int,
    frame_height: int,
) -> List[HandLandmarks]:
    """Convert MediaPipe Tasks hand-landmarker result into typed HandLandmarks."""
    hands: List[HandLandmarks] = []

    hand_landmarks_list = getattr(mp_result, "hand_landmarks", None) or []
    handedness_list = getattr(mp_result, "handedness", None) or []
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        label = "Unknown"
        if idx < len(handedness_list) and handedness_list[idx]:
            label = handedness_list[idx][0].category_name or "Unknown"

        norm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
        px = np.stack(
            [norm[:, 0] * frame_width, norm[:, 1] * frame_height],
            axis=1,
        ).astype(np.float32)
        hands.append(HandLandmarks(label=label, landmarks_norm=norm, landmarks_px=px))
    return hands


def _resolve_tasks_model_path() -> Path:
    """Resolve hand-landmarker model path for MediaPipe Tasks backend."""
    env_override = os.getenv("MP_HAND_LANDMARKER_MODEL")
    candidates: List[Path] = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.extend(
        [
            Path("models/hand_landmarker.task"),
            Path("hand_landmarker.task"),
            Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "MediaPipe installe ne contient pas 'mp.solutions' et aucun modele Tasks n'a ete trouve.\n"
        "Telecharge le modele puis relance:\n"
        "  mkdir -p models\n"
        "  wget -O models/hand_landmarker.task "
        + TASK_MODEL_URL
        + "\n"
        "Ou definis MP_HAND_LANDMARKER_MODEL=/chemin/vers/hand_landmarker.task"
    )


class _LegacyHandsDetector:
    """Compatibility wrapper around MediaPipe Solutions Hands API."""

    def __init__(self) -> None:
        self.backend_name = "solutions"
        self._mp_hands = mp.solutions.hands
        self._detector = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:
        del timestamp_ms  # Unused with the legacy backend.
        frame_h, frame_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        return extract_hands_legacy(results, frame_w, frame_h)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "_LegacyHandsDetector":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()


class _TasksHandsDetector:
    """Compatibility wrapper around MediaPipe Tasks HandLandmarker API."""

    def __init__(self) -> None:
        self.backend_name = "tasks"
        model_path = _resolve_tasks_model_path()
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:
        frame_h, frame_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        return extract_hands_tasks(result, frame_w, frame_h)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "_TasksHandsDetector":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()


def create_hands_detector() -> HandDetector:
    """Create a hand-detector backend compatible with current MediaPipe build."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return _LegacyHandsDetector()
    return _TasksHandsDetector()


def draw_landmark_debug(frame: np.ndarray, hands: List[HandLandmarks]) -> None:
    """Lightweight hand landmark overlay for easier user feedback."""
    for hand in hands:
        points = hand.landmarks_px.astype(np.int32)
        for idx in range(21):
            cv2.circle(frame, (int(points[idx, 0]), int(points[idx, 1])), 3, (80, 220, 80), -1, cv2.LINE_AA)


def remap_hands_to_frame(
    hands: List[HandLandmarks],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> List[HandLandmarks]:
    """Scale landmark coordinates from detector frame back to display frame."""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        return hands

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    remapped: List[HandLandmarks] = []
    for hand in hands:
        px = hand.landmarks_px.copy()
        px[:, 0] *= sx
        px[:, 1] *= sy
        norm = hand.landmarks_norm.copy()
        norm[:, 0] = np.clip(px[:, 0] / max(dst_w, 1), 0.0, 1.0)
        norm[:, 1] = np.clip(px[:, 1] / max(dst_h, 1), 0.0, 1.0)
        remapped.append(HandLandmarks(label=hand.label, landmarks_norm=norm, landmarks_px=px))
    return remapped


def setup_opencv_threads() -> None:
    """Enable OpenCV optimizations and tune thread count for CPU stability."""
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(2, (os.cpu_count() or 2) // 2))
    except Exception:
        pass


def open_camera(config: AppConfig) -> cv2.VideoCapture:
    """Open and configure webcam capture."""
    backend_candidates = [cv2.CAP_ANY]
    if os.name == "posix" and hasattr(cv2, "CAP_V4L2"):
        backend_candidates = [cv2.CAP_V4L2, cv2.CAP_ANY]

    cap: Optional[cv2.VideoCapture] = None
    for backend in backend_candidates:
        candidate = cv2.VideoCapture(config.camera_index, backend)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()

    if cap is None:
        raise RuntimeError(
            f"Impossible d'ouvrir la webcam index={config.camera_index}. "
            "Teste NGS_CAMERA_INDEX=1 (ou autre index disponible)."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    return cap


def resize_frame_for_detection(frame: np.ndarray, detector_width: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a downscaled frame for faster hand detection."""
    frame_h, frame_w = frame.shape[:2]
    detector_w = min(detector_width, frame_w)
    detector_h = int(frame_h * detector_w / max(frame_w, 1))
    if detector_w != frame_w:
        resized = cv2.resize(frame, (detector_w, detector_h), interpolation=cv2.INTER_LINEAR)
        return resized, (detector_w, detector_h)
    return frame, (frame_w, frame_h)


def update_fps(prev_fps: float, delta_s: float) -> float:
    """Smooth FPS using exponential moving average."""
    if delta_s <= 1e-6:
        return prev_fps
    instantaneous = 1.0 / delta_s
    return 0.9 * prev_fps + 0.1 * instantaneous


def tune_detector_width(
    current_width: int,
    fps: float,
    frame_width: int,
    config: AppConfig,
) -> int:
    """Adapt detector resolution to keep interaction fluid."""
    width = current_width
    if fps < config.low_fps_threshold and width > DETECTION_WIDTH_MIN:
        width -= 64
    elif fps > config.high_fps_threshold and width < min(frame_width, DETECTION_WIDTH_MAX):
        width += 64
    return int(np.clip(width, DETECTION_WIDTH_MIN, min(frame_width, DETECTION_WIDTH_MAX)))


def apply_gesture_to_object(
    neural_object: Optional[NeuralNetObject],
    gesture_state: GestureState,
    runtime: RuntimeState,
    config: AppConfig,
) -> Optional[NeuralNetObject]:
    """Apply gesture output to the neural object transform state."""
    if neural_object is None:
        runtime.dragging = False
        return None

    if gesture_state.pinch_active and gesture_state.pinch_point is not None:
        pinch_point = np.array(gesture_state.pinch_point, dtype=np.float32)
        if not runtime.dragging:
            runtime.drag_offset = neural_object.position - pinch_point
        runtime.dragging = True
        target_pos = pinch_point + runtime.drag_offset
        smooth_pos = neural_object.position + (target_pos - neural_object.position) * config.pinch_alpha
        neural_object.set_position(float(smooth_pos[0]), float(smooth_pos[1]))
    else:
        runtime.dragging = False

    if gesture_state.name == "TWO_HANDS":
        smoothed_scale_delta = 1.0 + (gesture_state.scale_delta - 1.0) * config.scale_alpha
        neural_object.apply_scale_delta(smoothed_scale_delta)

    if gesture_state.name == "ROTATE":
        yaw_delta, pitch_delta = gesture_state.rotate_delta
        neural_object.apply_rotation_delta(yaw_delta * config.rotate_alpha, pitch_delta * config.rotate_alpha)

    return neural_object


def main() -> None:
    config = AppConfig.from_env()
    setup_opencv_threads()
    cap = open_camera(config)
    renderer = Renderer()
    renderer.setup_window(config.width, config.height, fullscreen=False)
    gesture_engine = GestureEngine()

    neural_object: Optional[NeuralNetObject] = None
    now_s = time.monotonic()
    runtime = RuntimeState(
        fps=0.0,
        last_time_s=now_s,
        last_timestamp_ms=0,
        detector_width=min(config.detection_width, config.width),
        last_perf_tune_s=now_s,
        show_debug_landmarks=config.show_debug_landmarks,
        dragging=False,
        drag_offset=np.zeros(2, dtype=np.float32),
    )

    try:
        with create_hands_detector() as hands_detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                now_s = time.monotonic()

                timestamp_ms = int(now_s * 1000.0)
                if timestamp_ms <= runtime.last_timestamp_ms:
                    timestamp_ms = runtime.last_timestamp_ms + 1
                runtime.last_timestamp_ms = timestamp_ms

                detector_frame, detector_size = resize_frame_for_detection(frame, runtime.detector_width)
                detected_hands = hands_detector.process(detector_frame, timestamp_ms)

                if detector_size != (frame_w, frame_h):
                    hands = remap_hands_to_frame(detected_hands, src_size=detector_size, dst_size=(frame_w, frame_h))
                else:
                    hands = detected_hands

                gesture_state = gesture_engine.update(hands, (frame_w, frame_h), now_s)
                if gesture_state.circle_triggered:
                    neural_object = NeuralNetObject(frame_w, frame_h)
                    runtime.dragging = False

                neural_object = apply_gesture_to_object(neural_object, gesture_state, runtime, config)

                if runtime.show_debug_landmarks:
                    draw_landmark_debug(frame, hands)

                frame_delta_s = now_s - runtime.last_time_s
                runtime.fps = update_fps(runtime.fps, frame_delta_s)
                runtime.last_time_s = now_s

                if now_s - runtime.last_perf_tune_s >= config.auto_tune_interval_s:
                    runtime.detector_width = tune_detector_width(
                        runtime.detector_width,
                        runtime.fps,
                        frame_w,
                        config,
                    )
                    runtime.last_perf_tune_s = now_s

                composed = renderer.render(
                    frame=frame,
                    neural_object=neural_object,
                    gesture_name=gesture_state.name,
                    fps=runtime.fps,
                    timestamp_s=now_s,
                    hands_count=gesture_state.hands_count,
                    backend_name=hands_detector.backend_name,
                    detector_width=runtime.detector_width,
                    target_fps=config.target_fps,
                    debug_landmarks=runtime.show_debug_landmarks,
                )
                renderer.show(composed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    neural_object = NeuralNetObject(frame_w, frame_h)
                    runtime.dragging = False
                elif key == ord("x"):
                    neural_object = None
                    runtime.dragging = False
                elif key == ord("d"):
                    runtime.show_debug_landmarks = not runtime.show_debug_landmarks
                elif key == ord("r") and neural_object is not None:
                    neural_object.reset_transform()
                elif key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
