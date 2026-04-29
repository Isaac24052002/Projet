"""OpenCV renderer for Neural Gesture Sculptor."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from neural_object import NeuralNetObject


class Renderer:
    """Composes webcam frame, neural object overlay and HUD."""

    def __init__(self, window_name: str = "Neural Gesture Sculptor") -> None:
        self.window_name = window_name
        self.object_alpha = 0.85
        self._object_layer: Optional[np.ndarray] = None

    def setup_window(self, width: int, height: int, fullscreen: bool = False) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, width, height)

    def render(
        self,
        frame: np.ndarray,
        neural_object: Optional[NeuralNetObject],
        gesture_name: str,
        fps: float,
        timestamp_s: float,
        hands_count: int,
        backend_name: str,
        detector_width: int,
        target_fps: int,
        debug_landmarks: bool,
    ) -> np.ndarray:
        """Return fully annotated frame."""
        output = frame

        if neural_object is not None:
            object_layer = self._ensure_object_layer(frame)
            nodes, edges = neural_object.build_render_primitives(timestamp_s)
            for edge in edges:
                cv2.line(
                    object_layer,
                    edge.p1,
                    edge.p2,
                    edge.color,
                    edge.thickness,
                    lineType=cv2.LINE_AA,
                )
            for node in nodes:
                cv2.circle(
                    object_layer,
                    (node.x, node.y),
                    node.radius,
                    node.color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            output = cv2.addWeighted(output, 1.0, object_layer, self.object_alpha, 0.0)

        self._draw_hud(
            output,
            gesture_name=gesture_name,
            fps=fps,
            object_state=(neural_object.get_state_text() if neural_object is not None else "OBJ none"),
            hands_count=hands_count,
            backend_name=backend_name,
            detector_width=detector_width,
            target_fps=target_fps,
            debug_landmarks=debug_landmarks,
        )
        return output

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def _ensure_object_layer(self, frame: np.ndarray) -> np.ndarray:
        """Reuse the same overlay buffer to avoid per-frame allocations."""
        if self._object_layer is None or self._object_layer.shape != frame.shape:
            self._object_layer = np.zeros_like(frame)
        else:
            self._object_layer.fill(0)
        return self._object_layer

    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        gesture_name: str,
        fps: float,
        object_state: str,
        hands_count: int,
        backend_name: str,
        detector_width: int,
        target_fps: int,
        debug_landmarks: bool,
    ) -> None:
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2, debug_x = compute_hud_layout(frame_w, frame_h)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (18, 18, 18), thickness=-1)
            cv2.addWeighted(overlay, 0.45, roi, 0.55, 0.0, dst=roi)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Gesture: {gesture_name}", (28, 44), font, 0.8, (90, 235, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:5.1f}", (28, 74), font, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Hands: {hands_count}", (188, 74), font, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, object_state, (28, 104), font, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Backend: {backend_name} | Detector: {detector_width}px | Target FPS: {target_fps}",
            (28, 132),
            font,
            0.55,
            (180, 215, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Debug points: {'ON' if debug_landmarks else 'OFF'}",
            (debug_x, 74),
            font,
            0.58,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        help_text = (
            "Circle=create | C=spawn test | X=clear | D=debug pts | Pinch=move | "
            "2 hands=scale | Open hand=rotate | R=reset | Q/Esc=quit"
        )
        cv2.putText(
            frame,
            help_text,
            (20, frame_h - 18),
            font,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def compute_hud_layout(frame_w: int, frame_h: int) -> tuple[int, int, int, int, int]:
    """Compute adaptive HUD coordinates from frame dimensions."""
    panel_margin = 14
    x1, y1 = panel_margin, panel_margin
    x2 = min(frame_w - panel_margin, 760)
    y2 = min(frame_h - panel_margin, 154)
    debug_x = max(28, frame_w - 230)
    return x1, y1, x2, y2, debug_x
