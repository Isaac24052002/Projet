"""3D neural graph object with transforms and perspective projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class RenderNode:
    """Projected node ready for drawing."""

    x: int
    y: int
    radius: int
    color: Tuple[int, int, int]  # BGR


@dataclass
class RenderEdge:
    """Projected edge ready for drawing."""

    p1: Tuple[int, int]
    p2: Tuple[int, int]
    color: Tuple[int, int, int]  # BGR
    thickness: int


class NeuralNetObject:
    """Pseudo-3D animated neural network manipulated by gestures."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.layer_sizes: Sequence[int] = (4, 6, 6, 3)
        self.depth_distance = 3.0
        self.base_projection_scale = min(frame_width, frame_height) * 0.23

        self.position = np.array([frame_width * 0.5, frame_height * 0.5], dtype=np.float32)
        self.scale = 1.0
        self.rotation = np.array([0.0, 0.0], dtype=np.float32)  # [pitch_x, yaw_y] in degrees

        self._rng = np.random.default_rng(42)
        (
            self.nodes_3d,
            self.node_layers,
            self.node_radii,
            self.node_colors,
            self.edges,
            self.edge_weights,
            self.edge_phases,
            self.edge_colors,
        ) = self._build_graph()

    def _build_graph(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Tuple[int, int]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        layer_palette = np.array(
            [
                [255, 130, 40],  # blue-ish
                [215, 90, 180],  # violet
                [90, 145, 255],  # coral
                [70, 210, 255],  # amber
            ],
            dtype=np.float32,
        )

        nodes: List[np.ndarray] = []
        layers: List[int] = []
        radii: List[float] = []
        colors: List[np.ndarray] = []
        layer_indices: List[List[int]] = []

        node_index = 0
        for layer_id, size in enumerate(self.layer_sizes):
            x = np.interp(layer_id, [0, len(self.layer_sizes) - 1], [-0.95, 0.95])
            ys = np.linspace(-0.82, 0.82, size, dtype=np.float32)
            current_layer_indices: List[int] = []
            for y in ys:
                z = float(self._rng.uniform(-0.18, 0.18))
                jitter_y = float(self._rng.uniform(-0.05, 0.05))
                nodes.append(np.array([x, y + jitter_y, z], dtype=np.float32))
                layers.append(layer_id)
                radii.append(float(self._rng.uniform(8.0, 12.0)))
                colors.append(layer_palette[layer_id].copy())
                current_layer_indices.append(node_index)
                node_index += 1
            layer_indices.append(current_layer_indices)

        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        phases: List[float] = []
        edge_colors: List[np.ndarray] = []
        for li in range(len(layer_indices) - 1):
            for src in layer_indices[li]:
                for dst in layer_indices[li + 1]:
                    edges.append((src, dst))
                    weight = float(self._rng.uniform(0.35, 1.0))
                    weights.append(weight)
                    phases.append(float(self._rng.uniform(0.0, 2.0 * np.pi)))
                    edge_colors.append((colors[src] + colors[dst]) * 0.5)

        return (
            np.array(nodes, dtype=np.float32),
            np.array(layers, dtype=np.int32),
            np.array(radii, dtype=np.float32),
            np.array(colors, dtype=np.float32),
            edges,
            np.array(weights, dtype=np.float32),
            np.array(phases, dtype=np.float32),
            np.array(edge_colors, dtype=np.float32),
        )

    def set_position(self, x: float, y: float) -> None:
        """Move object center in screen space."""
        self.position[0] = float(np.clip(x, 0.0, self.frame_width - 1.0))
        self.position[1] = float(np.clip(y, 0.0, self.frame_height - 1.0))

    def reset_transform(self) -> None:
        """Reset object transform to its initial neutral state."""
        self.position[:] = (self.frame_width * 0.5, self.frame_height * 0.5)
        self.scale = 1.0
        self.rotation[:] = 0.0

    def apply_scale_delta(self, scale_delta: float) -> None:
        """Apply incremental scaling from two-hands gesture."""
        self.scale = float(np.clip(self.scale * scale_delta, 0.35, 3.5))

    def apply_rotation_delta(self, yaw_delta: float, pitch_delta: float) -> None:
        """Apply incremental rotation from one-hand open gesture."""
        self.rotation[1] += float(yaw_delta)
        self.rotation[0] += float(pitch_delta)
        self.rotation = np.clip(self.rotation, -180.0, 180.0)

    def get_state_text(self) -> str:
        """State string for HUD."""
        x, y = int(self.position[0]), int(self.position[1])
        pitch, yaw = float(self.rotation[0]), float(self.rotation[1])
        return f"OBJ pos=({x},{y}) scale={self.scale:.2f} rot=({pitch:.1f},{yaw:.1f})"

    def build_render_primitives(self, t_seconds: float) -> Tuple[List[RenderNode], List[RenderEdge]]:
        """Project transformed 3D graph into drawable 2D nodes and edges."""
        points = self.nodes_3d * self.scale
        points = points @ self._rotation_matrix().T

        z = points[:, 2]
        denom = np.clip(z + self.depth_distance, 0.25, None)
        fac = self.depth_distance / denom

        projected = np.empty((points.shape[0], 2), dtype=np.float32)
        projected[:, 0] = points[:, 0] * fac * self.base_projection_scale + self.position[0]
        projected[:, 1] = points[:, 1] * fac * self.base_projection_scale + self.position[1]

        node_radii = np.clip(self.node_radii * self.scale * fac, 3.0, 24.0)

        nodes: List[RenderNode] = []
        for idx in range(projected.shape[0]):
            px = int(projected[idx, 0])
            py = int(projected[idx, 1])
            color = tuple(int(c) for c in np.clip(self.node_colors[idx], 0, 255))
            nodes.append(RenderNode(x=px, y=py, radius=int(node_radii[idx]), color=color))

        pulse = 0.5 + 0.5 * np.sin(3.0 * t_seconds + self.edge_phases)
        edge_alpha = 0.25 + 0.75 * pulse

        edges: List[RenderEdge] = []
        for edge_idx, (src, dst) in enumerate(self.edges):
            p1 = (int(projected[src, 0]), int(projected[src, 1]))
            p2 = (int(projected[dst, 0]), int(projected[dst, 1]))
            depth_factor = float((fac[src] + fac[dst]) * 0.5)
            thickness = int(
                np.clip((0.8 + self.edge_weights[edge_idx] * 2.2) * self.scale * depth_factor, 1.0, 5.0)
            )
            color = np.clip(self.edge_colors[edge_idx] * edge_alpha[edge_idx], 0.0, 255.0)
            edges.append(
                RenderEdge(
                    p1=p1,
                    p2=p2,
                    color=(int(color[0]), int(color[1]), int(color[2])),
                    thickness=thickness,
                )
            )

        return nodes, edges

    def _rotation_matrix(self) -> np.ndarray:
        pitch = np.radians(float(self.rotation[0]))
        yaw = np.radians(float(self.rotation[1]))

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch), -np.sin(pitch)],
                [0.0, np.sin(pitch), np.cos(pitch)],
            ],
            dtype=np.float32,
        )
        ry = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float32,
        )
        return ry @ rx
