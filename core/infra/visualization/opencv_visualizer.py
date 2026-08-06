"""OpenCV 可视化器：用窗口展示 RGB/深度及检测框，支持多种显示模式。"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.types import Detection, Frame
from core.infra.visualization.base import BaseVisualizer


class OpenCvVisualizer(BaseVisualizer):
    """基于 OpenCV 窗口的调试可视化器；渲染出错会自动禁用以免影响主流程。"""

    SUPPORTED_MODES = {
        "rgb",
        "depth",
        "rgb_depth",
        "rgb_depth_overlay",
    }

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        self.window_name = config.get("window_name", "3DCV Debug View")
        self.mode = config.get("mode", "rgb_depth")
        if self.mode not in self.SUPPORTED_MODES:
            supported = ", ".join(sorted(self.SUPPORTED_MODES))
            raise ValueError(f"unsupported visualization mode: {self.mode}; supported: {supported}")
        self.depth_max_mm = int(config.get("depth_max_mm", 5000))
        self.depth_overlay_alpha = float(config.get("depth_overlay_alpha", 0.45))
        self.wait_key_ms = int(config.get("wait_key_ms", 1))
        self.draw_labels = bool(config.get("draw_labels", True))
        self.draw_status_label = bool(config.get("draw_status_label", True))
        self.draw_state_label = bool(config.get("draw_state_label", True))
        self.overlay_text_scale = float(config.get("overlay_text_scale", 0.45))
        self.overlay_line_height = int(config.get("overlay_line_height", 18))
        self.fps_smoothing = min(0.99, max(0.0, float(config.get("fps_smoothing", 0.8))))
        self.stages = set(config.get("stages", ["preview", "track", "final"]))
        self.stage_windows = len(self.stages) > 1
        self.open_windows = set()
        self._last_frame_key: Optional[Tuple[int, str]] = None
        self._last_frame_time: Optional[float] = None
        self._fps: Optional[float] = None
        self._disabled = False
        self.bbox_colors = self._load_bbox_colors(class_registry or {})

    @staticmethod
    def _load_bbox_colors(class_registry: Dict) -> Dict[str, Tuple[int, int, int]]:
        """读取 RGB bbox 颜色并转换为 OpenCV 使用的 BGR。"""
        colors = {}
        for class_name, rgb in dict(class_registry.get("bbox_colors", {})).items():
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                continue
            red, green, blue = (int(max(0, min(255, value))) for value in rgb)
            colors[str(class_name)] = (blue, green, red)
        return colors

    def render(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
        stage: str = "final",
        state_name: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        """渲染指定阶段画面；只显示配置里开启的阶段，出错则永久禁用可视化。"""
        if self._disabled or frame.rgb is None:
            return
        if stage not in self.stages:
            return

        try:
            fps = self._update_fps(frame, table)
            view = self._build_view(frame, detections, table, fps, state_name, elapsed_sec)
            window_name = self._window_name(stage)
            cv2.imshow(window_name, view)
            self.open_windows.add(window_name)
            cv2.waitKey(self.wait_key_ms)
        except Exception as exc:
            self._disabled = True
            print(f"[WARN] OpenCvVisualizer disabled: {exc}")

    def _window_name(self, stage: str) -> str:
        if not self.stage_windows:
            return self.window_name
        return f"{self.window_name} [{stage}]"

    def _build_view(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
        fps: Optional[float],
        state_name: Optional[str],
        elapsed_sec: Optional[float],
    ) -> np.ndarray:
        """按当前显示模式拼装出最终要展示的画面。"""
        rgb_panel = self._rgb_panel(frame.rgb)
        depth_panel = self._depth_panel(frame.depth, rgb_panel.shape[:2])

        if self.mode == "rgb":
            return self._draw_overlay(
                self._draw_detections(rgb_panel, detections, table),
                table,
                fps,
                state_name,
                elapsed_sec,
            )
        if self.mode == "depth":
            return self._draw_overlay(
                self._draw_detections(depth_panel, detections, table),
                table,
                fps,
                state_name,
                elapsed_sec,
            )
        if self.mode == "rgb_depth":
            rgb_with_detections = self._draw_detections(rgb_panel, detections, table)
            return self._draw_overlay(
                np.hstack((rgb_with_detections, depth_panel)),
                table,
                fps,
                state_name,
                elapsed_sec,
            )
        if self.mode == "rgb_depth_overlay":
            overlay = self._overlay_depth(rgb_panel, depth_panel, frame.depth)
            return self._draw_overlay(
                self._draw_detections(overlay, detections, table),
                table,
                fps,
                state_name,
                elapsed_sec,
            )

        raise AssertionError(f"unhandled visualization mode: {self.mode}")

    def _update_fps(self, frame: Frame, table: int) -> Optional[float]:
        """按唯一帧更新 FPS，避免多阶段渲染同一帧时重复计数。"""
        frame_key = (table, frame.frame_id)
        if frame_key == self._last_frame_key:
            return self._fps

        now = time.perf_counter()
        if self._last_frame_time is not None:
            elapsed = now - self._last_frame_time
            if elapsed > 1e-6:
                instant_fps = 1.0 / elapsed
                if self._fps is None:
                    self._fps = instant_fps
                else:
                    alpha = self.fps_smoothing
                    self._fps = alpha * self._fps + (1.0 - alpha) * instant_fps

        self._last_frame_key = frame_key
        self._last_frame_time = now
        return self._fps

    @staticmethod
    def _rgb_panel(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _depth_panel(self, depth, target_shape: Tuple[int, int]) -> np.ndarray:
        """把深度图归一化并伪彩色化，无效像素置黑，缩放到目标尺寸。"""
        height, width = target_shape
        if depth is None:
            return np.zeros((height, width, 3), dtype=np.uint8)

        valid = depth > 0
        clipped = np.clip(depth, 0, self.depth_max_mm)
        depth_u8 = ((clipped.astype(np.float32) / self.depth_max_mm) * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        depth_color[~valid] = (0, 0, 0)

        if depth_color.shape[:2] != target_shape:
            depth_color = cv2.resize(
                depth_color,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        return depth_color

    def _overlay_depth(self, rgb_panel: np.ndarray, depth_panel: np.ndarray, depth) -> np.ndarray:
        """把伪彩色深度按透明度叠加到 RGB 上，只在有效深度处叠加。"""
        overlay = rgb_panel.copy()
        alpha = min(1.0, max(0.0, self.depth_overlay_alpha))

        if depth is None:
            return overlay

        if depth_panel.shape[:2] != rgb_panel.shape[:2]:
            depth_panel = cv2.resize(
                depth_panel,
                (rgb_panel.shape[1], rgb_panel.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        valid = depth > 0
        if valid.shape != rgb_panel.shape[:2]:
            valid = cv2.resize(
                valid.astype(np.uint8),
                (rgb_panel.shape[1], rgb_panel.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        blended = cv2.addWeighted(rgb_panel, 1.0 - alpha, depth_panel, alpha, 0.0)
        overlay[valid] = blended[valid]
        return overlay

    def _draw_detections(
        self,
        panel: np.ndarray,
        detections: List[Detection],
        table: int,
    ) -> np.ndarray:
        """在画面上画出检测框，并按需标注类别与置信度。"""
        panel = panel.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self.bbox_colors.get(detection.class_name, (0, 255, 0))
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
            if self.draw_labels:
                label = f"T{table} {detection.class_name} {detection.score:.2f}"
                cv2.putText(
                    panel,
                    label,
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return panel

    def _draw_overlay(
        self,
        panel: np.ndarray,
        table: int,
        fps: Optional[float],
        state_name: Optional[str],
        elapsed_sec: Optional[float],
    ) -> np.ndarray:
        lines = []
        if self.draw_status_label:
            fps_text = "--" if fps is None else f"{fps:.1f}"
            elapsed_text = "--" if elapsed_sec is None else f"{elapsed_sec:.1f}s"
            lines.append(f"Time: {elapsed_text}  Table: {table}  FPS: {fps_text}")
        if self.draw_state_label:
            lines.append(f"State: {state_name or '--'}")
        if not lines:
            return panel
        for index, line in enumerate(lines):
            origin = (10, 18 + index * self.overlay_line_height)
            cv2.putText(
                panel,
                line,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.overlay_text_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                line,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.overlay_text_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return panel

    def close(self) -> None:
        """销毁所有已打开的窗口。"""
        if not self._disabled:
            for window_name in list(self.open_windows) or [self.window_name]:
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass
            self.open_windows.clear()
