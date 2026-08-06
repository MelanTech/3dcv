"""可视化器工厂：未启用时返回空实现，否则按 type 创建。"""
from __future__ import annotations

from typing import Dict, Optional

from core.infra.visualization.base import BaseVisualizer
from core.infra.visualization.noop_visualizer import NoopVisualizer


def build_visualizer(config: Optional[Dict], class_registry: Optional[Dict] = None) -> BaseVisualizer:
    """按配置构建可视化器；未配置或未启用时返回 NoopVisualizer。"""
    if not config or not config.get("enabled", False):
        return NoopVisualizer()

    visualizer_type = config["type"]
    if visualizer_type == "opencv":
        from core.infra.visualization.opencv_visualizer import OpenCvVisualizer

        return OpenCvVisualizer(config, class_registry)

    raise ValueError(f"unsupported visualization type: {visualizer_type}")
