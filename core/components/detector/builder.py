"""检测器工厂：根据 type 前缀选择检测器实现。"""
from __future__ import annotations

from typing import Dict, Optional

from core.components.detector.base import BaseDetector


def build_detector(config: dict, _round_name: str, class_registry: Optional[Dict] = None) -> BaseDetector:
    """按 config['type'] 构建检测器；以 "yolo" 开头的走 YOLO 系列工厂。"""
    detector_type = config["type"]

    if detector_type.startswith("yolo"):
        if class_registry is None:
            raise ValueError(f"{detector_type} requires class_registry config")

        from core.components.detector.yolo.factory import build_yolo_detector

        return build_yolo_detector(config, class_registry)

    raise ValueError(f"unsupported detector type: {detector_type}")
