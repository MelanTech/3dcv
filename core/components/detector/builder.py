"""检测器工厂：根据 type 选择隔离的检测器实现。"""
from __future__ import annotations

from typing import Dict, Optional

from core.components.detector.base import BaseDetector


def build_detector(config: dict, _round_name: str, class_registry: Optional[Dict] = None) -> BaseDetector:
    """按 config['type'] 构建检测器。"""
    detector_type = config["type"]

    if detector_type == "yolov11":
        if class_registry is None:
            raise ValueError(f"{detector_type} requires class_registry config")

        from core.components.detector.yolov11.detector import build_yolov11_detector

        return build_yolov11_detector(config, class_registry)

    if detector_type == "yolov11_rgbd":
        if class_registry is None:
            raise ValueError(f"{detector_type} requires class_registry config")

        from core.components.detector.yolov11_rgbd.detector import build_yolov11_rgbd_detector

        return build_yolov11_rgbd_detector(config, class_registry)

    raise ValueError(f"unsupported detector type: {detector_type}")
