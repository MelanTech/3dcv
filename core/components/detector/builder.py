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

    if detector_type == "yoloe26":
        from core.components.detector.yoloe26.detector import build_yoloe26_detector

        return build_yoloe26_detector(config)

    if detector_type == "yolo26":
        if class_registry is None:
            raise ValueError(f"{detector_type} requires class_registry config")

        from core.components.detector.yolo26.detector import build_yolo26_detector

        return build_yolo26_detector(config, class_registry)

    if detector_type in {"deim", "deimv2"}:
        if class_registry is None:
            raise ValueError(f"{detector_type} requires class_registry config")

        from core.components.detector.deim.detector import build_deim_detector

        return build_deim_detector(config, class_registry)

    if detector_type in {"rf_detr", "rfdetr", "rf-detr"}:
        if class_registry is None:
            raise ValueError(f"{detector_type} requires class_registry config")

        from core.components.detector.rf_detr.detector import (
            build_rf_detr_detector,
        )

        return build_rf_detr_detector(config, class_registry)

    raise ValueError(f"unsupported detector type: {detector_type}")
