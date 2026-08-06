"""YOLO 检测器装配：拼装 推理后端 + 输出解码器 + 检测器外壳。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from core.components.detector.base import BaseDetector
from core.components.detector.yolo.decoder.base import BaseYoloDecoder
from core.components.detector.yolo.resolver import resolve_yolo_backend
from core.infra.inference.backend.base import BaseInferenceBackend


def build_yolo_detector(config: dict, class_registry: Dict) -> BaseDetector:
    """根据配置解析后端与权重路径，组装出完整的 YOLO 检测器。"""
    from core.components.detector.yolo.detector import YoloDetector

    backend_name, model_path = resolve_yolo_backend(config)
    backend = _build_yolo_backend(backend_name, model_path, config)
    decoder = _build_yolo_decoder(config["type"])
    return YoloDetector(
        config=config,
        class_registry=class_registry,
        backend=backend,
        decoder=decoder,
    )


def _build_yolo_backend(
    backend_name: str,
    model_path: Path,
    config: dict,
) -> BaseInferenceBackend:
    """按解析出的后端名（onnx / acl）创建对应推理后端。"""
    if backend_name == "onnx":
        from core.infra.inference.backend.onnx import OnnxBackend

        return OnnxBackend(model_path, config)

    if backend_name == "acl":
        from core.infra.inference.backend.acl import AclBackend

        return AclBackend(model_path, config)

    raise ValueError(f"unsupported resolved YOLO backend: {backend_name}")


def _build_yolo_decoder(detector_type: str) -> BaseYoloDecoder:
    """按检测器类型创建对应的输出解码器（把原始张量解析成检测框）。"""
    if detector_type == "yolov11":
        from core.components.detector.yolo.decoder.yolov11 import YoloV11Decoder

        return YoloV11Decoder()

    raise ValueError(f"unsupported YOLO type: {detector_type}")
