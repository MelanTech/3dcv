"""YOLOv11 RGBD 后端解析：独立于 RGB YOLOv11 detector。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from core.utils.platform import is_orangepi


SUPPORTED_BACKENDS = {"auto", "acl", "onnx"}


def resolve_yolov11_rgbd_backend(config: Dict) -> Tuple[str, Path]:
    """确定 RGBD detector 推理后端并拼出模型路径。

    后端优先级：环境变量 ``3DCV_YOLO_RGBD_BACKEND`` > 配置 ``backend`` > ``auto``。
    ``auto`` 时香橙派/昇腾平台使用 ACL ``.om``，其它平台使用 ONNX ``.onnx``。
    """
    configured = os.environ.get(
        "3DCV_YOLO_RGBD_BACKEND",
        config.get("backend", "auto"),
    ).strip().lower()
    if configured not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(
            f"unsupported YOLO RGBD backend: {configured}; supported: {supported}"
        )

    backend = configured
    if backend == "auto":
        backend = "acl" if is_orangepi() else "onnx"

    weights = Path(config["weights"])
    if weights.suffix:
        raise ValueError(
            "detector.weights must not include a file extension; "
            "the YOLO RGBD backend appends .om or .onnx automatically"
        )

    suffix = ".om" if backend == "acl" else ".onnx"
    model_path = weights.with_suffix(suffix)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO RGBD {backend} model does not exist: {model_path}"
        )
    return backend, model_path

