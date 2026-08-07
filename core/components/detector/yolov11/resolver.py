"""YOLO 后端解析：根据平台/配置决定用 acl 还是 onnx，并定位模型文件。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from core.utils.platform import is_orangepi


SUPPORTED_BACKENDS = {"auto", "acl", "onnx"}


def resolve_yolo_backend(config: Dict) -> Tuple[str, Path]:
    """确定推理后端并拼出模型路径。

    后端优先级：环境变量 3DCV_YOLO_BACKEND > 配置 backend > auto；
    auto 时香橙派用 acl（.om），其他平台用 onnx（.onnx）。
    weights 只写不含扩展名的前缀，扩展名由后端自动补齐。
    """
    configured = os.environ.get(
        "3DCV_YOLO_BACKEND",
        config.get("backend", "auto"),
    ).strip().lower()
    if configured not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(
            f"unsupported YOLO backend: {configured}; supported: {supported}"
        )

    backend = configured
    if backend == "auto":
        backend = "acl" if is_orangepi() else "onnx"

    weights = Path(config["weights"])
    if weights.suffix:
        raise ValueError(
            "detector.weights must not include a file extension; "
            "the YOLO backend appends .om or .onnx automatically"
        )

    suffix = ".om" if backend == "acl" else ".onnx"
    model_path = weights.with_suffix(suffix)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO {backend} model does not exist: {model_path}"
        )
    return backend, model_path
