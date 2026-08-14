"""PaddleOCR 模型路径与后端选择解析。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from core.utils.platform import is_orangepi


SUPPORTED_BACKENDS = {"auto", "acl", "onnx"}
def resolve_paddleocr_backend(config: Dict) -> str:
    """解析 PaddleOCR 推理后端。

    优先级：环境变量 ``3DCV_OCR_BACKEND`` > 配置 ``engine.backend`` > ``auto``。
    ``auto`` 时香橙派使用 ACL（.om），其它平台使用 ONNX（.onnx）。
    """
    configured = os.environ.get(
        "3DCV_OCR_BACKEND",
        config.get("backend", "auto"),
    ).strip().lower()
    if configured not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"unsupported PaddleOCR backend: {configured}; supported: {supported}")
    if configured == "auto":
        return "acl" if is_orangepi() else "onnx"
    return configured


def resolve_model_path(model_path: str, backend: str) -> Path:
    """按后端把模型路径解析成真实文件。

    兼容两种写法：
    - ``models/ppocrv5/det``：自动追加 ``.onnx`` 或 ``.om``；
    - ``models/ppocrv5/det.onnx``：按后端替换成对应后缀。
    """
    suffix = ".om" if backend == "acl" else ".onnx"
    path = Path(model_path).expanduser().with_suffix(suffix).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PaddleOCR {backend} model does not exist: {path}")
    return path


def resolve_engine_config(config: Dict) -> Tuple[str, Dict]:
    """返回解析后的后端名和引擎配置副本。"""
    resolved = dict(config)
    backend = resolve_paddleocr_backend(resolved)
    resolved["backend"] = backend
    for key in ("det_model_dir", "rec_model_dir", "cls_model_dir"):
        if key in resolved:
            resolved[key] = str(resolve_model_path(str(resolved[key]), backend))
    if "rec_char_dict_path" in resolved:
        dict_path = Path(str(resolved["rec_char_dict_path"])).expanduser().resolve()
        if not dict_path.is_file():
            raise FileNotFoundError(f"PaddleOCR character dict does not exist: {dict_path}")
        resolved["rec_char_dict_path"] = str(dict_path)
    return backend, resolved
