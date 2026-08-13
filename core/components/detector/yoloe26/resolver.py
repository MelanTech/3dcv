"""YOLOE-26 backend resolver for fixed-prompt .onnx/.om models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from core.utils.platform import is_orangepi


SUPPORTED_BACKENDS = {"auto", "onnx", "acl"}


def resolve_yoloe26_backend(config: Dict) -> Tuple[str, Path]:
    """Resolve backend and model path for fixed-prompt YOLOE-26 weights.

    ``weights`` may be either an extension-less prefix, in which case ``.onnx``
    or ``.om`` is appended by backend, or a concrete ``.onnx``/``.om`` path.
    """
    configured = os.environ.get(
        "3DCV_YOLOE_BACKEND",
        config.get("backend", "auto"),
    ).strip().lower()
    if configured not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"unsupported YOLOE backend: {configured}; supported: {supported}")

    weights = Path(config["weights"]).expanduser()
    if weights.suffix in (".onnx", ".om"):
        backend = "acl" if weights.suffix == ".om" else "onnx"
        if configured != "auto" and configured != backend:
            raise ValueError(
                f"YOLOE backend={configured} does not match weight suffix {weights.suffix}"
            )
        model_path = weights
    else:
        backend = configured
        if backend == "auto":
            backend = "acl" if is_orangepi() else "onnx"
        model_path = weights.with_suffix(".om" if backend == "acl" else ".onnx")

    if not model_path.is_file():
        raise FileNotFoundError(f"YOLOE {backend} model does not exist: {model_path}")
    return backend, model_path
