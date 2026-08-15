"""DEIM backend resolver for .onnx/.om models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from core.utils.platform import is_orangepi


SUPPORTED_BACKENDS = {"auto", "onnx", "acl"}


def resolve_deim_backend(config: Dict) -> Tuple[str, Path]:
    """Resolve backend and model path from an extension-less or concrete weight path."""
    configured = os.environ.get(
        "3DCV_DEIM_BACKEND",
        config.get("backend", "auto"),
    ).strip().lower()
    if configured not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(
            f"unsupported DEIM backend: {configured}; supported: {supported}"
        )

    weights = Path(config["weights"]).expanduser()
    if weights.suffix in {".onnx", ".om"}:
        backend = "acl" if weights.suffix == ".om" else "onnx"
        if configured not in {"auto", backend}:
            raise ValueError(
                f"DEIM backend={configured} does not match weight suffix "
                f"{weights.suffix}"
            )
        model_path = weights
    else:
        backend = configured
        if backend == "auto":
            backend = "acl" if is_orangepi() else "onnx"
        model_path = weights.with_suffix(".om" if backend == "acl" else ".onnx")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"DEIM {backend} model does not exist: {model_path}"
        )
    return backend, model_path
