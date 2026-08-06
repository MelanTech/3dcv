"""把项目统一 infra 推理后端适配为 PaddleOCR 原始 session 接口。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from core.infra.inference.backend.base import BaseInferenceBackend


class PaddleOcrInferenceSession:
    """兼容 PaddleOCR 旧代码所需的 ``run/get_inputs/get_outputs`` 接口。"""

    def __init__(self, backend: BaseInferenceBackend):
        self.backend = backend
        self.backend_name = backend.name

    def get_inputs(self):
        return self.backend.get_inputs()

    def get_outputs(self):
        return self.backend.get_outputs()

    def run(self, _output_names: List[str], input_feed: Dict[str, np.ndarray]):
        if not input_feed:
            raise ValueError("PaddleOCR input_feed must not be empty")
        data = next(iter(input_feed.values()))
        data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        outputs = self.backend.execute(data)
        if outputs is None:
            raise RuntimeError("PaddleOCR backend execution failed")
        return outputs

    def close(self) -> None:
        self.backend.close()


def create_paddleocr_session(model_path: str, use_gpu: bool, config: Dict) -> PaddleOcrInferenceSession:
    """按 backend 配置创建统一 infra backend，并包装成 PaddleOCR session。"""
    backend = str(config.get("backend", "onnx")).strip().lower()
    if backend == "onnx":
        from core.infra.inference.backend.onnx import OnnxBackend

        backend_config = dict(config)
        backend_config.setdefault(
            "providers",
            (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_gpu
                else ["CPUExecutionProvider"]
            ),
        )
        return PaddleOcrInferenceSession(OnnxBackend(Path(model_path), backend_config))

    if backend == "acl":
        from core.infra.inference.backend.acl import AclBackend

        return PaddleOcrInferenceSession(AclBackend(Path(model_path), config))

    raise ValueError(f"unsupported PaddleOCR backend: {backend}")
