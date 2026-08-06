"""ONNX Runtime 推理后端（通用平台，默认 CPU）。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from core.infra.inference.backend.base import BaseInferenceBackend


class OnnxBackend(BaseInferenceBackend):
    """用 onnxruntime 运行 .onnx 模型的推理后端。"""

    name = "onnx"

    def __init__(self, model_path: Path, config: Dict):
        from onnxruntime import (
            ExecutionMode,
            GraphOptimizationLevel,
            InferenceSession,
            SessionOptions,
            get_available_providers,
        )

        sess_options = SessionOptions()
        sess_options.graph_optimization_level = self._graph_optimization_level(
            GraphOptimizationLevel,
            config.get("graph_optimization_level", "ORT_ENABLE_ALL"),
        )
        if "intra_op_num_threads" in config:
            sess_options.intra_op_num_threads = int(config["intra_op_num_threads"])
        if "inter_op_num_threads" in config:
            sess_options.inter_op_num_threads = int(config["inter_op_num_threads"])
        if "execution_mode" in config:
            sess_options.execution_mode = self._execution_mode(
                ExecutionMode,
                config["execution_mode"],
            )
        if "enable_mem_pattern" in config:
            sess_options.enable_mem_pattern = bool(config["enable_mem_pattern"])

        providers = self._select_providers(
            requested=config.get("providers", ["CPUExecutionProvider"]),
            available=get_available_providers(),
        )
        self.session = InferenceSession(
            str(model_path),
            providers=providers,
            sess_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name

    def execute(self, data: np.ndarray) -> List[np.ndarray]:
        return self.session.run(None, {self.input_name: data})

    def close(self) -> None:
        self.session = None

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    @staticmethod
    def _graph_optimization_level(graph_optimization_level, value: str):
        levels = {
            "ORT_DISABLE_ALL": graph_optimization_level.ORT_DISABLE_ALL,
            "ORT_ENABLE_BASIC": graph_optimization_level.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": graph_optimization_level.ORT_ENABLE_EXTENDED,
            "ORT_ENABLE_ALL": graph_optimization_level.ORT_ENABLE_ALL,
        }
        key = str(value).strip().upper()
        if key not in levels:
            supported = ", ".join(levels)
            raise ValueError(f"unsupported ONNX graph_optimization_level: {value}; supported: {supported}")
        return levels[key]

    @staticmethod
    def _execution_mode(execution_mode, value: str):
        modes = {
            "ORT_SEQUENTIAL": execution_mode.ORT_SEQUENTIAL,
            "ORT_PARALLEL": execution_mode.ORT_PARALLEL,
        }
        key = str(value).strip().upper()
        if key not in modes:
            supported = ", ".join(modes)
            raise ValueError(f"unsupported ONNX execution_mode: {value}; supported: {supported}")
        return modes[key]

    @staticmethod
    def _select_providers(requested, available) -> List[str]:
        selected = [provider for provider in requested if provider in available]
        missing = [provider for provider in requested if provider not in available]
        if missing:
            print(
                "[WARN] ONNX providers unavailable and skipped: "
                + ", ".join(missing)
            )
        if selected:
            return selected
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        if available:
            return [available[0]]
        raise RuntimeError("no ONNX Runtime execution providers are available")
