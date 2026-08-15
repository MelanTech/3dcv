"""推理后端抽象接口：屏蔽 ONNX / 昇腾 ACL 等不同推理引擎的差异。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorInfo:
    """推理后端输入/输出张量的最小元信息。"""

    name: str
    shape: List[Any]
    type: str = "tensor(float)"


class BaseInferenceBackend(ABC):
    """推理后端基类：输入预处理张量，输出网络原始张量列表。"""

    name: str

    @abstractmethod
    def execute(
        self,
        data: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
    ) -> List[np.ndarray]:
        """执行一次前向推理，支持单输入或按名称/顺序提供的多输入。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放推理引擎资源。"""
        raise NotImplementedError

    def get_inputs(self) -> List[TensorInfo]:
        """返回输入张量元信息；需要静态 shape 的调用方可使用。"""
        raise NotImplementedError

    def get_outputs(self) -> List[TensorInfo]:
        """返回输出张量元信息；需要静态 shape 的调用方可使用。"""
        raise NotImplementedError
