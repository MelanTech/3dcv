"""YOLO 输出解码器抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from core.types import Detection


class BaseYoloDecoder(ABC):
    """解码器基类：把网络原始输出还原成原图坐标下的检测框。"""

    @abstractmethod
    def decode(
        self,
        outputs,
        pad: Tuple[int, int],
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
        conf_thresh: float,
        nms_thresh: float,
        detector_id_to_class: Dict[int, str],
        table: int,
        backend_name: str,
    ) -> List[Detection]:
        raise NotImplementedError
