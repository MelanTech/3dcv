"""过滤器抽象接口：借助深度信息剔除桌面区域外的检测。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.types import Detection, Frame


class BaseFilter(ABC):
    """过滤器基类：结合帧的深度图对检测结果做空间过滤。"""

    @abstractmethod
    def process(self, detections: List[Detection], frame: Frame, table: int) -> List[Detection]:
        """过滤检测结果，返回保留下来的检测。"""
        raise NotImplementedError
