"""检测器抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.types import Detection, Frame


class BaseDetector(ABC):
    """检测器基类：对单帧做目标检测，输出图像坐标下的检测框。"""

    @abstractmethod
    def infer(self, frame: Frame, table: int) -> List[Detection]:
        """对一帧图像做推理，返回检测结果列表。"""
        raise NotImplementedError
