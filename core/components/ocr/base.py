"""OCR 抽象接口：对检测结果做文字识别，产出额外的识别项。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.types import Detection, Frame


class BaseOcr(ABC):
    """OCR 基类：基于帧和已有检测，返回新增的（如文字类别）检测。"""

    @abstractmethod
    def process(self, frame: Frame, detections: List[Detection], table: int) -> List[Detection]:
        """执行 OCR，返回额外的检测列表（可为空）。"""
        raise NotImplementedError
