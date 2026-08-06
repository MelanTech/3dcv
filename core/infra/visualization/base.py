"""可视化器抽象接口：把帧和检测结果渲染出来用于调试。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from core.types import Detection, Frame


class BaseVisualizer(ABC):
    """可视化器基类：按处理阶段渲染帧与检测框。"""

    @abstractmethod
    def render(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
        stage: str = "final",
        state_name: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        """渲染某一处理阶段的画面。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """关闭可视化窗口/资源。"""
        raise NotImplementedError
