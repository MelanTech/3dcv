"""计数器抽象接口：把逐帧检测汇聚成稳定的每类目标数量。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from core.types import Detection


class BaseCounter(ABC):
    """计数器基类：接收检测结果、维护内部状态并输出最终计数。"""

    @abstractmethod
    def update(self, detections: List[Detection]) -> None:
        """用当前帧的检测结果更新内部统计。"""
        raise NotImplementedError

    @abstractmethod
    def get_counts(self) -> Dict[str, int]:
        """返回当前每个类别的最终计数。"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """清空内部状态（切换桌位/时间窗时调用）。"""
        raise NotImplementedError
