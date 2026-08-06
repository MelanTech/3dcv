"""桌面定位器抽象接口：在检测流中稳定地锁定桌面区域。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.types import Detection


class BaseTableLocator(ABC):
    """桌面定位器基类：跟踪桌子候选并对外暴露定位/稳定/锁定状态。"""

    @property
    @abstractmethod
    def is_localized(self) -> bool:
        """是否已确定桌面区域。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_stable(self) -> bool:
        """当前桌面框是否已稳定。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_acquired(self) -> bool:
        """是否已成功锁定桌面（可开始正式识别）。"""
        raise NotImplementedError

    @abstractmethod
    def handle_acquire_timeout(self, reason: str) -> None:
        """锁定超时的处理（例如回退到默认桌面框）。"""
        raise NotImplementedError

    @abstractmethod
    def process(self, detections: List[Detection], table: int) -> List[Detection]:
        """更新桌面跟踪状态，并返回补充/修正后的检测。"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """清空定位状态。"""
        raise NotImplementedError
