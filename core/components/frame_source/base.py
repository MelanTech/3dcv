"""帧源抽象接口：统一离线图片序列与实时相机的取帧方式。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional

from core.types import Frame


class BaseFrameSource(ABC, Iterator[Frame]):
    """帧源基类：作为可迭代对象，每次产出一帧 RGB/深度数据。"""

    @abstractmethod
    def __iter__(self) -> "BaseFrameSource":
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> Frame:
        """产出下一帧；数据耗尽时抛出 StopIteration。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放底层资源（相机/文件句柄等）。"""
        raise NotImplementedError

    def get_filter_intrinsic(self) -> Optional[Dict[str, float]]:
        """返回供深度过滤器使用的相机内参；离线帧源默认不提供。"""
        return None

    def get_filter_intrinsic_source(self) -> str:
        """返回内参来源说明，便于日志确认当前使用的是运行时还是配置值。"""
        return "config"
