"""裁判客户端抽象接口：负责结果文件的生成与向裁判盒的通信。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from core.types import RecognitionItem


class BaseRefereeClient(ABC):
    """裁判客户端基类：定义连接、开始/旋转信号、结果写入与上报等接口。"""

    @abstractmethod
    def result_path(self, round_number: int) -> Path:
        """返回该轮次结果文件的落盘路径。"""
        raise NotImplementedError

    @abstractmethod
    def render_result(self, items: Iterable[RecognitionItem]) -> str:
        """把识别结果渲染成裁判要求的文本格式。"""
        raise NotImplementedError

    @abstractmethod
    def write_result(self, round_number: int, items: Iterable[RecognitionItem]) -> Path:
        """把结果写入文件并返回路径。"""
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> bool:
        """连接裁判盒，成功返回 True。"""
        raise NotImplementedError

    @abstractmethod
    def send_start(self) -> bool:
        """发送开始信号。"""
        raise NotImplementedError

    @abstractmethod
    def send_rotate(self) -> bool:
        """发送旋转/换桌信号。"""
        raise NotImplementedError

    @abstractmethod
    def send_result_file(self, path: Path) -> bool:
        """把结果文件内容发送给裁判盒。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """关闭与裁判盒的连接。"""
        raise NotImplementedError
