"""在各流水线组件之间传递的小型值对象（数据结构定义）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Frame:
    """来自帧源的一组时间对齐的 RGB/深度图像。"""

    frame_id: str
    rgb: Any
    depth: Any
    timestamp: float


@dataclass(frozen=True)
class Detection:
    """检测器或后处理阶段输出的目标，坐标为图像像素坐标。"""

    class_name: str
    bbox: tuple[int, int, int, int]
    score: float = 1.0
    class_id: int = -1
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecognitionItem:
    """按裁判结果文件格式组织的最终计数条目。"""

    goal_id: str
    num: int
    table: int
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewValidation:
    """针对相机/桌面视角的可选校验结果。"""

    ok: bool
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)


RecognitionResult = List[RecognitionItem]
