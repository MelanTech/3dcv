"""空可视化器：不做任何渲染，用于关闭可视化时保持接口一致。"""
from __future__ import annotations

from typing import List, Optional

from core.types import Detection, Frame
from core.infra.visualization.base import BaseVisualizer


class NoopVisualizer(BaseVisualizer):
    """什么都不做的可视化器（可视化未开启时使用）。"""

    def render(
        self,
        _frame: Frame,
        _detections: List[Detection],
        _table: int,
        stage: str = "final",
        state_name: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        return

    def close(self) -> None:
        return
