"""Detector-stage implementations used by the frame pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from core.components.detector.base import BaseDetector
from core.types import Detection, Frame


@dataclass(frozen=True)
class DetectedFrame:
    """A frame whose detector result is ready for downstream processing."""

    frame: Frame
    table: int
    detections: List[Detection]


class BaseDetectorStage(ABC):
    """Detector execution boundary for inline and pipelined implementations."""

    @abstractmethod
    def accept(self, frame: Frame, table: int) -> Optional[DetectedFrame]:
        """Accept one frame and return a ready detector result when available."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> Optional[DetectedFrame]:
        """Return the final pending detector result, if any."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release execution resources."""
        raise NotImplementedError

    def observe_roi(
        self,
        _frame: Frame,
        _detections: Iterable[Detection],
        table: int,
    ) -> None:
        """Optionally collect full-frame detections for a later ROI transition."""
        del table

    def activate_roi(
        self,
        _table_bbox: Optional[tuple[int, int, int, int]],
        table: int,
        *,
        using_default_bbox: bool = False,
    ) -> bool:
        """Optionally activate a table ROI; plain detector stages stay full-frame."""
        del table, using_default_bbox
        return False

    def reset_roi(self) -> None:
        """Optionally reset table ROI state."""

    def render_view(
        self,
        frame: Frame,
        detections: Iterable[Detection],
    ) -> Tuple[Frame, List[Detection]]:
        """Return the frame and boxes that should be shown by the visualizer."""
        return frame, list(detections)


class InlineDetectorStage(BaseDetectorStage):
    """Synchronous detector execution preserving the original frame semantics."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector

    def accept(self, frame: Frame, table: int) -> DetectedFrame:
        return DetectedFrame(
            frame=frame,
            table=table,
            detections=self.detector.infer(frame, table),
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None
