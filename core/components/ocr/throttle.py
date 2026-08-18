"""OCR throttling wrapper that reduces expensive OCR calls without caching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from core.components.ocr.base import BaseOcr
from core.types import Detection, Frame


@dataclass
class _TableThrottleState:
    frame_index: int = 0
    last_run_frame: Optional[int] = None
    last_success: bool = False
    had_candidate: bool = False


class ThrottledOcr(BaseOcr):
    """Gate calls to an OCR implementation while preserving the OCR API."""

    def __init__(
        self,
        wrapped: BaseOcr,
        config: Dict,
        class_registry: Optional[Dict] = None,
    ):
        self.wrapped = wrapped
        throttle_config = dict(config.get("throttle") or {})
        self.enabled = bool(throttle_config.get("enabled", False))
        self.interval_frames = int(throttle_config.get("interval_frames", 1))
        self.retry_interval_frames = int(
            throttle_config.get("retry_interval_frames", self.interval_frames)
        )
        if self.interval_frames < 1:
            raise ValueError("ocr.throttle.interval_frames must be at least 1")
        if self.retry_interval_frames < 1:
            raise ValueError("ocr.throttle.retry_interval_frames must be at least 1")

        registry = class_registry or {}
        self.candidate_classes: Set[str] = set(
            registry.get(
                "ocr_candidate_classes",
                getattr(wrapped, "candidate_classes", ["Book"]),
            )
        )
        self.output_classes: Set[str] = set(
            registry.get(
                "ocr_output_classes",
                getattr(wrapped, "output_classes", []),
            )
        )
        self._states: Dict[int, _TableThrottleState] = {}

    def process(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        if not self.enabled:
            return self.wrapped.process(frame, detections, table)

        state = self._states.setdefault(int(table), _TableThrottleState())
        state.frame_index += 1

        has_candidate = any(
            detection.class_name in self.candidate_classes
            for detection in detections
        )
        if not has_candidate:
            state.had_candidate = False
            return []

        should_run = self._should_run(state)
        state.had_candidate = True
        if not should_run:
            return []

        results = self.wrapped.process(frame, detections, table)
        state.last_run_frame = state.frame_index
        state.last_success = any(
            detection.class_name in self.output_classes
            for detection in results
        )
        return results

    def _should_run(self, state: _TableThrottleState) -> bool:
        if state.last_run_frame is None:
            return True
        if not state.had_candidate:
            return True

        elapsed = state.frame_index - state.last_run_frame
        if state.last_success:
            return elapsed >= self.interval_frames
        return elapsed >= self.retry_interval_frames

    def close(self) -> None:
        close = getattr(self.wrapped, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)
