"""帧采集器：在指定时间窗内从帧源拉取帧，支持时间缩放。"""
from __future__ import annotations

import time
from typing import Iterator, List, Optional

from core.infra.logging.event_logger import EventLogger
from core.types import Frame


class FrameCollector:
    """按“名义时长 × 时间缩放”从帧源采集帧；缩放便于离线回放时加速/减速。"""

    def __init__(self, frame_source, logger: EventLogger, time_scale: float):
        self.frame_source = frame_source
        self.logger = logger
        self.time_scale = float(time_scale)

    def scaled_duration(self, nominal_sec: float) -> float:
        """把比赛规定的名义时长换算成实际采集时长。"""
        return max(0.0, float(nominal_sec) * self.time_scale)

    def iter_frames(self, nominal_sec: float, reason: str) -> Iterator[Frame]:
        """在给定时间窗内逐帧产出；至少产出一帧，结束时记录采集统计。"""
        actual_sec = self.scaled_duration(nominal_sec)
        started = time.monotonic()
        deadline = started + actual_sec
        frame_count = 0

        try:
            while time.monotonic() < deadline or frame_count == 0:
                frame = next(self.frame_source)
                frame_count += 1
                yield frame
                if actual_sec <= 0:
                    break
        finally:
            elapsed = time.monotonic() - started
            self.logger.event(
                "frames_collected",
                reason=reason,
                nominal_sec=nominal_sec,
                actual_sec=actual_sec,
                elapsed_sec=elapsed,
                frame_count=frame_count,
            )

    def iter_until(self, deadline: float, reason: str, nominal_sec: Optional[float] = None) -> Iterator[Frame]:
        """按绝对 monotonic deadline 逐帧产出；deadline 已过则不产出帧。"""
        started = time.monotonic()
        actual_sec = max(0.0, float(deadline) - started)
        frame_count = 0

        try:
            while time.monotonic() < deadline:
                frame = next(self.frame_source)
                frame_count += 1
                yield frame
        finally:
            elapsed = time.monotonic() - started
            self.logger.event(
                "frames_collected",
                reason=reason,
                nominal_sec=nominal_sec,
                actual_sec=actual_sec,
                elapsed_sec=elapsed,
                frame_count=frame_count,
                deadline_sec=deadline,
            )

    def collect_frames(self, nominal_sec: float, reason: str) -> List[Frame]:
        """一次性把整个时间窗内的帧收集成列表返回。"""
        return list(self.iter_frames(nominal_sec, reason))
