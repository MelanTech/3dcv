"""状态日志上下文管理器：记录状态进入/退出及其耗时。"""
from __future__ import annotations

import time

from core.infra.logging.event_logger import EventLogger


class StateLogger:
    """上下文管理器：进入/退出某个状态时各打一条事件日志，并记录该状态耗时。"""

    def __init__(self, logger: EventLogger, state: str, **fields):
        self.logger = logger
        self.state = state
        self.fields = fields
        self.started = 0.0

    def __enter__(self):
        self.started = time.monotonic()
        self.logger.event("state_enter", state=self.state, **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.event(
            "state_exit",
            state=self.state,
            elapsed_sec=time.monotonic() - self.started,
            ok=exc is None,
            **self.fields,
        )
