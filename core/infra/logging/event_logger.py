"""事件日志器：把结构化事件写入按时间戳命名的日志文件（并可选打印到控制台）。"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict


class EventLogger:
    """结构化事件日志器，可作为上下文管理器使用，退出时自动记录异常并关闭。"""

    def __init__(self, base_dir: str, round_name: str, console: bool = True):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path(base_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{stamp}_{round_name}.log"

        self._logger = logging.getLogger(f"3dcv.{round_name}.{stamp}.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handlers: list[logging.Handler] = []

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(self.path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._add_handler(file_handler)

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._add_handler(console_handler)

    def _add_handler(self, handler: logging.Handler) -> None:
        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def event(self, event: str, **fields: Any) -> None:
        """记录一条事件：事件名加若干 key=value 字段。"""
        self._logger.info(self._format_event(event, fields))

    def _format_event(self, event: str, fields: Dict[str, Any]) -> str:
        """把事件名与字段格式化为 "event | k1=v1 | k2=v2" 形式（字段按键排序）。"""
        if not fields:
            return event
        parts = [event]
        for key in sorted(fields):
            parts.append(f"{key}={self._format_value(fields[key])}")
        return " | ".join(parts)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        return repr(value)

    def close(self) -> None:
        """移除并关闭所有日志 handler。"""
        for handler in self._handlers:
            self._logger.removeHandler(handler)
            handler.close()
        self._handlers.clear()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            message = self._format_event(
                "exception",
                {
                    "exc_type": str(exc_type),
                    "error": str(exc),
                },
            )
            if exc_type is KeyboardInterrupt:
                self._logger.error(message)
            else:
                self._logger.exception(message)
        self.close()
