"""轮次状态机的公共基类：结果保存/上报与运行时资源收尾。"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import sys
from typing import Iterable

from core.orchestration.state_machine.state_logger import StateLogger
from core.types import RecognitionItem


class BaseStateMachine(ABC):
    """轮次状态机基类：封装结果写入/上报以及运行时资源的收尾。"""

    round_number: int
    round_name: str
    _runtime_closed = False

    @abstractmethod
    def run(self) -> Path:
        raise NotImplementedError

    def _referee_enabled(self) -> bool:
        """裁判盒是否启用（未配置时默认视为启用）。"""
        return bool(getattr(self.referee_client, "enabled", True))

    def _write_and_send_result(
        self,
        items: Iterable[RecognitionItem],
        reason: str,
        strict: bool,
    ) -> Path:
        """先把结果落盘，再尝试发送给裁判盒；strict=True 时发送失败会抛异常。"""
        with StateLogger(self.logger, "SAVE_RESULT", reason=reason):
            result_path = self.referee_client.write_result(self.round_number, items)
            self.logger.event("result_saved", path=str(result_path), reason=reason)

        with StateLogger(self.logger, "SEND_RESULT", reason=reason):
            if self._referee_enabled():
                sent = self.referee_client.send_result_file(result_path)
                if not sent:
                    message = "failed to send result file"
                    self.logger.event(
                        "result_send_failed",
                        path=str(result_path),
                        reason=reason,
                    )
                    if strict:
                        raise RuntimeError(message)
            else:
                self.logger.event(
                    "result_send_skipped",
                    enabled=False,
                    path=str(result_path),
                    reason=reason,
                )

        return result_path

    def _close_runtime_resources(self) -> None:
        """按顺序关闭流水线、裁判客户端和帧源；保证只执行一次。"""
        if self._runtime_closed:
            return
        self._runtime_closed = True

        with StateLogger(self.logger, "STOP"):
            self._close_one("pipeline", self.pipeline.close)
            self._close_one("referee_client", self.referee_client.close)
            close_frame_source = getattr(self.frame_source, "close", None)
            if close_frame_source is not None:
                self._close_one("frame_source", close_frame_source)

        if self._should_fast_exit_after_stop():
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    def _should_fast_exit_after_stop(self) -> bool:
        """CLI 成功完成后可直接退出，跳过易崩的 native 解释器析构阶段。"""
        enabled = os.environ.get("CV3D_FAST_EXIT_AFTER_STOP", "0").strip().lower()
        return enabled in {"1", "true", "yes", "on"} and bool(
            getattr(self, "_normal_completion", False)
        )

    def _close_one(self, name: str, close) -> None:
        """关闭单个资源，吞掉异常并记录日志，避免收尾阶段互相影响。"""
        try:
            close()
            self.logger.event("resource_closed", resource=name)
        except Exception as exc:
            self.logger.event(
                "resource_close_failed",
                resource=name,
                error=str(exc),
                exc_type=type(exc).__name__,
            )
