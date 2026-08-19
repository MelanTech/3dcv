"""第一轮（round1）状态机：单桌识别流程。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from core.components.frame_source.base import BaseFrameSource
from core.components.referee.base import BaseRefereeClient
from core.infra import pause_clock
from core.infra.logging.event_logger import EventLogger
from core.orchestration.pipeline.frame_pipeline import FramePipeline
from core.orchestration.state_machine.base import BaseStateMachine
from core.orchestration.state_machine.frame_collector import FrameCollector
from core.orchestration.state_machine.state_logger import StateLogger
from core.types import RecognitionItem


class Round1StateMachine(BaseStateMachine):
    """第一轮流程：只处理 1 号桌，在一个时间窗内持续识别并输出结果。"""

    round_number = 1
    round_name = "round1"

    def __init__(
        self,
        config: dict,
        frame_source: BaseFrameSource,
        pipeline: FramePipeline,
        referee_client: BaseRefereeClient,
        logger: EventLogger,
        round_started_at: float,
        start_signal_sent: bool = False,
    ):
        self.config = config
        self.frame_source = frame_source
        self.pipeline = pipeline
        self.referee_client = referee_client
        self.logger = logger
        self.round_started_at = float(round_started_at)
        self.start_signal_sent = bool(start_signal_sent)
        self.state_config = config["state_machine"]
        self.log_per_frame = bool(config.get("logging", {}).get("per_frame", False))
        self.frame_collector = FrameCollector(frame_source, logger, self.state_config["time_scale"])
        self.results: List[RecognitionItem] = []

    def run(self) -> Path:
        """执行完整的第一轮流程；任何异常都会先尽力保存结果再向上抛出。"""
        round_deadline = self.round_started_at + self.frame_collector.scaled_duration(
            self.state_config["round1_recognize_sec"]
        )
        self.logger.event("round_started", round=self.round_name)

        try:
            with StateLogger(self.logger, "INIT"):
                pass

            if not self.start_signal_sent:
                with StateLogger(self.logger, "CONNECT_REFEREE"):
                    if not self.referee_client.connect():
                        raise RuntimeError("failed to connect referee box")

                with StateLogger(self.logger, "SEND_START"):
                    if not self.referee_client.send_start():
                        raise RuntimeError("failed to send start signal")
            else:
                self.logger.event("start_signal_already_sent")

            with StateLogger(self.logger, "INIT_PIPELINE"):
                self.pipeline.reset_table_state()

            # 锁定桌面：稳定定位到 1 号桌后再开始正式识别。
            with StateLogger(self.logger, "ACQUIRE_TABLE", table=1):
                self._acquire_table(table=1, deadline=round_deadline)

            with StateLogger(self.logger, "PROCESS_TABLE_1_WINDOW", table=1):
                # 在识别时间窗内逐帧处理，并把每帧最新结果落盘（便于中断时兜底）。
                self.pipeline.set_state("PROCESS_TABLE_1_WINDOW")
                try:
                    for frame in self.frame_collector.iter_until(
                        round_deadline,
                        reason="round1_table_1_window",
                        nominal_sec=self.state_config["round1_recognize_sec"],
                    ):
                        self.pipeline.process_frame(frame, table=1)
                        latest_items = self.pipeline.get_items(table=1)
                        latest_path = self.referee_client.write_result(self.round_number, latest_items)
                        if self.log_per_frame:
                            self.logger.event("latest_result_written", table=1, path=str(latest_path))
                    self.pipeline.flush()
                    latest_items = self.pipeline.get_items(table=1)
                    latest_path = self.referee_client.write_result(self.round_number, latest_items)
                    if self.log_per_frame:
                        self.logger.event("latest_result_written", table=1, path=str(latest_path))
                finally:
                    self.pipeline.clear_state()

            with StateLogger(self.logger, "COMMIT_TABLE_1", table=1):
                self.results = self.pipeline.get_items(table=1)
                self.logger.event(
                    "table_committed",
                    table=1,
                    items=[item.__dict__ for item in self.results],
                )

            result_path = self._write_and_send_result(
                self.results,
                reason="normal",
                strict=True,
            )
            self._print_result_summary_table(self.results, reason="normal")
            self.logger.event("round_finished", round=self.round_name, result_path=str(result_path))
            self._normal_completion = True
            return result_path
        except BaseException as exc:
            self._finalize_interrupted_run(exc)
            raise
        finally:
            self._close_runtime_resources()

    def _acquire_table(self, table: int, deadline: float) -> None:
        """在最长等待时间内尝试稳定锁定桌面；超时则回退到默认桌面框。"""
        min_actual_sec = self.frame_collector.scaled_duration(
            self.state_config["settle_wait_sec"]
        )
        max_acquire_sec = self.frame_collector.scaled_duration(
            self.state_config.get("max_acquire_sec", self.state_config["settle_wait_sec"])
        )
        started = pause_clock.now()
        acquire_deadline = min(started + max_acquire_sec, deadline)
        acquired = False
        frame_count = 0

        self.pipeline.set_state("ACQUIRE_TABLE")
        try:
            for frame in self.frame_collector.iter_until(
                acquire_deadline,
                reason=f"round1_table_{table}_acquire",
                nominal_sec=self.state_config.get("max_acquire_sec", self.state_config["settle_wait_sec"]),
            ):
                frame_count += 1
                self.pipeline.track_frame(frame, table=table)
                elapsed = pause_clock.now() - started
                if elapsed >= min_actual_sec and self.pipeline.table_locator.is_acquired:
                    acquired = True
                    break
        finally:
            self.pipeline.clear_state()

        self.logger.event(
            "table_acquire_finished",
            table=table,
            acquired=acquired,
            is_localized=self.pipeline.table_locator.is_localized,
            is_stable=self.pipeline.table_locator.is_stable,
            frame_count=frame_count,
            elapsed_sec=pause_clock.now() - started,
        )
        if not acquired:
            self.pipeline.table_locator.handle_acquire_timeout("acquire_timeout")
            self.logger.event(
                "table_acquire_fallback",
                table=table,
                reason="max_acquire_reached",
                fallback="default_bbox",
            )

    def _finalize_interrupted_run(self, exc: BaseException) -> None:
        """被中断（如 Ctrl+C）时，尽最大努力保存并上报当前已有结果。"""
        self.logger.event(
            "round_interrupted",
            round=self.round_name,
            error=str(exc),
            exc_type=type(exc).__name__,
        )

        try:
            items = self.results if self.results else self.pipeline.get_items(table=1)
        except Exception as collect_exc:
            self.logger.event(
                "interrupted_result_collect_failed",
                error=str(collect_exc),
                exc_type=type(collect_exc).__name__,
            )
            items = self.results

        self._print_result_summary_table(items, reason="interrupted")

        try:
            result_path = self._write_and_send_result(
                items,
                reason="interrupted",
                strict=False,
            )
            self.logger.event(
                "interrupted_result_finalized",
                round=self.round_name,
                result_path=str(result_path),
            )
        except Exception as finalize_exc:
            self.logger.event(
                "interrupted_result_finalize_failed",
                round=self.round_name,
                error=str(finalize_exc),
                exc_type=type(finalize_exc).__name__,
            )
