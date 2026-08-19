"""第二轮（round2）状态机：多桌轮转识别流程。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from core.components.frame_source.base import BaseFrameSource
from core.components.referee.base import BaseRefereeClient
from core.infra import pause_clock
from core.infra.logging.event_logger import EventLogger
from core.orchestration.pipeline.frame_pipeline import FramePipeline
from core.orchestration.state_machine.base import BaseStateMachine
from core.orchestration.state_machine.frame_collector import FrameCollector
from core.orchestration.state_machine.state_logger import StateLogger
from core.types import RecognitionItem


class Round2StateMachine(BaseStateMachine):
    """第二轮流程：依次处理多张桌子，每桌先锁定桌面再识别，桌间发送旋转指令并切换序列。"""

    round_number = 2
    round_name = "round2"

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
        self.current_table: Optional[int] = None
        self.current_table_committed = True

    def run(self) -> Path:
        """按配置的各桌时长依次处理所有桌子；异常时先尽力保存已提交/进行中的结果。"""
        self.logger.event(
            "round_started",
            round=self.round_name,
            table_durations_sec=self.state_config["round2_table_durations_sec"],
        )

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

            durations = self.state_config["round2_table_durations_sec"]
            cumulative_duration = 0.0
            for table, duration in enumerate(durations, start=1):
                cumulative_duration += float(duration)
                table_deadline = self.round_started_at + self.frame_collector.scaled_duration(
                    cumulative_duration
                )
                self.current_table = table
                self.current_table_committed = False

                # 1) 锁定桌面：稳定定位到当前桌后才开始正式识别。
                with StateLogger(self.logger, "ACQUIRE_TABLE", table=table):
                    self._acquire_table(table, deadline=table_deadline)

                # 2) 识别窗口：在该桌时长内逐帧累计计数，并持续落盘最新结果。
                state_name = f"PROCESS_TABLE_{table}_WINDOW"
                with StateLogger(self.logger, state_name, table=table):
                    self.pipeline.set_state(state_name)
                    try:
                        for frame in self.frame_collector.iter_until(
                            table_deadline,
                            reason=f"round2_table_{table}_window",
                            nominal_sec=duration,
                        ):
                            self.pipeline.process_frame(frame, table=table)
                            latest_items = self.results + self.pipeline.get_items(table=table)
                            latest_path = self.referee_client.write_result(self.round_number, latest_items)
                            if self.log_per_frame:
                                self.logger.event("latest_result_written", table=table, path=str(latest_path))

                        self.pipeline.flush()
                        latest_items = self.results + self.pipeline.get_items(table=table)
                        latest_path = self.referee_client.write_result(self.round_number, latest_items)
                        if self.log_per_frame:
                            self.logger.event("latest_result_written", table=table, path=str(latest_path))

                        table_results = self.pipeline.get_items(table=table)
                        self.logger.event(
                            "table_window_processed",
                            table=table,
                            nominal_duration_sec=duration,
                            item_count=len(table_results),
                        )
                    finally:
                        self.pipeline.clear_state()

                # 3) 提交当前桌结果，并入总结果。
                with StateLogger(self.logger, f"COMMIT_TABLE_{table}", table=table):
                    self.results.extend(table_results)
                    self.current_table_committed = True
                    self.logger.event(
                        "table_committed",
                        table=table,
                        items=[item.__dict__ for item in table_results],
                    )

                # 4) 若还有下一桌：通知裁判旋转、切换帧源序列并重置流水线状态。
                if table < len(durations):
                    with StateLogger(self.logger, "ROTATE", from_table=table, to_table=table + 1):
                        if not self.referee_client.send_rotate():
                            raise RuntimeError("failed to send rotate command")
                    with StateLogger(self.logger, "NEXT_SEQUENCE", from_table=table, to_table=table + 1):
                        self._advance_frame_source_sequence(table + 1)
                    with StateLogger(self.logger, "RESET_TABLE_STATE", next_table=table + 1):
                        self.pipeline.reset_table_state()

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

    def _latest_results_for_interruption(self) -> List[RecognitionItem]:
        """汇总中断时应上报的结果：已提交的加上当前桌尚未提交的计数。"""
        items = list(self.results)
        if self.current_table is not None and not self.current_table_committed:
            items.extend(self.pipeline.get_items(table=self.current_table))
        return items

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
                reason=f"round2_table_{table}_acquire",
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
        """被中断时尽力保存并上报当前结果（包含进行中桌位）。"""
        self.logger.event(
            "round_interrupted",
            round=self.round_name,
            current_table=self.current_table,
            error=str(exc),
            exc_type=type(exc).__name__,
        )

        try:
            items = self._latest_results_for_interruption()
        except Exception as collect_exc:
            self.logger.event(
                "interrupted_result_collect_failed",
                current_table=self.current_table,
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
                current_table=self.current_table,
                result_path=str(result_path),
            )
        except Exception as finalize_exc:
            self.logger.event(
                "interrupted_result_finalize_failed",
                round=self.round_name,
                current_table=self.current_table,
                error=str(finalize_exc),
                exc_type=type(finalize_exc).__name__,
            )

    def _advance_frame_source_sequence(self, next_table: int) -> None:
        """切换帧源到下一桌对应的序列；帧源不支持切换则跳过，切换失败则报错。"""
        next_sequence = getattr(self.frame_source, "next_sequence", None)
        if next_sequence is None:
            self.logger.event(
                "frame_source_sequence_switch_skipped",
                next_table=next_table,
                reason="unsupported_frame_source",
            )
            return

        switched = bool(next_sequence())
        sequence_info = self._get_frame_source_sequence_info()
        self.logger.event(
            "frame_source_sequence_switched",
            next_table=next_table,
            switched=switched,
            sequence=sequence_info,
        )
        if not switched:
            raise RuntimeError(f"failed to switch frame source sequence for table {next_table}")

    def _get_frame_source_sequence_info(self):
        """获取当前帧源序列的信息（若帧源未提供该能力则返回 None）。"""
        get_info = getattr(self.frame_source, "get_current_sequence_info", None)
        if get_info is None:
            return None
        return get_info()
