"""轮次状态机的公共基类：结果保存/上报与运行时资源收尾。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import os
from pathlib import Path
import sys
from typing import Iterable, List

from core.orchestration.state_machine.state_logger import StateLogger
from core.types import RecognitionItem


class BaseStateMachine(ABC):
    """轮次状态机基类：封装结果写入/上报以及运行时资源的收尾。"""

    round_number: int
    round_name: str
    _runtime_closed = False
    _summary_printed = False

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

    def _print_result_summary_table(
        self,
        items: Iterable[RecognitionItem],
        reason: str,
    ) -> None:
        """在进程退出前打印所有物品数量汇总；正常/异常路径均只打印一次。"""
        if self._summary_printed:
            return
        self._summary_printed = True

        item_list = list(items)
        class_registry = self.config.get("class_registry", {})
        result_classes = list(class_registry.get("result_classes", []))
        result_class_to_goal_id = dict(class_registry.get("result_class_to_goal_id", {}))
        if not result_classes:
            result_classes = sorted({item.goal_id for item in item_list})

        tables = self._summary_tables(item_list)
        counts = defaultdict(int)
        for item in item_list:
            counts[(int(item.table), str(item.goal_id))] += int(item.num)

        headers = ["Item", "Goal_ID"] + [f"T{table}" for table in tables] + ["Total"]
        rows: List[List[str]] = []
        for class_name in result_classes:
            goal_id = result_class_to_goal_id.get(class_name, class_name)
            per_table = [counts[(table, class_name)] for table in tables]
            rows.append(
                [class_name, goal_id]
                + [str(value) for value in per_table]
                + [str(sum(per_table))]
            )

        widths = [
            max(len(str(row[index])) for row in [headers] + rows)
            for index in range(len(headers))
        ]

        def fmt(row):
            return " | ".join(
                str(value).rjust(widths[index]) if index >= 2 else str(value).ljust(widths[index])
                for index, value in enumerate(row)
            )

        line = "-+-".join("-" * width for width in widths)
        print("", flush=True)
        print(f"=== Result Summary ({self.round_name}, reason={reason}) ===", flush=True)
        print(fmt(headers), flush=True)
        print(line, flush=True)
        for row in rows:
            print(fmt(row), flush=True)
        print("=== End Result Summary ===", flush=True)
        print("", flush=True)

        try:
            self.logger.event(
                "result_summary_printed",
                round=self.round_name,
                reason=reason,
                item_count=len(item_list),
                table_count=len(tables),
            )
        except Exception:
            pass

    def _summary_tables(self, items: List[RecognitionItem]) -> List[int]:
        """推断 summary table 应展示哪些桌号列。"""
        item_tables = {int(item.table) for item in items}
        if self.round_number == 2:
            durations = self.config.get("state_machine", {}).get(
                "round2_table_durations_sec",
                [],
            )
            configured_tables = set(range(1, len(durations) + 1))
            tables = item_tables | configured_tables
        elif self.round_number == 1:
            tables = item_tables | {1}
        else:
            tables = item_tables
        return sorted(tables) if tables else [1]
