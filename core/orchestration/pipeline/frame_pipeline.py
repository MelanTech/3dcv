"""逐帧处理流水线：被各轮次状态机共用。"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from core.components.counter.base import BaseCounter
from core.components.detector.base import BaseDetector
from core.components.filter.base import BaseFilter
from core.infra import pause_clock
from core.infra.logging.event_logger import EventLogger
from core.components.ocr.base import BaseOcr
from core.components.table_locator.base import BaseTableLocator
from core.types import Detection, Frame, RecognitionItem
from core.infra.visualization.base import BaseVisualizer
from core.infra.visualization.count_gui import BaseCountGui, NoopCountGui
from core.orchestration.pipeline.detector_stage import (
    BaseDetectorStage,
    DetectedFrame,
    InlineDetectorStage,
)
from core.utils.box import bbox_iou


class FramePipeline:
    """依次执行：检测 → 桌面定位 → 深度过滤 → OCR → 计数 → 可视化。"""

    def __init__(
        self,
        detector: BaseDetector,
        table_locator: BaseTableLocator,
        table_filter: BaseFilter,
        ocr: BaseOcr,
        counter: BaseCounter,
        visualizer: BaseVisualizer,
        logger: EventLogger,
        count_gui: Optional[BaseCountGui] = None,
        unknown_merger=None,
        log_per_frame: bool = False,
        ignored_by_counter: Iterable[str] = (),
        round_started_at: Optional[float] = None,
        detector_stage: Optional[BaseDetectorStage] = None,
    ):
        self.detector = detector
        self.table_locator = table_locator
        self.table_filter = table_filter
        self.ocr = ocr
        self.counter = counter
        self.visualizer = visualizer
        self.count_gui = count_gui or NoopCountGui()
        self.logger = logger
        self.unknown_merger = unknown_merger
        self.log_per_frame = bool(log_per_frame)
        self.ignored_by_counter = set(ignored_by_counter)
        self.current_state_name: Optional[str] = None
        self.round_started_at = round_started_at
        self._is_closing = False
        self.detector_stage = detector_stage or InlineDetectorStage(detector)

    def set_state(self, state_name: str) -> None:
        """记录当前状态机状态名，供可视化 overlay 使用。"""
        self.current_state_name = state_name

    def clear_state(self) -> None:
        """清空当前状态机状态名。"""
        self.current_state_name = None

    def reset_table_state(self) -> None:
        """清空不应从一个桌位/时间窗泄漏到下一个的状态。"""
        self.flush()
        self.table_locator.clear()
        self.counter.clear()
        self.logger.event("pipeline_table_state_reset")

    def preview_frame(self, frame: Frame, table: int) -> None:
        """仅渲染一帧，不改变识别状态。"""
        self._render(frame, [], table, stage="preview")

    def track_frame(self, frame: Frame, table: int) -> None:
        """只更新桌面定位状态；用于 round2 的桌面锁定（acquire）阶段。"""
        self._render(frame, [], table, stage="preview")
        detections = self.detector.infer(frame, table)
        self._render(frame, detections, table, stage="detect")
        detections = self.table_locator.process(detections, table)
        for stage in ("track", "locate", "filter", "ocr", "final"):
            self._render(frame, detections, table, stage=stage)
        if self.log_per_frame:
            self.logger.event(
                "pipeline_track",
                table=table,
                frame_id=frame.frame_id,
                is_localized=self.table_locator.is_localized,
                is_stable=self.table_locator.is_stable,
                detection_count=len(detections),
            )
        self._publish_counts(self.counter.get_counts())

    def process_frame(self, frame: Frame, table: int) -> Dict[str, int]:
        """处理单帧，并返回该桌位当前平滑后的计数结果。"""
        detected_frame = self.detector_stage.accept(frame, table)
        if detected_frame is None:
            return self._publish_counts(self.counter.get_counts())
        return self._process_detected_frame(detected_frame)

    def flush(self) -> Dict[str, int]:
        """Process the last async detector result before committing counts."""
        detected_frame = self.detector_stage.flush()
        if detected_frame is None:
            return self._publish_counts(self.counter.get_counts())
        return self._process_detected_frame(detected_frame)

    def _process_detected_frame(
        self,
        detected_frame: DetectedFrame,
    ) -> Dict[str, int]:
        """Run all pipeline stages after detector inference."""
        frame = detected_frame.frame
        table = detected_frame.table
        detections = detected_frame.detections
        self._render(frame, detections, table, stage="detect")
        if self.log_per_frame:
            self.logger.event(
                "pipeline_detect",
                table=table,
                frame_id=frame.frame_id,
                detection_count=len(detections),
            )

        detections = self.table_locator.process(detections, table)
        self._render(frame, detections, table, stage="locate")
        if self.log_per_frame:
            self.logger.event(
                "pipeline_locate_table",
                table=table,
                frame_id=frame.frame_id,
                is_localized=self.table_locator.is_localized,
                detection_count=len(detections),
            )

        if not self.table_locator.is_localized:
            # 桌面区域尚未定位时，过滤/计数结果不可靠，直接返回当前计数。
            self._render(frame, detections, table, stage="final")
            return self._publish_counts(self.counter.get_counts())

        if self.unknown_merger is not None:
            unknown_candidates = self.unknown_merger.infer(frame, detections, table)
            detections = self._merge_unknown_candidates(detections, unknown_candidates)
            if self.log_per_frame:
                self.logger.event(
                    "pipeline_unknown_merger",
                    table=table,
                    frame_id=frame.frame_id,
                    unknown_candidate_count=len(unknown_candidates),
                    detection_count=len(detections),
                )

        detections = self.table_filter.process(detections, frame, table)
        self._render(frame, detections, table, stage="filter")
        if self.log_per_frame:
            self.logger.event(
                "pipeline_filter",
                table=table,
                frame_id=frame.frame_id,
                detection_count=len(detections),
            )

        ocr_detections = self.ocr.process(frame, detections, table)
        detections, ocr_replaced_count = self._merge_ocr_detections(
            detections,
            ocr_detections,
        )
        self._render(frame, detections, table, stage="ocr")
        if self.log_per_frame:
            self.logger.event(
                "pipeline_ocr",
                table=table,
                frame_id=frame.frame_id,
                ocr_detection_count=len(ocr_detections),
                ocr_replaced_count=ocr_replaced_count,
                detection_count=len(detections),
            )

        self._render(frame, detections, table, stage="final")

        countable_detections = [
            detection
            for detection in detections
            if detection.class_name not in self.ignored_by_counter
        ]
        self.counter.update(countable_detections)
        counts = self.counter.get_counts()
        if self.log_per_frame:
            self.logger.event(
                "pipeline_counter_update",
                table=table,
                frame_id=frame.frame_id,
                counts=counts,
            )
        return self._publish_counts(counts)

    def _publish_counts(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Refresh the count panel and convert its exit action into graceful stop."""
        if self._is_closing:
            return counts
        if self.count_gui.update(counts):
            self.logger.event("count_gui_exit_requested", counts=counts)
            raise KeyboardInterrupt("count GUI exit requested")
        return counts

    def get_items(self, table: int) -> List[RecognitionItem]:
        """把当前计数器状态转换成提交给裁判的结果条目。"""
        return [
            RecognitionItem(
                goal_id=goal_id,
                num=count,
                table=table,
                confidence=1.0,
                evidence={"source": "counter"},
            )
            for goal_id, count in sorted(self.counter.get_counts().items())
        ]

    def _merge_ocr_detections(
        self,
        detections: List[Detection],
        ocr_detections: List[Detection],
    ) -> tuple[List[Detection], int]:
        """OCR 命中时替换原候选框；OCR 失败的候选框保留。

        PaddleOcr 返回的 bbox 与原始 Book bbox 一致。这里按 bbox 匹配：
        - 匹配到 OCR 结果：用 W00x 等 OCR 类别替换原 Book；
        - 没匹配到 OCR 结果：保留原 Book，后续 counter 仍按 ignored_by_counter 忽略。
        """
        if not ocr_detections:
            return detections, 0

        candidate_classes = set(getattr(self.ocr, "candidate_classes", ()))
        if not candidate_classes:
            candidate_classes = {"Book"}

        ocr_by_bbox = {
            tuple(ocr_detection.bbox): ocr_detection
            for ocr_detection in ocr_detections
        }
        merged: List[Detection] = []
        replaced_count = 0

        for detection in detections:
            replacement = None
            if detection.class_name in candidate_classes:
                replacement = ocr_by_bbox.pop(tuple(detection.bbox), None)

            if replacement is None:
                merged.append(detection)
                continue

            merged.append(replacement)
            replaced_count += 1

        # 理论上 OCR bbox 都来自原候选框；保留兜底，避免异常 OCR 结果被静默丢弃。
        merged.extend(ocr_by_bbox.values())
        return merged, replaced_count

    @staticmethod
    def _merge_unknown_candidates(
        detections: List[Detection],
        unknown_candidates: List[Detection],
    ) -> List[Detection]:
        """Append OCR-only unknown candidates without replacing main detections."""
        if not unknown_candidates:
            return detections

        suppressed = set()
        for candidate in unknown_candidates:
            source_classes = set(candidate.evidence.get("suppress_source_classes", ()))
            iou_threshold = float(candidate.evidence.get("suppress_source_iou_threshold", 0.0))
            if not source_classes or iou_threshold <= 0.0:
                continue
            for index, detection in enumerate(detections):
                if detection.class_name not in source_classes:
                    continue
                if bbox_iou(candidate.bbox, detection.bbox) >= iou_threshold:
                    suppressed.add(index)

        merged = [
            detection
            for index, detection in enumerate(detections)
            if index not in suppressed
        ]
        merged.extend(unknown_candidates)
        return merged

    def _render(
        self,
        frame: Frame,
        detections,
        table: int,
        stage: str,
    ) -> None:
        """带上当前状态机状态名进行可视化渲染。"""
        self.visualizer.render(
            frame,
            detections,
            table,
            stage=stage,
            state_name=self.current_state_name,
            elapsed_sec=(
                None
                if self.round_started_at is None
                else max(0.0, pause_clock.now() - self.round_started_at)
            ),
        )

    def close(self) -> None:
        """关闭流水线各组件持有的可选资源。"""
        self._is_closing = True
        try:
            self.flush()
        finally:
            self.detector_stage.close()
        self.count_gui.close()
        self.visualizer.close()
        close_filter = getattr(self.table_filter, "close", None)
        if close_filter is not None:
            close_filter()
        close_ocr = getattr(self.ocr, "close", None)
        if close_ocr is not None:
            close_ocr()
        close_detector = getattr(self.detector, "close", None)
        if close_detector is not None:
            close_detector()
        close_unknown = getattr(self.unknown_merger, "close", None)
        if close_unknown is not None:
            close_unknown()
        self._close_acl_runtime_if_initialized()

    @staticmethod
    def _close_acl_runtime_if_initialized() -> None:
        """推理模型释放后立即关闭 ACL runtime，避免拖到进程退出阶段释放。"""
        try:
            from core.infra.inference.acl.resource_manager import AclResourceManager
        except Exception:
            return

        instance = getattr(AclResourceManager, "_instance", None)
        if instance is not None:
            instance.close()
