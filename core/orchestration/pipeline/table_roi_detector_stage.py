"""Smooth table-region zoom around an existing detector stage."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from core.infra import pause_clock
from core.infra.logging.event_logger import EventLogger
from core.orchestration.pipeline.detector_stage import BaseDetectorStage, DetectedFrame
from core.types import Detection, Frame

RoiBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class RoiTransform:
    """Mapping between a fixed-size detector view and the original frame."""

    source_bbox: RoiBox
    source_width: int
    source_height: int
    view_width: int
    view_height: int

    @property
    def zoom(self) -> float:
        x1, y1, x2, y2 = self.source_bbox
        return min(
            self.view_width / max(1, x2 - x1),
            self.view_height / max(1, y2 - y1),
        )

    @property
    def is_identity(self) -> bool:
        return self.source_bbox == (0, 0, self.source_width, self.source_height)

    def crop_frame(self, frame: Frame) -> Frame:
        """Crop the ROI and resize it back to each source array's original shape."""
        if self.is_identity:
            return frame
        return Frame(
            frame_id=frame.frame_id,
            rgb=self._crop_array(frame.rgb, interpolation=cv2.INTER_LINEAR),
            depth=self._crop_array(frame.depth, interpolation=cv2.INTER_NEAREST),
            timestamp=frame.timestamp,
        )

    def detection_to_global(self, detection: Detection) -> Detection | None:
        """Map a detector-view box back into original-frame coordinates."""
        x1, y1, x2, y2 = self.source_bbox
        roi_width = x2 - x1
        roi_height = y2 - y1
        bx1, by1, bx2, by2 = detection.bbox
        mapped = self._clip_bbox(
            (
                round(x1 + bx1 * roi_width / self.view_width),
                round(y1 + by1 * roi_height / self.view_height),
                round(x1 + bx2 * roi_width / self.view_width),
                round(y1 + by2 * roi_height / self.view_height),
            ),
            self.source_width,
            self.source_height,
        )
        if mapped is None:
            return None
        evidence = dict(detection.evidence)
        evidence["detection_roi"] = {
            "source_bbox": self.source_bbox,
            "zoom": round(self.zoom, 4),
        }
        return Detection(
            class_name=detection.class_name,
            class_id=detection.class_id,
            bbox=mapped,
            score=detection.score,
            evidence=evidence,
        )

    def detection_to_view(self, detection: Detection) -> Detection | None:
        """Map a global box into the zoomed view for visualization."""
        x1, y1, x2, y2 = self.source_bbox
        roi_width = x2 - x1
        roi_height = y2 - y1
        bx1, by1, bx2, by2 = detection.bbox
        mapped = self._clip_bbox(
            (
                round((bx1 - x1) * self.view_width / roi_width),
                round((by1 - y1) * self.view_height / roi_height),
                round((bx2 - x1) * self.view_width / roi_width),
                round((by2 - y1) * self.view_height / roi_height),
            ),
            self.view_width,
            self.view_height,
        )
        if mapped is None:
            return None
        return Detection(
            class_name=detection.class_name,
            class_id=detection.class_id,
            bbox=mapped,
            score=detection.score,
            evidence=detection.evidence,
        )

    def _crop_array(self, value, interpolation: int):
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim < 2:
            return value

        array_height, array_width = array.shape[:2]
        x1, y1, x2, y2 = self.source_bbox
        ax1 = math.floor(x1 * array_width / self.source_width)
        ay1 = math.floor(y1 * array_height / self.source_height)
        ax2 = math.ceil(x2 * array_width / self.source_width)
        ay2 = math.ceil(y2 * array_height / self.source_height)
        ax1 = max(0, min(array_width - 1, ax1))
        ay1 = max(0, min(array_height - 1, ay1))
        ax2 = max(ax1 + 1, min(array_width, ax2))
        ay2 = max(ay1 + 1, min(array_height, ay2))
        crop = array[ay1:ay2, ax1:ax2]
        resized = cv2.resize(
            crop,
            (array_width, array_height),
            interpolation=interpolation,
        )
        return np.ascontiguousarray(resized)

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> RoiBox | None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


@dataclass(frozen=True)
class _RoiObservation:
    frame_shape: tuple[int, int]
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class _PendingFrame:
    original_frame: Frame
    detector_frame: Frame
    render_frame: Frame
    table: int
    detector_transform: RoiTransform
    render_transform: RoiTransform
    full_frame_probe: bool


class TableRoiDetectorStage(BaseDetectorStage):
    """Wrap a detector stage with smooth table ROI zoom and box remapping."""

    _FULL_FRAME_PROBE_INTERVAL_SEC = 2.0
    _MIN_EXPANSION_PX = 8

    def __init__(
        self,
        wrapped: BaseDetectorStage,
        config: dict,
        logger: EventLogger,
        clock: Callable[[], float] = pause_clock.now,
    ):
        self.wrapped = wrapped
        self.logger = logger
        self.clock = clock
        self.enabled = bool(config.get("enabled", False))
        self.padding_px = max(0, int(config.get("padding_px", 24)))
        self.transition_sec = float(config.get("transition_sec", 1.0))
        self.max_zoom = float(config.get("max_zoom", 2.0))
        if self.transition_sec <= 0.0:
            raise ValueError("pipeline.detection_roi.transition_sec must be positive")
        if self.max_zoom < 1.0:
            raise ValueError("pipeline.detection_roi.max_zoom must be at least 1.0")

        self._observations: deque[_RoiObservation] = deque(maxlen=64)
        self._pending: deque[_PendingFrame] = deque()
        self._source_shape: tuple[int, int] | None = None
        self._table_bbox: RoiBox | None = None
        self._start_roi: RoiBox | None = None
        self._target_roi: RoiBox | None = None
        self._transition_started_at = 0.0
        self._last_probe_at = 0.0
        self._active = False
        self._last_render_frame_id: str | None = None
        self._last_render_frame: Frame | None = None
        self._last_render_transform: RoiTransform | None = None

    def accept(self, frame: Frame, table: int) -> DetectedFrame | None:
        render_transform = self._current_transform(frame)
        full_frame_probe = self._should_probe(render_transform)
        detector_transform = (
            self._identity_transform(frame) if full_frame_probe else render_transform
        )
        detector_frame = detector_transform.crop_frame(frame)
        render_frame = (
            render_transform.crop_frame(frame) if full_frame_probe else detector_frame
        )
        ready = self.wrapped.accept(detector_frame, table)
        self._pending.append(
            _PendingFrame(
                original_frame=frame,
                detector_frame=detector_frame,
                render_frame=render_frame,
                table=table,
                detector_transform=detector_transform,
                render_transform=render_transform,
                full_frame_probe=full_frame_probe,
            )
        )
        if full_frame_probe:
            self.logger.event(
                "detection_roi_probe_submitted",
                table=table,
                frame_id=frame.frame_id,
            )
        if ready is None:
            return None
        return self._restore_result(ready)

    def flush(self) -> DetectedFrame | None:
        ready = self.wrapped.flush()
        if ready is None:
            return None
        return self._restore_result(ready)

    def close(self) -> None:
        self.wrapped.close()
        self._pending.clear()
        self._clear_render_state()

    def observe_roi(
        self,
        frame: Frame,
        detections: Iterable[Detection],
        table: int,
    ) -> None:
        """Collect full-frame acquisition detections used to build the target ROI."""
        del table
        if not self.enabled or frame.rgb is None:
            return
        height, width = np.asarray(frame.rgb).shape[:2]
        self._observations.append(
            _RoiObservation(
                frame_shape=(height, width),
                detections=tuple(detections),
            )
        )

    def activate_roi(
        self,
        table_bbox: RoiBox | None,
        table: int,
        *,
        using_default_bbox: bool = False,
    ) -> bool:
        """Lock the target ROI and start a smooth full-frame-to-ROI transition."""
        if not self.enabled:
            return False
        if using_default_bbox:
            self.logger.event(
                "detection_roi_fallback",
                table=table,
                reason="default_table_bbox",
            )
            return False
        if not self._observations:
            self.logger.event(
                "detection_roi_fallback",
                table=table,
                reason="no_acquisition_observations",
            )
            return False

        height, width = self._observations[-1].frame_shape
        target = self._build_target_roi(table_bbox, width, height)
        if target is None or target == (0, 0, width, height):
            self.logger.event(
                "detection_roi_fallback",
                table=table,
                reason="invalid_or_full_frame_roi",
            )
            return False

        self._source_shape = (height, width)
        self._table_bbox = self._clip_bbox(table_bbox, width, height)
        self._start_roi = (0, 0, width, height)
        self._target_roi = target
        self._transition_started_at = self.clock()
        self._last_probe_at = self._transition_started_at
        self._active = True
        target_width = target[2] - target[0]
        target_height = target[3] - target[1]
        self.logger.event(
            "detection_roi_activated",
            table=table,
            bbox=target,
            padding_px=self.padding_px,
            transition_sec=self.transition_sec,
            zoom=round(min(width / target_width, height / target_height), 4),
        )
        return True

    def reset_roi(self) -> None:
        """Return to full-frame inference before acquiring another table."""
        self._observations.clear()
        self._source_shape = None
        self._table_bbox = None
        self._start_roi = None
        self._target_roi = None
        self._active = False
        self._transition_started_at = 0.0
        self._last_probe_at = 0.0
        self._clear_render_state()

    def render_view(
        self,
        frame: Frame,
        detections: Iterable[Detection],
    ) -> tuple[Frame, list[Detection]]:
        """Return the detector view and view-space boxes for visualization."""
        transform = self._last_render_transform
        if (
            transform is None
            or self._last_render_frame is None
            or self._last_render_frame_id != frame.frame_id
        ):
            return frame, list(detections)

        view_detections = []
        for detection in detections:
            mapped = transform.detection_to_view(detection)
            if mapped is not None:
                view_detections.append(mapped)
        return self._last_render_frame, view_detections

    def current_roi(self, frame_shape: tuple[int, int]) -> RoiBox:
        """Expose the current interpolated ROI for diagnostics and tests."""
        height, width = frame_shape
        return self._interpolated_roi(width, height)

    def _restore_result(self, ready: DetectedFrame) -> DetectedFrame:
        if not self._pending:
            raise RuntimeError(
                "ROI detector stage received a result without a pending frame"
            )
        pending = self._pending.popleft()
        if ready.frame.frame_id != pending.detector_frame.frame_id:
            raise RuntimeError(
                "ROI detector stage result order mismatch: "
                f"expected {pending.detector_frame.frame_id!r}, got {ready.frame.frame_id!r}"
            )

        detections = []
        for detection in ready.detections:
            mapped = pending.detector_transform.detection_to_global(detection)
            if mapped is not None:
                detections.append(mapped)

        if pending.full_frame_probe:
            self._expand_roi_if_needed(
                detections=detections,
                table=pending.table,
                width=pending.detector_transform.source_width,
                height=pending.detector_transform.source_height,
            )

        self._last_render_frame_id = pending.original_frame.frame_id
        self._last_render_frame = pending.render_frame
        self._last_render_transform = pending.render_transform
        return DetectedFrame(
            frame=pending.original_frame,
            table=pending.table,
            detections=detections,
        )

    def _current_transform(self, frame: Frame) -> RoiTransform:
        if frame.rgb is None:
            raise ValueError("table ROI requires frame.rgb")
        height, width = np.asarray(frame.rgb).shape[:2]
        return RoiTransform(
            source_bbox=self._interpolated_roi(width, height),
            source_width=width,
            source_height=height,
            view_width=width,
            view_height=height,
        )

    @staticmethod
    def _identity_transform(frame: Frame) -> RoiTransform:
        if frame.rgb is None:
            raise ValueError("table ROI requires frame.rgb")
        height, width = np.asarray(frame.rgb).shape[:2]
        return RoiTransform(
            source_bbox=(0, 0, width, height),
            source_width=width,
            source_height=height,
            view_width=width,
            view_height=height,
        )

    def _should_probe(self, render_transform: RoiTransform) -> bool:
        if not self._active or render_transform.is_identity:
            return False
        now = self.clock()
        if now - self._last_probe_at < self._FULL_FRAME_PROBE_INTERVAL_SEC:
            return False
        self._last_probe_at = now
        return True

    def _expand_roi_if_needed(
        self,
        detections: Iterable[Detection],
        table: int,
        width: int,
        height: int,
    ) -> None:
        if not self._active or self._table_bbox is None or self._target_roi is None:
            return

        candidate = self._build_roi_from_detections(
            self._table_bbox,
            detections,
            width,
            height,
        )
        if candidate is None:
            return
        union = (
            min(self._target_roi[0], candidate[0]),
            min(self._target_roi[1], candidate[1]),
            max(self._target_roi[2], candidate[2]),
            max(self._target_roi[3], candidate[3]),
        )
        expanded = self._fit_roi_to_frame(union, width, height)
        if not self._meaningfully_expands(self._target_roi, expanded):
            return

        previous_target = self._target_roi
        current = self._interpolated_roi(width, height)
        self._start_roi = current
        self._target_roi = expanded
        self._transition_started_at = self.clock()
        self.logger.event(
            "detection_roi_expanded",
            table=table,
            previous_bbox=previous_target,
            bbox=expanded,
            previous_zoom=round(
                min(
                    width / (previous_target[2] - previous_target[0]),
                    height / (previous_target[3] - previous_target[1]),
                ),
                4,
            ),
            zoom=round(
                min(
                    width / (expanded[2] - expanded[0]),
                    height / (expanded[3] - expanded[1]),
                ),
                4,
            ),
        )

    def _interpolated_roi(self, width: int, height: int) -> RoiBox:
        full = (0, 0, width, height)
        if (
            not self._active
            or self._source_shape != (height, width)
            or self._start_roi is None
            or self._target_roi is None
        ):
            return full

        progress = min(
            1.0,
            max(
                0.0, (self.clock() - self._transition_started_at) / self.transition_sec
            ),
        )
        eased = progress * progress * (3.0 - 2.0 * progress)
        start_x1, start_y1, start_x2, start_y2 = self._start_roi
        target_x1, target_y1, target_x2, target_y2 = self._target_roi
        center_x = self._lerp(
            (start_x1 + start_x2) * 0.5,
            (target_x1 + target_x2) * 0.5,
            eased,
        )
        center_y = self._lerp(
            (start_y1 + start_y2) * 0.5,
            (target_y1 + target_y2) * 0.5,
            eased,
        )
        desired_width = self._lerp(
            start_x2 - start_x1,
            target_x2 - target_x1,
            eased,
        )
        aspect_width, aspect_height = self._aspect_units(width, height)
        steps = max(1, min(width // aspect_width, round(desired_width / aspect_width)))
        roi_width = aspect_width * steps
        roi_height = aspect_height * steps
        left = round(center_x - roi_width * 0.5)
        top = round(center_y - roi_height * 0.5)
        left = max(0, min(width - roi_width, left))
        top = max(0, min(height - roi_height, top))
        return left, top, left + roi_width, top + roi_height

    def _build_target_roi(
        self,
        table_bbox: RoiBox | None,
        width: int,
        height: int,
    ) -> RoiBox | None:
        table = self._clip_bbox(table_bbox, width, height)
        if table is None:
            return None

        detections = (
            detection
            for observation in self._observations
            if observation.frame_shape == (height, width)
            for detection in observation.detections
        )
        return self._build_roi_from_detections(
            table,
            detections,
            width,
            height,
        )

    def _build_roi_from_detections(
        self,
        table: RoiBox,
        detections: Iterable[Detection],
        width: int,
        height: int,
    ) -> RoiBox:
        tx1, ty1, tx2, ty2 = table
        object_boxes: list[RoiBox] = []
        for detection in detections:
            if detection.class_name == "Table":
                continue
            bbox = self._clip_bbox(detection.bbox, width, height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) * 0.5
            if center_x < tx1 - self.padding_px or center_x > tx2 + self.padding_px:
                continue
            if y2 < ty1 - self.padding_px or y1 > ty2 + self.padding_px:
                continue
            object_boxes.append(bbox)

        left = min([tx1, *(bbox[0] for bbox in object_boxes)]) - self.padding_px
        right = max([tx2, *(bbox[2] for bbox in object_boxes)]) + self.padding_px
        top = min([ty1, *(bbox[1] for bbox in object_boxes)]) - self.padding_px
        bottom = ty2 + self.padding_px
        return self._fit_roi_to_frame(
            (left, top, right, bottom),
            width,
            height,
        )

    def _meaningfully_expands(self, previous: RoiBox, candidate: RoiBox) -> bool:
        contains_previous = (
            candidate[0] <= previous[0]
            and candidate[1] <= previous[1]
            and candidate[2] >= previous[2]
            and candidate[3] >= previous[3]
        )
        if not contains_previous:
            return False
        expansion = max(
            previous[0] - candidate[0],
            previous[1] - candidate[1],
            candidate[2] - previous[2],
            candidate[3] - previous[3],
        )
        threshold = max(self._MIN_EXPANSION_PX, self.padding_px // 2)
        return expansion >= threshold

    def _fit_roi_to_frame(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> RoiBox:
        x1, y1, x2, y2 = bbox
        raw_width = max(1.0, x2 - x1)
        raw_height = max(1.0, y2 - y1)
        aspect_width, aspect_height = self._aspect_units(width, height)
        aspect_steps = math.ceil(
            max(
                raw_width / aspect_width,
                raw_height / aspect_height,
                width / self.max_zoom / aspect_width,
                height / self.max_zoom / aspect_height,
            )
        )
        target_width = aspect_width * aspect_steps
        target_height = aspect_height * aspect_steps
        if target_width >= width or target_height >= height:
            return 0, 0, width, height

        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        left = round(center_x - target_width * 0.5)
        top = round(center_y - target_height * 0.5)
        left = max(0, min(width - target_width, left))
        top = max(0, min(height - target_height, top))
        return left, top, left + target_width, top + target_height

    @staticmethod
    def _aspect_units(width: int, height: int) -> tuple[int, int]:
        divisor = math.gcd(width, height)
        return width // divisor, height // divisor

    @staticmethod
    def _lerp(start: float, target: float, progress: float) -> float:
        return start + (target - start) * progress

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> RoiBox | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _clear_render_state(self) -> None:
        self._last_render_frame_id = None
        self._last_render_frame = None
        self._last_render_transform = None
