"""Whole-table OCR fallback for missed book/OCR candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.components.ocr.base import BaseOcr
from core.types import Detection, Frame
from core.utils.box import bbox_iou


@dataclass
class _Line:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


class TableFallbackOcr(BaseOcr):
    """Run low-frequency OCR over the table when no OCR candidate is available."""

    def __init__(
        self,
        wrapped: BaseOcr,
        config: Dict,
        class_registry: Optional[Dict] = None,
    ):
        self.wrapped = wrapped
        fallback_config = dict(config.get("table_fallback") or {})
        self.enabled = bool(fallback_config.get("enabled", False))
        self.interval_frames = max(1, int(fallback_config.get("interval_frames", 12)))
        self.min_match_score = float(fallback_config.get("min_match_score", 80.0))
        self.trigger_only_without_candidates = bool(
            fallback_config.get("trigger_only_without_candidates", True)
        )
        self.table_padding_ratio = max(
            0.0,
            float(fallback_config.get("table_padding_ratio", 0.0)),
        )
        self.line_group_max_gap_ratio = max(
            0.0,
            float(fallback_config.get("line_group_max_gap_ratio", 2.5)),
        )
        self.line_group_min_x_overlap_ratio = max(
            0.0,
            min(1.0, float(fallback_config.get("line_group_min_x_overlap_ratio", 0.25))),
        )
        self.suppress_iou_threshold = max(
            0.0,
            min(1.0, float(fallback_config.get("suppress_iou_threshold", 0.5))),
        )

        registry = class_registry or {}
        self.candidate_classes = set(
            registry.get(
                "ocr_candidate_classes",
                getattr(wrapped, "candidate_classes", ["Book"]),
            )
        )
        self.output_classes = set(
            registry.get(
                "ocr_output_classes",
                getattr(wrapped, "output_classes", []),
            )
        )
        configured_outputs = fallback_config.get("output_classes")
        if configured_outputs:
            self.output_classes &= {str(class_name) for class_name in configured_outputs}
        self._frame_indices: Dict[int, int] = {}

    def process(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        results = self.wrapped.process(frame, detections, table)
        if not self.enabled or frame.rgb is None:
            return results

        has_candidate = any(
            detection.class_name in self.candidate_classes
            for detection in detections
        )
        if self.trigger_only_without_candidates and has_candidate:
            return results

        table = int(table)
        frame_index = self._frame_indices.get(table, 0) + 1
        self._frame_indices[table] = frame_index
        if (frame_index - 1) % self.interval_frames != 0:
            return results

        fallback_results = self._run_table_fallback(frame, detections, table)
        if not fallback_results:
            return results
        return results + self._suppress_existing(fallback_results, results)

    def _run_table_fallback(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        table_bbox = self._table_bbox(detections, frame.rgb.shape[:2])
        if table_bbox is None:
            return []

        details = self._read_details(frame.rgb, table_bbox)
        # If document orientation rotated the table crop, OCR boxes are no longer
        # in the original crop coordinate system. Skip rather than returning bad boxes.
        if details.get("orientation_applied"):
            return []

        lines = self._details_to_lines(details)
        if not lines:
            return []

        x1, y1, _x2, _y2 = table_bbox
        best_by_class: Dict[str, Detection] = {}
        for group in self._group_lines(lines):
            text = "".join(line.text for line in group)
            class_name, score = self._classify(text)
            if class_name is None or class_name not in self.output_classes:
                continue
            if score < self.min_match_score:
                continue
            bbox = self._union_bbox([line.bbox for line in group])
            global_bbox = (
                bbox[0] + x1,
                bbox[1] + y1,
                bbox[2] + x1,
                bbox[3] + y1,
            )
            detection = Detection(
                class_name=class_name,
                bbox=global_bbox,
                score=1.0,
                evidence={
                    "source": "table_ocr_fallback",
                    "table": table,
                    "text": text,
                    "line_count": len(group),
                    "match_score": score,
                },
            )
            previous = best_by_class.get(class_name)
            if previous is None or score > float(previous.evidence.get("match_score", 0.0)):
                best_by_class[class_name] = detection
        return list(best_by_class.values())

    def _read_details(self, rgb, bbox) -> Dict:
        read_details = getattr(self.wrapped, "_read_details", None)
        if read_details is None:
            return {}
        return dict(read_details(rgb, bbox))

    def _classify(self, text: str):
        classify = getattr(self.wrapped, "_classify", None)
        if classify is None:
            return None, 0.0
        class_name, score = classify(text)
        if class_name is None:
            return None, 0.0
        return str(class_name), float(score)

    def _details_to_lines(self, details: Dict) -> List[_Line]:
        boxes = list(details.get("boxes") or [])
        recognitions = list(details.get("recognitions") or [])
        if not boxes or not recognitions:
            return []

        lines: List[_Line] = []
        for box, recognition in zip(boxes, recognitions):
            text = str(recognition.get("text", "")).strip()
            if not text:
                continue
            bbox = self._box_to_bbox(box)
            if bbox is None:
                continue
            lines.append(
                _Line(
                    text=text,
                    confidence=float(recognition.get("confidence", 0.0)),
                    bbox=bbox,
                )
            )
        return lines

    def _group_lines(self, lines: List[_Line]) -> List[List[_Line]]:
        groups: List[List[_Line]] = []
        for line in sorted(lines, key=lambda item: (self._center_y(item.bbox), item.bbox[0])):
            for group in groups:
                if self._belongs_to_group(line, group):
                    group.append(line)
                    break
            else:
                groups.append([line])
        return groups

    def _belongs_to_group(self, line: _Line, group: List[_Line]) -> bool:
        group_bbox = self._union_bbox([item.bbox for item in group])
        line_height = max(1, line.bbox[3] - line.bbox[1])
        group_height = max(1, group_bbox[3] - group_bbox[1])
        max_gap = max(line_height, group_height) * self.line_group_max_gap_ratio
        y_gap = max(0, line.bbox[1] - group_bbox[3], group_bbox[1] - line.bbox[3])
        if y_gap > max_gap:
            return False

        overlap = max(
            0,
            min(line.bbox[2], group_bbox[2]) - max(line.bbox[0], group_bbox[0]),
        )
        min_width = max(1, min(line.bbox[2] - line.bbox[0], group_bbox[2] - group_bbox[0]))
        return (overlap / min_width) >= self.line_group_min_x_overlap_ratio

    def _suppress_existing(
        self,
        fallback_results: List[Detection],
        existing_results: List[Detection],
    ) -> List[Detection]:
        if not existing_results:
            return fallback_results
        kept = []
        for candidate in fallback_results:
            if any(bbox_iou(candidate.bbox, existing.bbox) >= self.suppress_iou_threshold for existing in existing_results):
                continue
            kept.append(candidate)
        return kept

    def _table_bbox(
        self,
        detections: List[Detection],
        image_shape: tuple[int, int],
    ) -> Optional[tuple[int, int, int, int]]:
        height, width = image_shape
        tables = [detection for detection in detections if detection.class_name == "Table" or detection.class_id == 0]
        if not tables:
            return None
        table = max(tables, key=lambda detection: self._area(detection.bbox))
        x1, y1, x2, y2 = table.bbox
        if self.table_padding_ratio > 0:
            pad_x = int(round((x2 - x1) * self.table_padding_ratio))
            pad_y = int(round((y2 - y1) * self.table_padding_ratio))
            x1 -= pad_x
            x2 += pad_x
            y1 -= pad_y
            y2 += pad_y
        x1 = max(0, min(width - 1, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height - 1, int(y1)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _box_to_bbox(box) -> Optional[tuple[int, int, int, int]]:
        arr = np.asarray(box, dtype=np.float32)
        if arr.size < 4:
            return None
        arr = arr.reshape((-1, 2))
        x1, y1 = np.floor(arr.min(axis=0)).astype(int)
        x2, y2 = np.ceil(arr.max(axis=0)).astype(int)
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _union_bbox(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    @staticmethod
    def _center_y(bbox: tuple[int, int, int, int]) -> float:
        return (float(bbox[1]) + float(bbox[3])) * 0.5

    @staticmethod
    def _area(bbox: tuple[int, int, int, int]) -> int:
        return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))

    def close(self) -> None:
        close = getattr(self.wrapped, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)
