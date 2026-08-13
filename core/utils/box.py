"""Bounding-box utilities shared by post-processing components."""
from __future__ import annotations

from typing import Iterable, List

from core.types import Detection


def bbox_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Return IoU for two xyxy pixel boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area) / float(union)


def max_iou(
    detection: Detection,
    others: Iterable[Detection],
) -> float:
    """Return the maximum IoU between one detection and an iterable of detections."""
    best = 0.0
    for other in others:
        best = max(best, bbox_iou(detection.bbox, other.bbox))
    return best


def nms_detections(
    detections: Iterable[Detection],
    iou_threshold: float,
) -> List[Detection]:
    """Greedy NMS for Detection lists, keeping higher score boxes first."""
    kept: List[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(bbox_iou(detection.bbox, existing.bbox) < iou_threshold for existing in kept):
            kept.append(detection)
    return kept
