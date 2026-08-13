"""Composite unknown merger provider."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from core.components.unknown_merger.base import BaseUnknownMerger
from core.types import Detection, Frame
from core.utils.box import max_iou, nms_detections


class MergedUnknownMerger(BaseUnknownMerger):
    """Merges candidate boxes from one or more unknown merger providers."""

    def __init__(
        self,
        providers: Iterable[BaseUnknownMerger],
        result_classes: Iterable[str],
        candidate_iou_threshold: float,
        known_overlap_iou_threshold: float,
        suppress_source_classes: Iterable[str],
        source_overlap_iou_threshold: float,
    ):
        self.providers = list(providers)
        self.result_classes = set(result_classes)
        self.candidate_iou_threshold = float(candidate_iou_threshold)
        self.known_overlap_iou_threshold = float(known_overlap_iou_threshold)
        self.suppress_source_classes = set(suppress_source_classes)
        self.source_overlap_iou_threshold = float(source_overlap_iou_threshold)

    def infer(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        candidates: List[Detection] = []
        for provider in self.providers:
            candidates.extend(provider.infer(frame, detections, table))

        if not candidates:
            return []

        known_detections = [
            detection
            for detection in detections
            if detection.class_name in self.result_classes
        ]
        if self.known_overlap_iou_threshold > 0.0 and known_detections:
            candidates = [
                candidate
                for candidate in candidates
                if max_iou(candidate, known_detections) < self.known_overlap_iou_threshold
            ]

        if self.candidate_iou_threshold > 0.0:
            candidates = nms_detections(candidates, self.candidate_iou_threshold)
        candidates = self._annotate_source_suppression(candidates, detections)
        return candidates

    def close(self) -> None:
        for provider in self.providers:
            provider.close()

    def _annotate_source_suppression(
        self,
        candidates: List[Detection],
        detections: List[Detection],
    ) -> List[Detection]:
        """Mark candidates that should suppress overlapping detector source boxes."""
        if self.source_overlap_iou_threshold <= 0.0 or not self.suppress_source_classes:
            return candidates

        source_detections = [
            detection
            for detection in detections
            if detection.class_name in self.suppress_source_classes
        ]
        if not source_detections:
            return candidates

        annotated: List[Detection] = []
        for candidate in candidates:
            if max_iou(candidate, source_detections) < self.source_overlap_iou_threshold:
                annotated.append(candidate)
                continue

            evidence = dict(candidate.evidence)
            evidence["suppress_source_classes"] = sorted(self.suppress_source_classes)
            evidence["suppress_source_iou_threshold"] = self.source_overlap_iou_threshold
            annotated.append(replace(candidate, evidence=evidence))
        return annotated
