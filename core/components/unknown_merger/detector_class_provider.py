"""Unknown OCR candidates derived from classes emitted by the main detector."""
from __future__ import annotations

from typing import Dict, List

from core.components.unknown_merger.base import BaseUnknownMerger
from core.types import Detection, Frame


class DetectorClassUnknownMerger(BaseUnknownMerger):
    """Maps configured detector classes, such as Book, into OCR candidate boxes."""

    def __init__(self, config: Dict, output_class: str):
        self.enabled = bool(config.get("enabled", True))
        self.classes = set(str(value) for value in config.get("classes", ["Book"]))
        self.output_class = str(output_class)

    def infer(
        self,
        _frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        if not self.enabled:
            return []

        candidates: List[Detection] = []
        for detection in detections:
            if detection.class_name not in self.classes:
                continue
            evidence = dict(detection.evidence)
            evidence.update(
                {
                    "source": "unknown_merger.detector_class",
                    "source_class": detection.class_name,
                    "table": table,
                }
            )
            candidates.append(
                Detection(
                    class_name=self.output_class,
                    class_id=-1,
                    bbox=detection.bbox,
                    score=detection.score,
                    evidence=evidence,
                )
            )
        return candidates
