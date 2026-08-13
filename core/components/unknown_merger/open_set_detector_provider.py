"""Unknown-merger adapter for configurable open-set detectors."""
from __future__ import annotations

from typing import Dict, List, Optional

from core.components.detector.builder import build_detector
from core.components.detector.base import BaseDetector
from core.components.unknown_merger.base import BaseUnknownMerger
from core.types import Detection, Frame


class OpenSetDetectorUnknownMerger(BaseUnknownMerger):
    """Runs a configured detector as an OCR candidate source."""

    def __init__(
        self,
        config: Dict,
        output_class: str,
        class_registry: Optional[Dict] = None,
    ):
        self.enabled = bool(config.get("enabled", False))
        self.detector: BaseDetector | None = None
        if self.enabled:
            detector_config = self._resolve_detector_config(config, output_class)
            self.detector = build_detector(
                detector_config,
                _round_name="unknown_merger",
                class_registry=class_registry,
            )

    def infer(
        self,
        frame: Frame,
        _detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        if self.detector is None:
            return []
        return self.detector.infer(frame, table)

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    @staticmethod
    def _resolve_detector_config(config: Dict, output_class: str) -> Dict:
        raw_detector_config = config.get("detector")
        if raw_detector_config is None:
            # Backward-compatible flat open_set config.
            raw_detector_config = {
                key: value
                for key, value in config.items()
                if key not in {"enabled", "detector"}
            }

        if "detector" in raw_detector_config:
            detector_config = dict(raw_detector_config["detector"])
        else:
            detector_config = dict(raw_detector_config)

        detector_config.setdefault("output_class", output_class)
        return detector_config
