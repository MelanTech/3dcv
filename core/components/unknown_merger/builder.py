"""Factory for unknown OCR candidate mergers."""
from __future__ import annotations

from typing import Dict, Optional

from core.components.unknown_merger.base import (
    BaseUnknownMerger,
    DisabledUnknownMerger,
)
from core.components.unknown_merger.detector_class_provider import (
    DetectorClassUnknownMerger,
)
from core.components.unknown_merger.merged_provider import MergedUnknownMerger
from core.components.unknown_merger.open_set_detector_provider import (
    OpenSetDetectorUnknownMerger,
)


def build_unknown_merger(
    config: Optional[Dict],
    class_registry: Optional[Dict] = None,
) -> BaseUnknownMerger:
    """Build the configured merger for OCR unknown candidates."""
    if not config or not config.get("enabled", False):
        return DisabledUnknownMerger()

    class_registry = class_registry or {}
    mode = str(config.get("mode", "detector_class")).strip().lower()
    if mode == "disabled":
        return DisabledUnknownMerger()
    if mode not in ("detector_class", "open_set", "both"):
        raise ValueError(
            "unknown_merger.mode must be detector_class, open_set, both, or disabled"
        )

    output_class = str(config.get("output_class", "UnknownOcrCandidate"))
    providers = []

    if mode in ("detector_class", "both"):
        providers.append(
            DetectorClassUnknownMerger(
                config.get("detector_class", {}),
                output_class=output_class,
            )
        )

    if mode in ("open_set", "both"):
        open_set_config = config.get("open_set", {})
        providers.append(
            OpenSetDetectorUnknownMerger(
                open_set_config,
                output_class=output_class,
                class_registry=class_registry,
            )
        )

    merge_config = config.get("merge", {})
    suppress_source_classes = merge_config.get(
        "suppress_source_classes",
        config.get("detector_class", {}).get("classes", ["Book"]),
    )
    return MergedUnknownMerger(
        providers=providers,
        result_classes=class_registry.get("result_classes", []),
        candidate_iou_threshold=float(merge_config.get("candidate_iou_threshold", 0.6)),
        known_overlap_iou_threshold=float(merge_config.get("known_overlap_iou_threshold", 0.5)),
        suppress_source_classes=suppress_source_classes,
        source_overlap_iou_threshold=float(merge_config.get("source_overlap_iou_threshold", 0.6)),
    )
