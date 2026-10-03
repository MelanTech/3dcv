"""Assembly of detector-stage implementations for the runtime pipeline."""
from __future__ import annotations

import sys
from typing import Dict, Optional

from core.components.detector.base import BaseDetector
from core.infra.logging.event_logger import EventLogger
from core.orchestration.pipeline.detector_stage import (
    BaseDetectorStage,
    InlineDetectorStage,
)
from core.orchestration.pipeline.process_detector_stage import ProcessDetectorStage
from core.orchestration.pipeline.table_roi_detector_stage import TableRoiDetectorStage
from core.utils.platform import is_orangepi


def build_detector_stage(
    detector: BaseDetector,
    config: Optional[Dict],
    roi_config: Optional[Dict],
    detector_config: Dict,
    class_registry: Optional[Dict],
    round_name: str,
    logger: EventLogger,
) -> BaseDetectorStage:
    """Build the configured detector execution stage with explicit fallback logs."""
    async_config = dict(config or {})
    if not async_config.get("enabled", False):
        stage: BaseDetectorStage = InlineDetectorStage(detector)
    elif sys.platform != "linux" or not is_orangepi():
        logger.event(
            "detector_stage_fallback",
            requested="process",
            selected="inline",
            reason="unsupported_platform",
        )
        stage = InlineDetectorStage(detector)
    else:
        logger.event("detector_stage_selected", selected="process")
        stage = ProcessDetectorStage(
            detector_config=detector_config,
            class_registry=class_registry,
            round_name=round_name,
        )

    resolved_roi_config = dict(roi_config or {})
    if not resolved_roi_config.get("enabled", False):
        return stage
    logger.event("detection_roi_stage_enabled")
    return TableRoiDetectorStage(
        wrapped=stage,
        config=resolved_roi_config,
        logger=logger,
    )
