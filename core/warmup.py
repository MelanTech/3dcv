"""Boot-time model warmup without starting a competition round."""
from __future__ import annotations

from typing import Callable, TypeVar

from core.components.detector.builder import build_detector
from core.components.ocr.builder import build_ocr
from core.config_loader import load_config
from core.infra.logging.event_logger import EventLogger
from core.orchestration.pipeline.builder import build_detector_stage


T = TypeVar("T")


def _build_component(logger: EventLogger, name: str, factory: Callable[[], T]) -> T:
    import time

    started = time.perf_counter()
    logger.event("warmup_component_init_started", component=name)
    value = factory()
    logger.event(
        "warmup_component_init_finished",
        component=name,
        elapsed_sec=round(time.perf_counter() - started, 3),
    )
    return value


def _close_component(logger: EventLogger, name: str, component) -> None:
    close = getattr(component, "close", None)
    if close is None:
        return
    try:
        close()
        logger.event("warmup_resource_closed", resource=name)
    except Exception as exc:
        logger.event(
            "warmup_resource_close_failed",
            resource=name,
            error=str(exc),
            exc_type=type(exc).__name__,
        )


def run_warmup(config_path: str, round_name: str) -> None:
    """Load model resources once so later runtime startup benefits from OS caches."""
    config = load_config(config_path)
    logging_config = config["logging"]
    detector = None
    ocr = None
    detector_stage = None

    with EventLogger(
        logging_config["base_dir"],
        f"{round_name}_warmup",
        console=logging_config.get("console", True),
    ) as logger:
        logger.event("warmup_started", path=config_path, round=round_name)
        try:
            detector = _build_component(
                logger,
                "detector",
                lambda: build_detector(
                    config["detector"],
                    round_name,
                    config.get("class_registry"),
                ),
            )
            ocr = _build_component(
                logger,
                "ocr",
                lambda: build_ocr(config["ocr"], config.get("class_registry")),
            )
            detector_stage = _build_component(
                logger,
                "detector_stage",
                lambda: build_detector_stage(
                    detector=detector,
                    config=config.get("pipeline", {}).get("async_detector"),
                    detector_config=config["detector"],
                    class_registry=config.get("class_registry"),
                    round_name=round_name,
                    logger=logger,
                ),
            )
            logger.event("warmup_finished", round=round_name)
        finally:
            for name, component in (
                ("detector_stage", detector_stage),
                ("ocr", ocr),
                ("detector", detector),
            ):
                if component is not None:
                    _close_component(logger, name, component)
