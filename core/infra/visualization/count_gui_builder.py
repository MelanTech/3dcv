"""Factory for the optional Tk live-count panel."""
from __future__ import annotations

from typing import Dict, Optional

from core.infra.logging.event_logger import EventLogger
from core.infra.visualization.count_gui import BaseCountGui, NoopCountGui, TkCountGui


def build_count_gui(
    config: Optional[Dict],
    class_registry: Optional[Dict],
    logger: EventLogger,
) -> BaseCountGui:
    """Build the live count panel or an inert implementation when disabled."""
    if not config or not config.get("enabled", False):
        return NoopCountGui()

    try:
        gui = TkCountGui(
            config=config,
            count_rows=_count_rows(class_registry or {}),
        )
    except Exception as exc:
        logger.event(
            "count_gui_disabled",
            reason="initialization_failed",
            error=str(exc),
            exc_type=type(exc).__name__,
        )
        return NoopCountGui()

    logger.event("count_gui_enabled")
    return gui


def _count_rows(class_registry: Dict) -> list[tuple[tuple[str, ...], str]]:
    """Use English class names for normal items and keep Wxxx unknown codes."""
    result_classes = [str(value) for value in class_registry.get("result_classes", ())]
    result_class_to_goal_id = {
        str(class_name): str(goal_id)
        for class_name, goal_id in dict(
            class_registry.get("result_class_to_goal_id", {})
        ).items()
    }
    rows: list[tuple[tuple[str, ...], str]] = []
    for class_name in result_classes:
        goal_id = result_class_to_goal_id.get(class_name, class_name)
        display_name = class_name
        if class_name == goal_id and class_name.startswith("W"):
            display_name = goal_id
        keys = (class_name,) if goal_id == class_name else (class_name, goal_id)
        rows.append((keys, display_name))
    return rows
