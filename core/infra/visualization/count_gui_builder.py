"""Factory for the optional Tk live-count panel."""

from __future__ import annotations

from core.infra.logging.event_logger import EventLogger
from core.infra.visualization.count_gui import BaseCountGui, NoopCountGui, TkCountGui


def build_count_gui(
    config: dict | None,
    class_registry: dict | None,
    logger: EventLogger,
) -> BaseCountGui:
    """Build the live count panel or an inert implementation when disabled."""
    if not config or not config.get("enabled", False):
        return NoopCountGui()

    try:
        registry = class_registry or {}
        gui = TkCountGui(
            config=config,
            count_rows=_count_rows(registry),
            bbox_colors=_bbox_colors(registry),
        )
    except Exception as exc:  # noqa: BLE001 - optional GUI must fail closed
        logger.event(
            "count_gui_disabled",
            reason="initialization_failed",
            error=str(exc),
            exc_type=type(exc).__name__,
        )
        return NoopCountGui()

    logger.event("count_gui_enabled")
    return gui


def _count_rows(class_registry: dict) -> list[tuple[tuple[str, ...], str]]:
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


def _bbox_colors(class_registry: dict) -> dict[str, tuple[int, int, int]]:
    """Load RGB bbox colors for the RGB preview."""
    colors = {}
    for class_name, value in dict(class_registry.get("bbox_colors", {})).items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        colors[str(class_name)] = tuple(
            int(max(0, min(255, channel))) for channel in value
        )
    return colors
