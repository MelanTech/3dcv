"""过滤器工厂：按配置 type 创建对应过滤器。"""
from __future__ import annotations

from typing import Any, Optional

from core.components.filter.base import BaseFilter


def build_filter(config: Any, intrinsic_override: Optional[dict] = None) -> BaseFilter:
    """按 config['type'] 构建过滤器；目前支持深度过滤器。"""
    config = _unwrap_filter_config(config)
    filter_type = config["type"]

    if filter_type == "depth":
        from core.components.filter.depth_filter import DepthFilter

        return DepthFilter(config, intrinsic_override=intrinsic_override)

    raise ValueError(f"unsupported filter type: {filter_type}")


def _unwrap_filter_config(config: Any) -> dict:
    """Accept both ``{filter: {...}}`` include output and raw filter bodies."""
    if not isinstance(config, dict):
        raise ValueError("filter config must be a mapping")
    if set(config) == {"filter"}:
        return dict(config["filter"])
    return dict(config)
