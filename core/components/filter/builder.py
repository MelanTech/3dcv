"""过滤器工厂：按配置 type 创建对应过滤器。"""
from __future__ import annotations

from typing import Optional

from core.components.filter.base import BaseFilter


def build_filter(config: dict, intrinsic_override: Optional[dict] = None) -> BaseFilter:
    """按 config['type'] 构建过滤器；目前仅支持深度过滤器。"""
    filter_type = config["type"]

    if filter_type == "depth":
        from core.components.filter.depth_filter import DepthFilter

        return DepthFilter(config, intrinsic_override=intrinsic_override)

    raise ValueError(f"unsupported filter type: {filter_type}")
