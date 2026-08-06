"""桌面定位器工厂：按配置 type 创建对应实现。"""
from __future__ import annotations

from core.components.table_locator.base import BaseTableLocator


def build_table_locator(config: dict) -> BaseTableLocator:
    """按 config['type'] 构建桌面定位器。"""
    table_locator_type = config["type"]

    if table_locator_type == "table_locator":
        from core.components.table_locator.table_locator import TableLocator

        return TableLocator(config)

    raise ValueError(f"unsupported table_locator type: {table_locator_type}")
