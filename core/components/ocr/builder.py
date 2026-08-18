"""OCR 工厂：按配置 type 创建对应 OCR 实现。"""
from __future__ import annotations

from typing import Dict, Optional

from core.components.ocr.base import BaseOcr


def build_ocr(config: dict, class_registry: Optional[Dict] = None) -> BaseOcr:
    """按 config['type'] 构建 OCR；目前支持基于 PaddleOCR 的 paddle 实现。"""
    ocr_type = config["type"]

    if ocr_type == "paddle":
        from core.components.ocr.paddle_ocr import PaddleOcr

        return _maybe_throttle(
            PaddleOcr(config, class_registry),
            config,
            class_registry,
        )

    if ocr_type == "paddleocrv5":
        from core.components.ocr.paddleocrv5.component import PaddleOcrV5

        return _maybe_throttle(
            PaddleOcrV5(config, class_registry),
            config,
            class_registry,
        )

    if ocr_type == "paddleocrv6":
        from core.components.ocr.paddleocrv6.component import PaddleOcrV6

        return _maybe_throttle(
            PaddleOcrV6(config, class_registry),
            config,
            class_registry,
        )

    raise ValueError(f"unsupported ocr type: {ocr_type}")


def _maybe_throttle(
    ocr: BaseOcr,
    config: dict,
    class_registry: Optional[Dict],
) -> BaseOcr:
    throttle_config = config.get("throttle")
    if not isinstance(throttle_config, dict) or not throttle_config.get(
        "enabled",
        False,
    ):
        return ocr

    from core.components.ocr.throttle import ThrottledOcr

    return ThrottledOcr(ocr, config, class_registry)
