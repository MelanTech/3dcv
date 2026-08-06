"""平台探测工具：判断当前是否运行在香橙派（用于自动选择推理后端）。"""
from __future__ import annotations

import os
import platform as platform_module
from pathlib import Path


def current_platform() -> str:
    """返回当前运行平台标识；可用 3DCV_PLATFORM 覆盖，便于离线调试。"""
    platform_override = os.environ.get("3DCV_PLATFORM", "").strip().lower()
    if platform_override:
        return _normalize_platform(platform_override)

    if is_orangepi():
        return "orangepi"

    system = platform_module.system().strip().lower()
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    if system == "windows":
        return "windows"
    return system or "unknown"


def is_orangepi() -> bool:
    """判断是否为香橙派：优先看环境变量覆盖，否则读设备树 model 信息。"""
    platform_override = os.environ.get("3DCV_PLATFORM", "").strip().lower()
    if platform_override:
        return _normalize_platform(platform_override) == "orangepi"

    model_path = Path("/proc/device-tree/model")
    try:
        model_name = model_path.read_text(encoding="utf-8").strip("\x00").lower()
    except (OSError, UnicodeError):
        return False
    return "orange pi" in model_name or "orangepi" in model_name


def _normalize_platform(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "darwin": "macos",
        "mac": "macos",
        "osx": "macos",
        "orange-pi": "orangepi",
        "orangepi-aipro": "orangepi",
        "orange-pi-aipro": "orangepi",
        "orange-pi-ai-pro": "orangepi",
    }
    return aliases.get(normalized, normalized)
