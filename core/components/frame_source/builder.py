"""帧源工厂：按配置选择离线图片序列或 OpenNI 相机，并做轮次配置解析。"""
from __future__ import annotations

from copy import deepcopy

from core.components.frame_source.base import BaseFrameSource
from core.utils.platform import current_platform


def build_frame_source(config: dict, round_name: str) -> BaseFrameSource:
    """先按轮次解析出生效配置，再按 type 构建对应帧源。"""
    config = _resolve_round_config(config, round_name)
    frame_source_type = config["type"]

    if frame_source_type == "image_sequence":
        from core.components.frame_source.image_sequence_reader import ImageSequenceFrameSource

        return ImageSequenceFrameSource(config)

    if frame_source_type == "openni":
        from core.components.frame_source.openni_reader import OpenNIFrameSource

        return OpenNIFrameSource(
            openni_lib=config.get("openni_lib"),
            width=config["width"],
            height=config["height"],
            fps=config["fps"],
            color_source=config["color_source"],
            uvc_device=config["uvc_device"],
            allow_unregistered=config["allow_unregistered"],
            mirror=config["mirror"],
            d2c=config.get("d2c"),
            sync=config.get("sync"),
        )

    if frame_source_type == "orbbecsdk":
        from core.components.frame_source.orbbecsdk_reader import OrbbecSdkFrameSource

        return OrbbecSdkFrameSource(config)

    raise ValueError(f"unsupported frame_source type: {frame_source_type}")


def _resolve_round_config(config: dict, round_name: str) -> dict:
    """把 common 配置与该轮次专属配置合并成最终生效的帧源配置。"""
    if "rounds" not in config:
        return _resolve_platform_base_path(deepcopy(config))

    rounds = config["rounds"]
    if round_name not in rounds:
        raise ValueError(f"frame_source.rounds does not define {round_name}")

    resolved = {
        "type": config["type"],
    }
    if "base_path" in config:
        resolved["base_path"] = config["base_path"]
    if "base_paths" in config:
        resolved["base_paths"] = deepcopy(config["base_paths"])
    resolved.update(deepcopy(config.get("common", {})))
    resolved.update(deepcopy(rounds[round_name]))
    return _resolve_platform_base_path(resolved)


def _resolve_platform_base_path(config: dict) -> dict:
    """把 base_paths 平台映射解析成单个 base_path；兼容旧的 base_path 写法。"""
    base_paths = config.pop("base_paths", None)
    if base_paths is None:
        return config
    if not isinstance(base_paths, dict):
        raise ValueError("frame_source.base_paths must be a mapping")

    platform_name = current_platform()
    normalized_paths = {
        str(platform).strip().lower().replace("_", "-"): path
        for platform, path in base_paths.items()
    }
    selected = (
        normalized_paths.get(platform_name)
        or normalized_paths.get("default")
        or config.get("base_path")
    )
    if selected is None:
        supported = ", ".join(sorted(normalized_paths))
        raise ValueError(
            f"frame_source.base_paths does not define current platform '{platform_name}' "
            f"and has no default/base_path fallback; configured: {supported}"
        )

    config["base_path"] = selected
    return config
