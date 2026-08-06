"""OrbbecSDK v1 实时帧源：通过 native 扩展采集硬件 D2C RGB-D 帧。"""
from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Dict, Optional

from core.components.frame_source.base import BaseFrameSource
from core.types import Frame


class OrbbecSdkFrameSource(BaseFrameSource):
    """OrbbecSDK v1 帧源：native 层负责取帧、硬件 D2C 和 ApproximateTime 配对。"""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.sdk_root = str(
            Path(self.config.get("sdk_root", "third_party/orbbecsdk/v1.10.27"))
        )
        self.width = int(self.config.get("width", 640))
        self.height = int(self.config.get("height", 480))
        self.fps = int(self.config.get("fps", 30))
        self.align_config = dict(self.config.get("align", {}))
        self.sync_config = dict(self.config.get("sync", {}))
        self.align_mode = str(self.align_config.get("mode", "hardware"))
        self.enable_frame_sync = bool(self.sync_config.get("enable_frame_sync", False))
        self.approx_sync_ms = float(self.sync_config.get("approx_sync_ms", 50.0))
        self.max_queue_size = int(self.sync_config.get("max_queue_size", 10))
        self.wait_timeout_ms = int(self.sync_config.get("wait_timeout_ms", 1000))

        native = importlib.import_module(
            "core.components.frame_source.native.orbbecsdk_native"
        )
        self.reader = native.OrbbecSdkReader(
            sdk_root=self.sdk_root,
            width=self.width,
            height=self.height,
            fps=self.fps,
            align_mode=self.align_mode,
            enable_frame_sync=self.enable_frame_sync,
            approx_sync_ms=self.approx_sync_ms,
            max_queue_size=self.max_queue_size,
            wait_timeout_ms=self.wait_timeout_ms,
        )
        self._idx = 0
        self._closed = False
        self.filter_intrinsic = self._read_intrinsic()
        self.filter_intrinsic_source = "orbbecsdk_d2c"
        self.filter_intrinsic_error: Optional[str] = None

    def __iter__(self) -> "OrbbecSdkFrameSource":
        return self

    def __next__(self) -> Frame:
        if self._closed:
            raise StopIteration

        sample = self.reader.next()
        self._idx += 1
        return Frame(
            frame_id=f"orbbecsdk_{self._idx:06d}",
            rgb=sample["rgb"],
            depth=sample["depth"],
            timestamp=time.time(),
        )

    def _read_intrinsic(self) -> Dict[str, float]:
        intrinsic = self.reader.get_intrinsic()
        return {
            "width": int(intrinsic["width"]),
            "height": int(intrinsic["height"]),
            "fx": float(intrinsic["fx"]),
            "fy": float(intrinsic["fy"]),
            "cx": float(intrinsic["cx"]),
            "cy": float(intrinsic["cy"]),
        }

    def get_filter_intrinsic(self) -> Optional[Dict[str, float]]:
        return self.filter_intrinsic

    def get_filter_intrinsic_source(self) -> str:
        return self.filter_intrinsic_source

    def get_sync_stats(self) -> dict:
        get_stats = getattr(self.reader, "get_stats", None)
        return {} if get_stats is None else dict(get_stats())

    def close(self) -> None:
        self._closed = True
        if getattr(self, "reader", None) is not None:
            self.reader.close()
