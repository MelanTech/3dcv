"""OpenNI 实时相机帧源：驱动 Astra 类深度相机采集 RGB/深度帧。

彩色源可来自 OpenNI 自身或外接 UVC 摄像头；深度到彩色对齐支持硬件
（相机内部 registration）或关闭两种模式。
"""
from __future__ import annotations

from collections import deque
import re
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import cv2
import numpy as np
from openni import _openni2 as c_api
from openni import openni2

from core.components.frame_source.base import BaseFrameSource
from core.types import Frame


DEFAULT_OPENNI_ENV_VARS = ("OPENNI2_REDIST", "OPENNI2_LIB")
COMMON_OPENNI_DIRS = (
    "/usr/lib",
    "/usr/local/lib",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/local/OpenNI2/Redist",
    "/opt/OpenNI2/Redist",
    "/opt/openni2/Redist",
    "/usr/lib/OpenNI2/Redist",
)


class OpenNIFrameSource(BaseFrameSource):
    """OpenNI 相机帧源：初始化设备/数据流，每次产出一帧对齐的 RGB/深度。"""

    def __init__(
        self,
        openni_lib: Optional[str],
        width: int,
        height: int,
        fps: int,
        color_source: str,
        uvc_device: str,
        allow_unregistered: bool,
        mirror: bool,
        d2c: Optional[dict] = None,
        sync: Optional[dict] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.color_source = color_source
        self.uvc_device = uvc_device
        self.allow_unregistered = bool(allow_unregistered)
        self.mirror = bool(mirror)
        self.d2c_config = dict(d2c or {})
        self.sync_config = dict(sync or {})
        self.d2c_mode = self.d2c_config.get("mode", "hardware")
        if self.d2c_mode not in ("hardware", "off"):
            raise ValueError("frame_source.d2c.mode must be hardware or off")
        self.uvc_latest_thread_enabled = bool(
            self.sync_config.get("uvc_latest_thread", False)
        )
        self.approx_sync_ms = float(self.sync_config.get("approx_sync_ms", 0.0) or 0.0)
        self.approx_sync_enabled = self.approx_sync_ms > 0.0
        self.approx_sync_threshold_ns = int(self.approx_sync_ms * 1_000_000)
        self.approx_sync_max_queue = max(
            2, int(self.sync_config.get("approx_sync_max_queue", 10) or 10)
        )
        self.approx_sync_fallback_latest = bool(
            self.sync_config.get("approx_sync_fallback_latest", True)
        )

        self.device = None
        self.depth_stream = None
        self.color_stream = None
        self.uvc_capture = None
        self._pending_uvc_frame = None
        self._uvc_lock = threading.Lock()
        self._uvc_stop = threading.Event()
        self._uvc_thread: Optional[threading.Thread] = None
        self._latest_uvc_frame = None
        self._latest_uvc_timestamp_ns: Optional[int] = None
        self._uvc_queue = deque(maxlen=self.approx_sync_max_queue)
        self._latest_uvc_error: Optional[str] = None
        self._closed = False
        self._idx = 0
        self._sync_matched = 0
        self._sync_missed = 0
        self._sync_dropped = 0
        self.registration_enabled = False
        self.filter_intrinsic: Optional[Dict[str, float]] = None
        self.filter_intrinsic_source = "config"
        self.filter_intrinsic_error: Optional[str] = None

        try:
            self._initialize(openni_lib)
        except Exception:
            self.close()
            raise

    def _candidate_openni_paths(self, cli_path: Optional[str]) -> Iterable[Optional[str]]:
        if cli_path:
            yield cli_path

        import os

        for env_name in DEFAULT_OPENNI_ENV_VARS:
            value = os.environ.get(env_name)
            if value:
                yield value

        yield None

        for path in COMMON_OPENNI_DIRS:
            yield path

    def _initialize_openni(self, cli_path: Optional[str]) -> str:
        """依次尝试各候选路径初始化 OpenNI2，全部失败则汇总错误抛出。"""
        errors = []
        seen: Set[Optional[str]] = set()

        for path in self._candidate_openni_paths(cli_path):
            if path in seen:
                continue
            seen.add(path)

            if path is not None and not Path(path).exists():
                errors.append(f"{path}: path does not exist")
                continue

            try:
                if path is None:
                    openni2.initialize()
                    return "system library path"
                openni2.initialize(path)
                return path
            except Exception as exc:
                label = path if path is not None else "system library path"
                errors.append(f"{label}: {exc}")

        raise RuntimeError("Failed to initialize OpenNI2: " + " | ".join(errors))

    def _make_video_mode(self, pixel_format: int):
        return c_api.OniVideoMode(
            pixelFormat=pixel_format,
            resolutionX=self.width,
            resolutionY=self.height,
            fps=self.fps,
        )

    def _set_mirror(self, stream) -> None:
        try:
            stream.set_mirroring_enabled(self.mirror)
        except Exception:
            pass

    def _initialize(self, openni_lib: Optional[str]) -> None:
        """打开设备、创建并配置深度/彩色数据流，设置 D2C 对齐并启动采集。"""
        self._initialize_openni(openni_lib)
        self.device = openni2.Device.open_any()

        self.depth_stream = self.device.create_depth_stream()
        if self.depth_stream is None:
            raise RuntimeError("OpenNI device does not expose a depth sensor")
        self.depth_stream.set_video_mode(
            self._make_video_mode(c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM)
        )
        self._set_mirror(self.depth_stream)

        color_source = self.color_source
        if color_source in ("auto", "openni"):
            self.color_stream = self.device.create_color_stream()
            if self.color_stream is not None:
                self.color_stream.set_video_mode(
                    self._make_video_mode(c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888)
                )
                self._set_mirror(self.color_stream)
                color_source = "openni"
            elif color_source == "openni":
                raise RuntimeError("OpenNI device does not expose a color sensor")

        if color_source == "auto":
            color_source = "uvc"

        if color_source == "uvc":
            self.uvc_capture = cv2.VideoCapture(
                self._parse_uvc_device(self.uvc_device),
                cv2.CAP_V4L2,
            )
            if not self.uvc_capture.isOpened():
                raise RuntimeError(f"Could not open UVC RGB frame source: {self.uvc_device}")
            self.uvc_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.uvc_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.uvc_capture.set(cv2.CAP_PROP_FPS, self.fps)
            ok, frame = self.uvc_capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Opened UVC RGB frame source but could not read a frame: {self.uvc_device}")
            if self.uvc_latest_thread_enabled:
                with self._uvc_lock:
                    timestamp_ns = time.monotonic_ns()
                    self._latest_uvc_frame = frame
                    self._latest_uvc_timestamp_ns = timestamp_ns
                    self._uvc_queue.append((timestamp_ns, frame.copy()))
                self._start_uvc_latest_thread()
            else:
                self._pending_uvc_frame = (time.monotonic_ns(), frame)
        elif color_source == "none":
            pass

        if self.d2c_mode == "hardware":
            self.registration_enabled = self._enable_registration()
        if self.color_stream is not None:
            self._enable_depth_color_sync()
        self._load_filter_intrinsic()

        self.depth_stream.start()
        if self.color_stream is not None:
            self.color_stream.start()

    def _parse_uvc_device(self, device):
        """OpenCV V4L2 on the target board opens /dev/videoN more reliably as index N."""
        text = str(device)
        if text.isdigit():
            return int(text)
        match = re.fullmatch(r"/dev/video(\d+)", text)
        if match is not None:
            return int(match.group(1))
        return device

    def _enable_registration(self) -> bool:
        mode = c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR
        try:
            supported = self.device.is_image_registration_mode_supported(mode)
        except Exception:
            supported = False

        if not supported:
            if self.allow_unregistered:
                return False
            raise RuntimeError("OpenNI reports DEPTH_TO_COLOR registration is not supported")

        self.device.set_image_registration_mode(mode)
        return True

    def _enable_depth_color_sync(self) -> None:
        try:
            self.device.set_depth_color_sync_enabled(True)
        except Exception:
            pass

    def _load_filter_intrinsic(self) -> None:
        """从 Orbbec 设备读取内参，供 DepthFilter 覆盖静态配置使用。"""
        try:
            from scripts.orbbec_calibration import read_orbbec_camera_params

            params = read_orbbec_camera_params(
                device=self.device,
                width=self.width,
                height=self.height,
            )
            profile = "color" if self.registration_enabled else "depth"
            model = params.color if profile == "color" else params.depth
            self.filter_intrinsic = {
                "width": int(model.width),
                "height": int(model.height),
                "fx": float(model.fx),
                "fy": float(model.fy),
                "cx": float(model.cx),
                "cy": float(model.cy),
            }
            self.filter_intrinsic_source = f"openni_orbbec_{profile}"
        except Exception as exc:
            self.filter_intrinsic = None
            self.filter_intrinsic_source = "config"
            self.filter_intrinsic_error = str(exc)

    def __iter__(self) -> "OpenNIFrameSource":
        return self

    def __next__(self) -> Frame:
        """读取一帧深度和彩色数据，按需软件对齐后封装成 Frame。"""
        if self._closed:
            raise StopIteration

        depth_frame = self.depth_stream.read_frame()
        depth_timestamp_ns = time.monotonic_ns()
        depth = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
        depth = depth.reshape((depth_frame.height, depth_frame.width))

        if self.color_stream is not None:
            color_frame = self.color_stream.read_frame()
            rgb = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
            rgb = rgb.reshape((color_frame.height, color_frame.width, 3))
        elif self.uvc_capture is not None:
            bgr = self._read_uvc_frame(depth_timestamp_ns)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.zeros((depth_frame.height, depth_frame.width, 3), dtype=np.uint8)

        self._idx += 1
        return Frame(
            frame_id=f"openni_{self._idx:06d}",
            rgb=rgb,
            depth=depth,
            timestamp=time.time(),
        )

    def _start_uvc_latest_thread(self) -> None:
        """后台持续读取 UVC，只保留最新 RGB，避免主线程处理慢导致队列积压。"""
        self._uvc_thread = threading.Thread(
            target=self._uvc_latest_loop,
            name="uvc-latest-frame",
            daemon=True,
        )
        self._uvc_thread.start()

    def _uvc_latest_loop(self) -> None:
        while not self._uvc_stop.is_set():
            ok, frame = self.uvc_capture.read()
            if not ok or frame is None:
                with self._uvc_lock:
                    self._latest_uvc_error = "Failed to read RGB frame from UVC frame source"
                time.sleep(0.01)
                continue

            timestamp_ns = time.monotonic_ns()
            with self._uvc_lock:
                self._latest_uvc_frame = frame
                self._latest_uvc_timestamp_ns = timestamp_ns
                if self.approx_sync_enabled:
                    self._uvc_queue.append((timestamp_ns, frame.copy()))
                self._latest_uvc_error = None

    def _read_uvc_frame(self, depth_timestamp_ns: int):
        if not self.uvc_latest_thread_enabled:
            if self._pending_uvc_frame is not None:
                _, bgr = self._pending_uvc_frame
                self._pending_uvc_frame = None
                return bgr

            ok, bgr = self.uvc_capture.read()
            if not ok or bgr is None:
                raise RuntimeError("Failed to read RGB frame from UVC frame source")
            return bgr

        with self._uvc_lock:
            frame = self._select_approx_uvc_frame_locked(depth_timestamp_ns)
            error = self._latest_uvc_error

        if frame is None:
            raise RuntimeError(error or "UVC latest frame is not ready")
        return frame

    def _select_approx_uvc_frame_locked(self, depth_timestamp_ns: int):
        if not self.approx_sync_enabled:
            return None if self._latest_uvc_frame is None else self._latest_uvc_frame.copy()

        if not self._uvc_queue:
            self._sync_missed += 1
            return None if self._latest_uvc_frame is None else self._latest_uvc_frame.copy()

        best_index = 0
        best_delta = abs(self._uvc_queue[0][0] - depth_timestamp_ns)
        for index, (timestamp_ns, _) in enumerate(self._uvc_queue):
            delta = abs(timestamp_ns - depth_timestamp_ns)
            if delta < best_delta:
                best_index = index
                best_delta = delta

        if best_delta <= self.approx_sync_threshold_ns:
            timestamp_ns, frame = self._uvc_queue[best_index]
            while self._uvc_queue and self._uvc_queue[0][0] <= timestamp_ns:
                self._uvc_queue.popleft()
            self._sync_matched += 1
            return frame.copy()

        while (
            self._uvc_queue
            and self._uvc_queue[0][0] + self.approx_sync_threshold_ns < depth_timestamp_ns
        ):
            self._uvc_queue.popleft()
            self._sync_dropped += 1

        self._sync_missed += 1
        if self.approx_sync_fallback_latest:
            return None if self._latest_uvc_frame is None else self._latest_uvc_frame.copy()
        return None

    def get_filter_intrinsic(self) -> Optional[Dict[str, float]]:
        """实时相机模式下返回从设备读取到的过滤内参。"""
        return self.filter_intrinsic

    def get_filter_intrinsic_source(self) -> str:
        return self.filter_intrinsic_source

    def close(self) -> None:
        self._closed = True
        if self._uvc_thread is not None:
            self._uvc_stop.set()
            self._uvc_thread.join(timeout=1.0)
            self._uvc_thread = None
        for stream in (self.depth_stream, self.color_stream):
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception:
                pass
        if self.uvc_capture is not None:
            self.uvc_capture.release()
