#!/usr/bin/env python3
"""Preview RGB camera frames without detector, depth, or pipeline overlays."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview only the RGB camera image for local debugging.",
    )
    parser.add_argument(
        "--source",
        choices=("frame-source", "uvc"),
        default="frame-source",
        help=(
            "frame-source uses the project frame_source config; "
            "uvc opens a plain OpenCV VideoCapture device."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="project config used when --source=frame-source",
    )
    parser.add_argument(
        "--round",
        default="round1",
        help="round name used to resolve frame_source.rounds",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="UVC device index/path used when --source=uvc, e.g. 0 or /dev/video0",
    )
    parser.add_argument("--width", type=int, default=None, help="optional UVC width")
    parser.add_argument("--height", type=int, default=None, help="optional UVC height")
    parser.add_argument("--fps", type=int, default=None, help="optional UVC FPS")
    parser.add_argument("--mirror", action="store_true", help="mirror preview horizontally")
    parser.add_argument("--show-fps", action="store_true", help="draw FPS in the preview")
    parser.add_argument("--window-name", default="RGB Camera Preview")
    parser.add_argument("--wait-key-ms", type=int, default=1)
    return parser.parse_args()


def project_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_device(value: Optional[str]):
    if value is None:
        return 0
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_f = image.astype(np.float32)
    if image_f.max(initial=0.0) <= 1.0:
        image_f *= 255.0
    return np.clip(image_f, 0, 255).astype(np.uint8)


def rgb_to_bgr(rgb) -> np.ndarray:
    image = as_uint8(np.asarray(rgb))
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {image.shape}")
    image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def frame_source_rgb_frames(config_path: Path, round_name: str) -> Iterator[np.ndarray]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.components.frame_source.builder import build_frame_source
    from core.config_loader import load_config

    config = load_config(str(config_path))
    frame_source = build_frame_source(config["frame_source"], round_name)
    try:
        for frame in frame_source:
            if frame.rgb is None:
                continue
            yield rgb_to_bgr(frame.rgb)
    finally:
        frame_source.close()


def uvc_bgr_frames(args: argparse.Namespace) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(parse_device(args.device))
    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    if args.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, int(args.fps))

    if not cap.isOpened():
        raise RuntimeError(f"failed to open UVC camera: {args.device or 0}")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError("failed to read UVC RGB frame")
            yield as_uint8(frame_bgr)
    finally:
        cap.release()


def draw_fps(frame_bgr: np.ndarray, fps: float) -> np.ndarray:
    view = frame_bgr.copy()
    cv2.putText(
        view,
        f"{fps:5.1f} FPS",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return view


def run_preview(args: argparse.Namespace) -> None:
    if args.source == "frame-source":
        frames = frame_source_rgb_frames(project_path(args.config), args.round)
    else:
        frames = uvc_bgr_frames(args)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    last_ts = time.monotonic()
    fps_smooth = 0.0
    try:
        for frame_bgr in frames:
            now = time.monotonic()
            delta = max(1e-6, now - last_ts)
            last_ts = now
            fps = 1.0 / delta
            fps_smooth = fps if fps_smooth <= 0 else fps_smooth * 0.8 + fps * 0.2

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            if args.show_fps:
                frame_bgr = draw_fps(frame_bgr, fps_smooth)

            cv2.imshow(args.window_name, frame_bgr)
            key = cv2.waitKey(max(1, int(args.wait_key_ms))) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        cv2.destroyWindow(args.window_name)


def main() -> int:
    args = parse_args()
    run_preview(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
