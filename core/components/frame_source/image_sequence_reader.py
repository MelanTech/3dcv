"""离线图片序列帧源：从磁盘按序读取 RGB/深度图对，用于评测回放。

支持多段序列顺序播放、可选的序列间滑动转场动画、帧率节流。
round2 通过 next_sequence() 在桌位间切换序列。
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.components.frame_source.base import BaseFrameSource
from core.types import Frame


class ImageSequenceFrameSource(BaseFrameSource):
    """从图片目录读取 RGB/深度图对并按帧产出的离线帧源。"""

    def __init__(self, config: Dict):
        self.base_path = Path(config["base_path"]).expanduser()
        self.sequence_names = config.get("sequences")
        self.read_interval = int(config.get("read_interval", 1))
        self.fps = float(config.get("fps", 0.0))
        if self.fps < 0:
            raise ValueError("fps must be non-negative")
        self.frame_interval_sec = 1.0 / self.fps if self.fps > 0 else 0.0
        self.loop = bool(config.get("loop", False))
        self.rgb_dir_name = config.get("rgb_dir", "rgb")
        self.depth_dir_name = config.get("depth_dir", "depth")
        self.rgb_suffix = config.get("rgb_suffix", "_rgb")
        self.depth_suffix = config.get("depth_suffix", "_depth")
        self.rgb_extensions = tuple(config.get("rgb_extensions", [".jpg", ".jpeg", ".png"]))
        self.depth_extensions = tuple(config.get("depth_extensions", [".png"]))
        self.convert_rgb = bool(config.get("convert_rgb", True))
        self.transition_config = dict(config.get("transition", {}))
        self.transition_enabled = bool(self.transition_config.get("enabled", False))
        self.transition_direction = self.transition_config.get("direction", "left")
        self.transition_frame_range = tuple(
            self.transition_config.get("duration_frames_range", [0, 0])
        )
        self.transition_depth_mode = self.transition_config.get("depth_mode", "nearest")
        self.random = random.Random(self.transition_config.get("random_seed"))
        self.selected_transition_direction: Optional[str] = None

        if self.read_interval <= 0:
            raise ValueError("read_interval must be positive")
        self._validate_transition_config()
        if not self.base_path.exists():
            raise ValueError(f"image sequence base_path does not exist: {self.base_path}")

        self.sequences = self._load_sequences()
        self.current_sequence_index = 0
        self.current_frame_index = 0
        self.transition_state: Optional[Dict] = None
        self.next_frame_time: Optional[float] = None
        self._closed = False

    def _validate_transition_config(self) -> None:
        if self.transition_direction not in ("left", "right", "random"):
            raise ValueError("transition.direction must be left, right, or random")
        if len(self.transition_frame_range) != 2:
            raise ValueError("transition.duration_frames_range must contain two integers")
        min_frames = int(self.transition_frame_range[0])
        max_frames = int(self.transition_frame_range[1])
        if min_frames < 0 or max_frames < min_frames:
            raise ValueError("invalid transition.duration_frames_range")
        self.transition_frame_range = (min_frames, max_frames)
        if self.transition_depth_mode != "nearest":
            raise ValueError("only transition.depth_mode=nearest is supported")

    def _load_sequences(self) -> List[Dict]:
        """加载所有序列目录，为每段序列收集配对好的 RGB/深度帧。"""
        sequence_dirs = self._resolve_sequence_dirs()
        sequences = []
        for sequence_dir in sequence_dirs:
            rgb_dir = sequence_dir / self.rgb_dir_name
            depth_dir = sequence_dir / self.depth_dir_name
            if not rgb_dir.exists():
                raise ValueError(f"RGB folder does not exist: {rgb_dir}")
            if not depth_dir.exists():
                raise ValueError(f"depth folder does not exist: {depth_dir}")

            pairs = self._collect_pairs(rgb_dir, depth_dir)
            if pairs:
                sequences.append(
                    {
                        "name": sequence_dir.name,
                        "base_path": sequence_dir,
                        "rgb_dir": rgb_dir,
                        "depth_dir": depth_dir,
                        "pairs": pairs,
                    }
                )

        if not sequences:
            raise ValueError(f"no valid image sequences found under: {self.base_path}")
        return sequences

    def _resolve_sequence_dirs(self) -> List[Path]:
        if self.sequence_names:
            return [self.base_path / name for name in self.sequence_names]

        if (self.base_path / self.rgb_dir_name).exists() and (self.base_path / self.depth_dir_name).exists():
            return [self.base_path]

        return sorted(path for path in self.base_path.iterdir() if path.is_dir())

    def _collect_pairs(self, rgb_dir: Path, depth_dir: Path) -> List[Dict]:
        """按 frame_id 把 RGB 与深度图配对，只保留两边都存在的帧并排序。"""
        rgb_by_id = {}
        for path in rgb_dir.iterdir():
            if path.suffix.lower() in self.rgb_extensions:
                frame_id = self._frame_id(path, self.rgb_suffix)
                rgb_by_id[frame_id] = path

        depth_by_id = {}
        for path in depth_dir.iterdir():
            if path.suffix.lower() in self.depth_extensions:
                frame_id = self._frame_id(path, self.depth_suffix)
                depth_by_id[frame_id] = path

        common_ids = sorted(set(rgb_by_id).intersection(depth_by_id), key=self._sort_key)
        return [
            {
                "frame_id": frame_id,
                "rgb_path": rgb_by_id[frame_id],
                "depth_path": depth_by_id[frame_id],
            }
            for frame_id in common_ids
        ]

    @staticmethod
    def _sort_key(frame_id: str):
        tail = frame_id.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else frame_id

    @staticmethod
    def _frame_id(path: Path, suffix: str) -> str:
        stem = path.stem
        if suffix and stem.endswith(suffix):
            return stem[: -len(suffix)]
        return stem

    def __iter__(self) -> "ImageSequenceFrameSource":
        self.current_sequence_index = 0
        self.current_frame_index = 0
        self.transition_state = None
        self.selected_transition_direction = None
        self.next_frame_time = None
        return self

    def __next__(self) -> Frame:
        """产出下一帧：优先播放转场帧，否则按当前序列顺序取帧，序列耗尽则切换/结束。"""
        if self._closed:
            raise StopIteration

        while True:
            if self.transition_state is not None:
                return self._read_transition_frame()

            if self.current_sequence_index >= len(self.sequences):
                if self.loop:
                    self.current_sequence_index = 0
                    self.current_frame_index = 0
                else:
                    raise StopIteration

            sequence = self.sequences[self.current_sequence_index]
            pairs = sequence["pairs"]
            if self.current_frame_index < len(pairs):
                pair = pairs[self.current_frame_index]
                self.current_frame_index += self.read_interval
                self._throttle_frame_rate()
                return self._read_frame(sequence["name"], pair)

            if self._start_transition_to_next_sequence(sequence):
                continue

            self.current_sequence_index += 1
            self.current_frame_index = 0

    def _read_frame(self, sequence_name: str, pair: Dict) -> Frame:
        """读取一对已经对齐好的 RGB/深度图，按需做色彩空间转换后封装成 Frame。"""
        rgb = cv2.imread(str(pair["rgb_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"failed to read RGB image: {pair['rgb_path']}")
        if self.convert_rgb:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(pair["depth_path"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"failed to read depth image: {pair['depth_path']}")

        return Frame(
            frame_id=f"{sequence_name}/{pair['frame_id']}",
            rgb=rgb,
            depth=depth,
            timestamp=time.time(),
        )

    def _start_transition_to_next_sequence(self, sequence: Dict, use_current_frame: bool = False) -> bool:
        """若开启转场，初始化到下一段序列的滑动动画状态，成功返回 True。"""
        next_index = self.current_sequence_index + 1
        if not self.transition_enabled or next_index >= len(self.sequences):
            return False

        min_frames, max_frames = self.transition_frame_range
        frame_count = self.random.randint(min_frames, max_frames)
        if frame_count <= 0:
            return False

        current_pairs = sequence["pairs"]
        next_sequence = self.sequences[next_index]
        if not current_pairs or not next_sequence["pairs"]:
            return False

        old_pair_index = len(current_pairs) - 1
        if use_current_frame:
            old_pair_index = max(0, min(self.current_frame_index - self.read_interval, len(current_pairs) - 1))
        old_frame = self._read_frame(sequence["name"], current_pairs[old_pair_index])
        new_frame = self._read_frame(next_sequence["name"], next_sequence["pairs"][0])
        direction = self._select_transition_direction()
        self.transition_state = {
            "from_sequence": sequence["name"],
            "to_sequence": next_sequence["name"],
            "old_frame": old_frame,
            "new_frame": new_frame,
            "frame_count": frame_count,
            "frame_index": 0,
            "direction": direction,
            "next_sequence_index": next_index,
        }
        return True

    def _select_transition_direction(self) -> str:
        if self.transition_direction == "random":
            if self.selected_transition_direction is None:
                self.selected_transition_direction = self.random.choice(["left", "right"])
            return self.selected_transition_direction
        return self.transition_direction

    def _read_transition_frame(self) -> Frame:
        """合成并产出一帧转场动画；转场结束时切换到下一段序列。"""
        state = self.transition_state
        frame_index = int(state["frame_index"])
        frame_count = int(state["frame_count"])
        direction = state["direction"]

        progress = float(frame_index + 1) / float(frame_count + 1)
        old_frame: Frame = state["old_frame"]
        new_frame: Frame = state["new_frame"]
        rgb = self._compose_slide(old_frame.rgb, new_frame.rgb, progress, direction, fill_value=0)
        depth = self._compose_slide(old_frame.depth, new_frame.depth, progress, direction, fill_value=0)

        self._throttle_frame_rate()

        state["frame_index"] = frame_index + 1
        if state["frame_index"] >= frame_count:
            self.current_sequence_index = int(state["next_sequence_index"])
            self.current_frame_index = self.read_interval
            self.transition_state = None

        return Frame(
            frame_id=(
                f"{state['from_sequence']}_to_{state['to_sequence']}/"
                f"transition_{frame_index + 1:06d}"
            ),
            rgb=rgb,
            depth=depth,
            timestamp=time.time(),
        )

    def _throttle_frame_rate(self) -> None:
        """按配置的 fps 做节流，让离线回放的取帧速度接近真实相机。"""
        if self.frame_interval_sec <= 0.0:
            return

        now = time.monotonic()
        if self.next_frame_time is None:
            self.next_frame_time = now
            return

        self.next_frame_time += self.frame_interval_sec
        delay = self.next_frame_time - now
        if delay > 0.0:
            time.sleep(delay)
            return

        if -delay > self.frame_interval_sec:
            self.next_frame_time = now

    @staticmethod
    def _compose_slide(old_image, new_image, progress: float, direction: str, fill_value: int):
        """按进度把旧帧和新帧做左/右滑动拼接，生成一帧转场画面。"""
        if old_image is None:
            return new_image
        if new_image is None:
            return old_image

        height, width = old_image.shape[:2]
        if new_image.shape[:2] != (height, width):
            new_image = cv2.resize(new_image, (width, height), interpolation=cv2.INTER_NEAREST)

        output = np.full_like(old_image, fill_value)
        offset = int(round(width * progress))

        if direction == "left":
            old_x = -offset
            new_x = width - offset
        else:
            old_x = offset
            new_x = offset - width

        ImageSequenceFrameSource._paste_with_clip(output, old_image, old_x)
        ImageSequenceFrameSource._paste_with_clip(output, new_image, new_x)
        return output

    @staticmethod
    def _paste_with_clip(canvas, image, x_offset: int) -> None:
        width = canvas.shape[1]
        source_width = image.shape[1]
        dst_x1 = max(0, x_offset)
        dst_x2 = min(width, x_offset + source_width)
        if dst_x2 <= dst_x1:
            return

        src_x1 = dst_x1 - x_offset
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        canvas[:, dst_x1:dst_x2] = image[:, src_x1:src_x2]

    def next_sequence(self) -> bool:
        """切换到下一段序列（round2 桌间切换用）；开启转场时改为启动转场动画。"""
        if self.current_sequence_index + 1 >= len(self.sequences):
            return False

        if self.transition_enabled:
            sequence = self.sequences[self.current_sequence_index]
            return self._start_transition_to_next_sequence(sequence, use_current_frame=True)

        self.current_sequence_index += 1
        self.current_frame_index = 0
        self.transition_state = None
        return True

    def get_current_sequence_info(self) -> Optional[Dict]:
        """返回当前序列（或转场状态）的进度信息，供状态机记录日志。"""
        if self.transition_state is not None:
            state = self.transition_state
            return {
                "sequence_index": self.current_sequence_index,
                "name": f"{state['from_sequence']}_to_{state['to_sequence']}",
                "from_sequence": state["from_sequence"],
                "to_sequence": state["to_sequence"],
                "transition_frame_index": state["frame_index"],
                "transition_frame_count": state["frame_count"],
                "remaining_frames": max(0, state["frame_count"] - state["frame_index"]),
            }
        if self.current_sequence_index >= len(self.sequences):
            return None
        sequence = self.sequences[self.current_sequence_index]
        return {
            "sequence_index": self.current_sequence_index,
            "name": sequence["name"],
            "base_path": str(sequence["base_path"]),
            "total_frames": len(sequence["pairs"]),
            "remaining_frames": max(0, len(sequence["pairs"]) - self.current_frame_index),
        }

    def __len__(self) -> int:
        return sum(len(sequence["pairs"]) for sequence in self.sequences)

    def close(self) -> None:
        self._closed = True
