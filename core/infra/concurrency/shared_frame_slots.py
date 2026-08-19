"""Bounded shared-memory transport for camera RGB/depth frame arrays."""
from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any, Dict, List

import numpy as np

from core.types import Frame


SharedArraySpec = tuple[str, tuple[int, ...], str]
SharedFrameReference = tuple[str, float, SharedArraySpec, SharedArraySpec]


class SharedFrameSlots:
    """Two frame slots, matching the one-frame detector pipeline delay."""

    def __init__(self, frame: Frame):
        rgb = self._as_supported_array(frame.rgb, "rgb")
        depth = self._as_supported_array(frame.depth, "depth")
        self._rgb_slots = self._create_slots(rgb)
        self._depth_slots = self._create_slots(depth)
        self._rgb_shape = tuple(rgb.shape)
        self._depth_shape = tuple(depth.shape)
        self._rgb_dtype = rgb.dtype
        self._depth_dtype = depth.dtype

    @staticmethod
    def _as_supported_array(value: Any, name: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} is not a numpy array")
        if value.dtype.hasobject or not value.flags.c_contiguous:
            raise TypeError(f"{name} is not a contiguous numeric array")
        return value

    @staticmethod
    def _create_slots(array: np.ndarray) -> List[shared_memory.SharedMemory]:
        return [
            shared_memory.SharedMemory(create=True, size=array.nbytes)
            for _ in range(2)
        ]

    def write(self, frame: Frame, slot: int) -> SharedFrameReference:
        rgb = self._as_supported_array(frame.rgb, "rgb")
        depth = self._as_supported_array(frame.depth, "depth")
        if rgb.shape != self._rgb_shape or rgb.dtype != self._rgb_dtype:
            raise ValueError("rgb shape or dtype changed")
        if depth.shape != self._depth_shape or depth.dtype != self._depth_dtype:
            raise ValueError("depth shape or dtype changed")

        np.ndarray(
            self._rgb_shape,
            dtype=self._rgb_dtype,
            buffer=self._rgb_slots[slot].buf,
        )[...] = rgb
        np.ndarray(
            self._depth_shape,
            dtype=self._depth_dtype,
            buffer=self._depth_slots[slot].buf,
        )[...] = depth
        return (
            frame.frame_id,
            frame.timestamp,
            (self._rgb_slots[slot].name, self._rgb_shape, self._rgb_dtype.str),
            (self._depth_slots[slot].name, self._depth_shape, self._depth_dtype.str),
        )

    def close(self) -> None:
        for block in self._rgb_slots + self._depth_slots:
            try:
                block.close()
            finally:
                try:
                    block.unlink()
                except FileNotFoundError:
                    pass


def frame_from_shared_reference(
    reference: SharedFrameReference,
    blocks: Dict[str, shared_memory.SharedMemory],
) -> Frame:
    """Rebuild a zero-copy Frame view in the worker process."""
    frame_id, timestamp, rgb_spec, depth_spec = reference
    return Frame(
        frame_id=frame_id,
        rgb=_shared_array(rgb_spec, blocks),
        depth=_shared_array(depth_spec, blocks),
        timestamp=timestamp,
    )


def _shared_array(
    spec: SharedArraySpec,
    blocks: Dict[str, shared_memory.SharedMemory],
) -> np.ndarray:
    name, shape, dtype = spec
    block = blocks.get(name)
    if block is None:
        block = shared_memory.SharedMemory(name=name)
        blocks[name] = block
    return np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf)
