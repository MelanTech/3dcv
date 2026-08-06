"""YOLO 图像预处理：letterbox 缩放 + 归一化 + 转为网络输入张量。"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """等比缩放并用灰边填充到目标尺寸，返回图像及 (上, 左) 填充量。"""
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    pad_w = (new_shape[1] - new_unpad[0]) / 2
    pad_h = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return image, (top, left)


def preprocess_image(
    image: np.ndarray,
    input_width: int,
    input_height: int,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """预处理为 NCHW float32 张量，并返回填充量和原图尺寸（供解码时还原坐标）。"""
    original_shape = image.shape[:2]
    letterboxed, pad = letterbox(image, (input_height, input_width))
    data = np.asarray(letterboxed, dtype=np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0).astype(np.float32)
    return data, pad, original_shape
