"""Ultralytics-compatible YOLO detect 后处理。

本模块不依赖 PyTorch/Ultralytics runtime，便于 ONNX/ACL 部署；函数语义对齐
Ultralytics 的 ``xywh2xyxy``、``non_max_suppression``、``scale_boxes`` 和
``clip_boxes``，供 RGBD detector 独立使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PostprocessConfig:
    conf_thresh: float
    nms_thresh: float
    max_det: int = 300
    max_nms: int = 30000
    max_wh: int = 7680
    agnostic_nms: bool = False


def prepare_yolo_output(outputs: Sequence[np.ndarray], num_classes: int | None = None) -> np.ndarray:
    """把 YOLO11 detect 输出统一为 ``(num_boxes, 4 + nc)``。

    兼容常见导出形态：
    - ``(1, 4 + nc, N)``：Ultralytics ONNX 默认；
    - ``(1, N, 4 + nc)``；
    - ``(4 + nc, N)`` / ``(N, 4 + nc)``。
    """
    if not outputs:
        raise ValueError("YOLO output is empty")
    prediction = np.asarray(outputs[0])
    prediction = np.squeeze(prediction)
    if prediction.ndim == 1:
        expected_cols = 4 + num_classes if num_classes is not None else None
        if expected_cols is not None and prediction.shape[0] == expected_cols:
            prediction = prediction[None, :]
    if prediction.ndim != 2:
        raise ValueError(f"unsupported YOLO output shape: {prediction.shape}")

    expected_cols = 4 + num_classes if num_classes is not None else None
    if expected_cols is not None and prediction.shape[1] == expected_cols:
        pass
    elif expected_cols is not None and prediction.shape[0] == expected_cols:
        prediction = prediction.T
    elif prediction.shape[0] < prediction.shape[1]:
        # YOLO detect head 的通道数通常远小于候选框数量，如 (22, 8400)。
        prediction = prediction.T
    if prediction.shape[1] < 5:
        raise ValueError(f"unsupported YOLO output shape after transpose: {prediction.shape}")
    return prediction.astype(np.float32, copy=False)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert boxes from center ``xywh`` to corner ``xyxy`` format."""
    y = np.empty_like(x)
    xy = x[:, :2]
    wh = x[:, 2:4] / 2
    y[:, :2] = xy - wh
    y[:, 2:4] = xy + wh
    return y


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Clip ``xyxy`` boxes to image shape ``(height, width)``."""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
    return boxes


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: np.ndarray,
    img0_shape: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
) -> np.ndarray:
    """Rescale boxes from letterboxed image shape to original image shape.

    Args:
        img1_shape: Network input shape ``(height, width)``.
        boxes: ``xyxy`` boxes in letterboxed image coordinates.
        img0_shape: Original image shape ``(height, width)``.
        ratio_pad: Optional ``((gain_w, gain_h), (pad_w, pad_h))``. Mirrors
            Ultralytics ``scale_boxes`` semantics.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain
    return clip_boxes(boxes, img0_shape)


def non_max_suppression(
    prediction: np.ndarray,
    config: PostprocessConfig,
    classes: Iterable[int] | None = None,
) -> np.ndarray:
    """Ultralytics-compatible single-label NMS.

    Input is ``(N, 4 + nc)`` with ``xywh`` boxes and per-class confidences.
    Output is ``(M, 6)`` in ``xyxy, conf, cls`` format.
    """
    if prediction.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    class_scores = prediction[:, 4:]
    conf = class_scores.max(axis=1)
    keep = conf > config.conf_thresh
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)

    prediction = prediction[keep]
    class_scores = class_scores[keep]
    conf = conf[keep]
    cls = class_scores.argmax(axis=1).astype(np.float32)

    if classes is not None:
        allowed = np.asarray(list(classes), dtype=np.float32)
        class_keep = (cls[:, None] == allowed[None]).any(axis=1)
        if not np.any(class_keep):
            return np.zeros((0, 6), dtype=np.float32)
        prediction = prediction[class_keep]
        conf = conf[class_keep]
        cls = cls[class_keep]

    boxes = xywh2xyxy(prediction[:, :4])
    order = conf.argsort()[::-1]
    if order.size > config.max_nms:
        order = order[: config.max_nms]
    boxes = boxes[order]
    conf = conf[order]
    cls = cls[order]

    offsets = 0 if config.agnostic_nms else cls[:, None] * config.max_wh
    nms_boxes = boxes + offsets
    selected = _nms(nms_boxes, conf, config.nms_thresh)
    selected = selected[: config.max_det]
    if not selected:
        return np.zeros((0, 6), dtype=np.float32)

    selected = np.asarray(selected, dtype=np.int64)
    return np.column_stack((boxes[selected], conf[selected], cls[selected])).astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Pure NumPy greedy NMS over ``xyxy`` boxes."""
    if boxes.size == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter + 1e-7
        iou = inter / union
        order = rest[iou <= iou_thres]

    return keep
