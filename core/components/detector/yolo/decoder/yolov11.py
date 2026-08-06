"""YOLOv11 输出解码：阈值过滤 → 坐标还原 → NMS → 生成 Detection。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.components.detector.yolo.decoder.base import BaseYoloDecoder
from core.types import Detection


class YoloV11Decoder(BaseYoloDecoder):
    """YOLOv11 解码器实现。"""

    @staticmethod
    def _normalize_nms_indices(indices) -> np.ndarray:
        """把不同 OpenCV 版本返回的 NMS 索引统一成一维 int32 数组。"""
        if len(indices) == 0:
            return np.array([], dtype=np.int32)
        return np.asarray(indices, dtype=np.int32).reshape(-1)

    def decode(
        self,
        outputs,
        pad: Tuple[int, int],
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
        conf_thresh: float,
        nms_thresh: float,
        detector_id_to_class: Dict[int, str],
        table: int,
        backend_name: str,
    ) -> List[Detection]:
        """把 YOLOv11 原始输出解码为原图坐标下、经过 NMS 的检测框列表。"""
        output = np.transpose(np.squeeze(outputs[0]))
        pad_y, pad_x = pad
        original_height, original_width = original_shape
        input_height, input_width = input_shape
        # gain 是 letterbox 时的等比缩放系数，用于把坐标从网络输入尺寸还原回原图。
        gain = min(
            input_height / original_height,
            input_width / original_width,
        )

        # 每个预测的前 4 维是框（中心点+宽高），其余是各类别分数。
        class_scores = output[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        valid_mask = max_scores >= conf_thresh
        if not np.any(valid_mask):
            return []

        valid_outputs = output[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_class_ids = np.argmax(class_scores[valid_mask], axis=1)

        # 先减去 letterbox 填充，再按 gain 缩放回原图坐标系。
        x_center = valid_outputs[:, 0] - pad_x
        y_center = valid_outputs[:, 1] - pad_y
        width = valid_outputs[:, 2]
        height = valid_outputs[:, 3]

        left = np.floor((x_center - width / 2) / gain).astype(np.int32)
        top = np.floor((y_center - height / 2) / gain).astype(np.int32)
        box_width = np.floor(width / gain).astype(np.int32)
        box_height = np.floor(height / gain).astype(np.int32)

        boxes = np.column_stack((left, top, left + box_width, top + box_height))
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_width)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_height)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_height)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            valid_scores.tolist(),
            conf_thresh,
            nms_thresh,
        )
        indices = self._normalize_nms_indices(indices)

        detections: List[Detection] = []
        for index in indices:
            class_id = int(valid_class_ids[index])
            class_name = detector_id_to_class.get(
                class_id,
                f"class_{class_id}",
            )
            x1, y1, x2, y2 = (int(value) for value in boxes[index])
            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=class_id,
                    bbox=(x1, y1, x2, y2),
                    score=float(valid_scores[index]),
                    evidence={
                        "table": table,
                        "detector": "yolov11",
                        "backend": backend_name,
                    },
                )
            )
        return detections
