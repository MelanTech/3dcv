"""YOLOv11 检测器：本版本自包含预处理、推理和输出解码。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.components.detector.base import BaseDetector
from core.components.detector.yolov11.resolver import resolve_yolo_backend
from core.infra.inference.backend.base import BaseInferenceBackend
from core.types import Detection, Frame


class YoloV11Detector(BaseDetector):
    """YOLOv11 检测器实现。

    YOLO 不同版本的输入预处理、输出 tensor 结构和后处理规则容易分叉，
    因此 YOLOv11 的这些逻辑集中放在当前模块内，避免跨版本共享错误抽象。
    """

    def __init__(
        self,
        config: Dict,
        class_registry: Dict,
        backend: BaseInferenceBackend,
    ):
        self.detector_type = str(config["type"])
        self.input_width = int(config["input_width"])
        self.input_height = int(config["input_height"])
        self.conf_thresh = float(config["conf_thresh"])
        self.nms_thresh = float(config["nms_thresh"])
        self.detector_id_to_class = {
            int(class_id): str(class_name)
            for class_id, class_name in class_registry["detector_id_to_class"].items()
        }
        self.backend = backend

    def infer(self, frame: Frame, table: int) -> List[Detection]:
        """预处理 RGB → 后端推理 → 解码为图像坐标下的检测框。"""
        if frame.rgb is None:
            raise ValueError(f"{self.detector_type} requires frame.rgb")

        data, pad, original_shape = self._preprocess_image(frame.rgb)
        outputs = self.backend.execute(data)
        return self._decode_outputs(
            outputs=outputs,
            pad=pad,
            original_shape=original_shape,
            table=table,
        )

    def close(self) -> None:
        self.backend.close()

    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """预处理为 NCHW float32 张量，并返回填充量和原图尺寸。"""
        original_shape = image.shape[:2]
        letterboxed, pad = self._letterbox(
            image,
            (self.input_height, self.input_width),
        )
        data = np.asarray(letterboxed, dtype=np.float32) / 255.0
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=0).astype(np.float32)
        return data, pad, original_shape

    @staticmethod
    def _letterbox(
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

    def _decode_outputs(
        self,
        outputs,
        pad: Tuple[int, int],
        original_shape: Tuple[int, int],
        table: int,
    ) -> List[Detection]:
        """把 YOLOv11 原始输出解码为原图坐标下、经过 NMS 的检测框列表。"""
        output = np.transpose(np.squeeze(outputs[0]))
        pad_y, pad_x = pad
        original_height, original_width = original_shape
        gain = min(
            self.input_height / original_height,
            self.input_width / original_width,
        )

        class_scores = output[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        valid_mask = max_scores >= self.conf_thresh
        if not np.any(valid_mask):
            return []

        valid_outputs = output[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_class_ids = np.argmax(class_scores[valid_mask], axis=1)

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
            self.conf_thresh,
            self.nms_thresh,
        )
        indices = self._normalize_nms_indices(indices)

        detections: List[Detection] = []
        for index in indices:
            class_id = int(valid_class_ids[index])
            class_name = self.detector_id_to_class.get(
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
                        "backend": self.backend.name,
                    },
                )
            )
        return detections

    @staticmethod
    def _normalize_nms_indices(indices) -> np.ndarray:
        """把不同 OpenCV 版本返回的 NMS 索引统一成一维 int32 数组。"""
        if len(indices) == 0:
            return np.array([], dtype=np.int32)
        return np.asarray(indices, dtype=np.int32).reshape(-1)


def build_yolov11_detector(config: dict, class_registry: Dict) -> BaseDetector:
    """构建 YOLOv11 检测器。"""
    backend_name, model_path = resolve_yolo_backend(config)
    backend = _build_backend(backend_name, model_path, config)
    return YoloV11Detector(
        config=config,
        class_registry=class_registry,
        backend=backend,
    )


def _build_backend(
    backend_name: str,
    model_path: Path,
    config: dict,
) -> BaseInferenceBackend:
    """按解析出的后端名（onnx / acl）创建对应推理后端。"""
    if backend_name == "onnx":
        from core.infra.inference.backend.onnx import OnnxBackend

        return OnnxBackend(model_path, config)

    if backend_name == "acl":
        from core.infra.inference.backend.acl import AclBackend

        return AclBackend(model_path, config)

    raise ValueError(f"unsupported resolved YOLO backend: {backend_name}")
