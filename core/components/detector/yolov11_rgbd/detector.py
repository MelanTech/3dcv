"""YOLOv11 RGBD 检测器：ONNX/ACL 推理，Ultralytics-compatible 后处理。

本 detector 与 ``core.components.detector.yolov11`` 完全隔离：RGBD 的四通道预处理、
输出解析和 NMS 均在当前包内维护，避免影响已有 RGB YOLOv11 逻辑。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.components.detector.base import BaseDetector
from core.components.detector.yolov11_rgbd.postprocess import (
    PostprocessConfig,
    non_max_suppression,
    prepare_yolo_output,
    scale_boxes,
)
from core.components.detector.yolov11_rgbd.resolver import resolve_yolov11_rgbd_backend
from core.infra.inference.backend.base import BaseInferenceBackend
from core.types import Detection, Frame


class YoloV11RgbdDetector(BaseDetector):
    """YOLOv11 RGBD 检测器。

    输入通道顺序固定为 ``RGBD``。Depth 以毫米为单位，按训练时常量
    ``clip(depth_mm, 0, depth_max_mm) / depth_max_mm`` 归一化。
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
        self.depth_max_mm = float(config.get("depth_max_mm", 3000))
        self.pad_value = int(config.get("pad_value", 114))
        self.detector_id_to_class = {
            int(class_id): str(class_name)
            for class_id, class_name in class_registry["detector_id_to_class"].items()
        }
        self.backend = backend
        self.postprocess_config = PostprocessConfig(
            conf_thresh=float(config["conf_thresh"]),
            nms_thresh=float(config["nms_thresh"]),
            max_det=int(config.get("max_det", 300)),
            max_nms=int(config.get("max_nms", 30000)),
            max_wh=int(config.get("max_wh", 7680)),
            agnostic_nms=bool(config.get("agnostic_nms", False)),
        )
        self._validate_backend_input()

    def infer(self, frame: Frame, table: int) -> List[Detection]:
        """预处理 RGBD → 后端推理 → Ultralytics-compatible decode/NMS。"""
        if frame.rgb is None:
            raise ValueError(f"{self.detector_type} requires frame.rgb")
        if frame.depth is None:
            raise ValueError(f"{self.detector_type} requires frame.depth")

        data, ratio_pad, original_shape = self._preprocess_rgbd(frame.rgb, frame.depth)
        outputs = self.backend.execute(data)
        return self._decode_outputs(
            outputs=outputs,
            ratio_pad=ratio_pad,
            original_shape=original_shape,
            table=table,
        )

    def close(self) -> None:
        self.backend.close()

    def _preprocess_rgbd(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]], Tuple[int, int]]:
        """预处理为 NCHW float32 RGBD 张量，并返回 ratio/pad 与原图尺寸。"""
        rgb = self._normalize_rgb_input(rgb)
        depth_u8 = self._encode_depth(depth, rgb.shape[:2])
        rgbd = np.dstack((rgb, depth_u8[..., None]))

        letterboxed, ratio_pad = self._letterbox_rgbd(
            rgbd,
            (self.input_height, self.input_width),
        )
        data = letterboxed.astype(np.float32) / 255.0
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=0).astype(np.float32)
        return data, ratio_pad, rgb.shape[:2]

    def _normalize_rgb_input(self, rgb: np.ndarray) -> np.ndarray:
        """校验并规整 RGB 输入为 HWC uint8 三通道。"""
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"{self.detector_type} requires HWC 3-channel RGB, got shape={rgb.shape}")
        if rgb.dtype == np.uint8:
            return rgb
        rgb_float = rgb.astype(np.float32)
        if rgb_float.max(initial=0.0) <= 1.0:
            rgb_float *= 255.0
        return np.clip(rgb_float, 0, 255).astype(np.uint8)

    def _encode_depth(self, depth: np.ndarray, rgb_shape: Tuple[int, int]) -> np.ndarray:
        """把深度毫米图编码为 uint8，与训练输入量纲一致。"""
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"{self.detector_type} requires 2D depth, got shape={depth.shape}")
        if depth.shape[:2] != rgb_shape:
            depth = cv2.resize(
                depth,
                (rgb_shape[1], rgb_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        depth_f = depth.astype(np.float32)
        depth_u8 = np.clip(depth_f, 0, self.depth_max_mm) / self.depth_max_mm * 255.0
        return depth_u8.astype(np.uint8)

    def _letterbox_rgbd(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """等比缩放并填充到目标尺寸，行为对齐 Ultralytics LetterBox。"""
        shape = image.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (
            int(round(shape[1] * ratio)),
            int(round(shape[0] * ratio)),
        )
        pad_w = (new_shape[1] - new_unpad[0]) / 2
        pad_h = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            if image.ndim == 2:
                image = image[..., None]

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        h, w, c = image.shape
        padded = np.full(
            (h + top + bottom, w + left + right, c),
            fill_value=self.pad_value,
            dtype=image.dtype,
        )
        padded[top : top + h, left : left + w] = image
        return padded, ((ratio, ratio), (float(left), float(top)))

    def _decode_outputs(
        self,
        outputs,
        ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
        original_shape: Tuple[int, int],
        table: int,
    ) -> List[Detection]:
        """把 YOLOv11 原始输出解码为原图坐标下、经过 NMS 的检测框。"""
        prediction = prepare_yolo_output(outputs, num_classes=len(self.detector_id_to_class))
        detections_np = non_max_suppression(prediction, self.postprocess_config)
        if detections_np.size == 0:
            return []

        boxes = detections_np[:, :4].copy()
        boxes = scale_boxes(
            img1_shape=(self.input_height, self.input_width),
            boxes=boxes,
            img0_shape=original_shape,
            ratio_pad=ratio_pad,
        )

        detections: List[Detection] = []
        for row, box in zip(detections_np, boxes):
            class_id = int(row[5])
            class_name = self.detector_id_to_class.get(
                class_id,
                f"class_{class_id}",
            )
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=class_id,
                    bbox=(x1, y1, x2, y2),
                    score=float(row[4]),
                    evidence={
                        "table": table,
                        "detector": "yolov11_rgbd",
                        "backend": self.backend.name,
                    },
                )
            )
        return detections

    def _validate_backend_input(self) -> None:
        """若模型输入为静态 NCHW，则确认通道数为 4。"""
        try:
            inputs = self.backend.get_inputs()
        except Exception:
            return
        if not inputs:
            return
        shape = list(getattr(inputs[0], "shape", []))
        if len(shape) != 4:
            return
        channel = shape[1]
        if isinstance(channel, int) and channel != 4:
            raise ValueError(
                f"{self.detector_type} model input channel must be 4, got shape={shape}"
            )


def build_yolov11_rgbd_detector(config: dict, class_registry: Dict) -> BaseDetector:
    """构建 YOLOv11 RGBD 检测器。"""
    backend_name, model_path = resolve_yolov11_rgbd_backend(config)
    backend = _build_backend(backend_name, model_path, config)
    return YoloV11RgbdDetector(
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

    raise ValueError(f"unsupported resolved YOLO RGBD backend: {backend_name}")
