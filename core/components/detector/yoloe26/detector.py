"""YOLOE-26 fixed-prompt detector runtime.

The model must already have text prompt embeddings baked in before export:
``.pt + set_classes(labels, embeddings) -> .onnx -> .om``.
At runtime this detector only loads the fixed .onnx/.om model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.components.detector.base import BaseDetector
from core.components.detector.yoloe26.resolver import resolve_yoloe26_backend
from core.infra.inference.backend.base import BaseInferenceBackend
from core.types import Detection, Frame
from core.utils.box import nms_detections


class YoloE26Detector(BaseDetector):
    """Runs fixed-prompt YOLOE-26 and emits one of the configured prompt classes."""

    def __init__(
        self,
        config: Dict,
        backend: BaseInferenceBackend,
        backend_name: str,
        model_path: Path,
    ):
        self.output_class = str(config.get("output_class", "UnknownOcrCandidate"))
        prompt_config = config.get("prompt", {})
        self.prompt_labels = [str(label) for label in prompt_config.get("labels", [])]
        if not self.prompt_labels:
            raise ValueError("YOLOE-26 prompt.labels must not be empty")

        self.input_width = int(config.get("input_width", 640))
        self.input_height = int(config.get("input_height", self.input_width))
        self.conf_thresh = float(config.get("conf_thresh", 0.25))
        self.nms_thresh = float(config.get("nms_thresh", 0.7))
        self.backend = backend
        self.backend_name = backend_name
        self.model_path = model_path

    def infer(self, frame: Frame, table: int) -> List[Detection]:
        if frame.rgb is None:
            return []

        data, ratio_pad, original_shape = self._preprocess_image(frame.rgb)
        outputs = self.backend.execute(data)
        detections = self._decode_outputs(
            outputs=outputs,
            ratio_pad=ratio_pad,
            original_shape=original_shape,
            table=table,
        )
        if self.nms_thresh > 0.0:
            detections = nms_detections(detections, self.nms_thresh)
        return detections

    def close(self) -> None:
        self.backend.close()

    def _preprocess_image(
        self,
        rgb: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]], Tuple[int, int]]:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"YOLOE-26 requires HWC 3-channel RGB, got shape={rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        original_shape = rgb.shape[:2]
        letterboxed, ratio_pad = self._letterbox(
            rgb,
            (self.input_height, self.input_width),
        )
        data = letterboxed.astype(np.float32) / 255.0
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=0).astype(np.float32)
        return data, ratio_pad, original_shape

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]:
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
        return image, ((ratio, ratio), (float(left), float(top)))

    def _decode_outputs(
        self,
        outputs,
        ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
        original_shape: Tuple[int, int],
        table: int,
    ) -> List[Detection]:
        output = np.asarray(outputs[0])
        if output.ndim == 3:
            output = output[0]
        if output.ndim != 2 or output.shape[1] < 6:
            raise ValueError(f"unsupported YOLOE-26 output shape: {output.shape}")

        boxes = output[:, :4].astype(np.float32)
        scores = output[:, 4].astype(np.float32)
        class_ids = output[:, 5].astype(np.int32)
        keep = scores >= self.conf_thresh
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        boxes = self._scale_boxes(boxes, ratio_pad, original_shape)

        detections: List[Detection] = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            prompt_label = (
                self.prompt_labels[int(class_id)]
                if 0 <= int(class_id) < len(self.prompt_labels)
                else None
            )
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            detections.append(
                Detection(
                    class_name=self.output_class,
                    class_id=-1,
                    bbox=(x1, y1, x2, y2),
                    score=float(score),
                    evidence={
                        "source": f"detector.yoloe26.{self.backend_name}",
                        "table": table,
                        "prompt_class_id": int(class_id),
                        "prompt_label": prompt_label,
                        "weights": str(self.model_path),
                    },
                )
            )
        return detections

    @staticmethod
    def _scale_boxes(
        boxes: np.ndarray,
        ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        (gain_x, gain_y), (pad_x, pad_y) = ratio_pad
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, [0, 2]] /= gain_x
        boxes[:, [1, 3]] /= gain_y
        height, width = original_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        return boxes


def build_yoloe26_detector(config: Dict) -> YoloE26Detector:
    """Build a fixed-prompt YOLOE-26 detector."""
    backend_name, model_path = resolve_yoloe26_backend(config)
    backend = _build_backend(backend_name, model_path, config)
    return YoloE26Detector(
        config=config,
        backend=backend,
        backend_name=backend_name,
        model_path=model_path,
    )


def _build_backend(
    backend_name: str,
    model_path: Path,
    config: Dict,
) -> BaseInferenceBackend:
    if backend_name == "onnx":
        from core.infra.inference.backend.onnx import OnnxBackend

        return OnnxBackend(model_path, config)
    if backend_name == "acl":
        from core.infra.inference.backend.acl import AclBackend

        return AclBackend(model_path, config)
    raise ValueError(f"unsupported resolved YOLOE-26 backend: {backend_name}")
