"""YOLO26 detector runtime with ONNX/ACL backends.

This component follows Ultralytics detect export conventions without depending
on the Ultralytics runtime at inference time. It supports both common export
styles:

- raw detect head: ``(1, 4 + nc, N)`` or ``(1, N, 4 + nc)`` with ``xywh`` and
  per-class scores;
- end-to-end/NMS output: ``(1, N, 6)`` or ``(N, 6)`` with
  ``xyxy, score, class``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from core.components.detector.base import BaseDetector
from core.components.detector.yolo26.resolver import resolve_yolo26_backend
from core.components.detector.yolov11_rgbd.postprocess import (
    PostprocessConfig,
    non_max_suppression,
    prepare_yolo_output,
    scale_boxes,
)
from core.infra.inference.backend.base import BaseInferenceBackend
from core.types import Detection, Frame


class Yolo26Detector(BaseDetector):
    """RGB YOLO26 detector compatible with fixed-shape ONNX/OM deployment."""

    def __init__(
        self,
        config: Dict,
        class_registry: Dict,
        backend: BaseInferenceBackend,
        backend_name: str,
        model_path: Path,
    ):
        self.detector_type = str(config.get("type", "yolo26"))
        self.input_width = int(config.get("input_width", 640))
        self.input_height = int(config.get("input_height", self.input_width))
        self.pad_value = int(config.get("pad_value", 114))
        self.backend = backend
        self.backend_name = backend_name
        self.model_path = model_path
        self.detector_id_to_class = {
            int(class_id): str(class_name)
            for class_id, class_name in class_registry["detector_id_to_class"].items()
        }
        self.postprocess_config = PostprocessConfig(
            conf_thresh=float(config.get("conf_thresh", 0.25)),
            nms_thresh=float(config.get("nms_thresh", 0.7)),
            max_det=int(config.get("max_det", 300)),
            max_nms=int(config.get("max_nms", 30000)),
            max_wh=int(config.get("max_wh", 7680)),
            agnostic_nms=bool(config.get("agnostic_nms", False)),
        )
        self.output_format = str(config.get("output_format", "auto")).strip().lower()
        if self.output_format not in {"auto", "raw", "end2end"}:
            raise ValueError("yolo26.output_format must be auto, raw, or end2end")
        self.allowed_class_ids = self._resolve_allowed_class_ids(config.get("classes"))
        self._validate_backend_input()

    def infer(self, frame: Frame, table: int) -> List[Detection]:
        """Run RGB preprocessing, backend inference, and YOLO postprocessing."""
        if frame.rgb is None:
            raise ValueError(f"{self.detector_type} requires frame.rgb")

        data, ratio_pad, original_shape = self._preprocess_image(frame.rgb)
        outputs = self.backend.execute(data)
        return self._decode_outputs(
            outputs=outputs,
            ratio_pad=ratio_pad,
            original_shape=original_shape,
            table=table,
        )

    def close(self) -> None:
        self.backend.close()

    def _preprocess_image(
        self,
        rgb: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]], Tuple[int, int]]:
        """Preprocess HWC RGB into NCHW float32 tensor."""
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"{self.detector_type} requires HWC 3-channel RGB, got shape={rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb_float = rgb.astype(np.float32)
            if rgb_float.max(initial=0.0) <= 1.0:
                rgb_float *= 255.0
            rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)

        original_shape = rgb.shape[:2]
        letterboxed, ratio_pad = self._letterbox(
            rgb,
            (self.input_height, self.input_width),
        )
        data = letterboxed.astype(np.float32) / 255.0
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=0).astype(np.float32)
        return data, ratio_pad, original_shape

    def _letterbox(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Resize and pad like Ultralytics LetterBox with ``auto=False``."""
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
            value=(self.pad_value, self.pad_value, self.pad_value),
        )
        return image, ((ratio, ratio), (float(left), float(top)))

    def _decode_outputs(
        self,
        outputs,
        ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
        original_shape: Tuple[int, int],
        table: int,
    ) -> List[Detection]:
        """Decode raw or end-to-end YOLO26 outputs into image-space detections."""
        if self.output_format == "end2end" or (
            self.output_format == "auto" and self._looks_like_end2end(outputs)
        ):
            detections_np = self._decode_end2end_output(outputs)
        else:
            prediction = prepare_yolo_output(
                outputs,
                num_classes=len(self.detector_id_to_class),
            )
            detections_np = non_max_suppression(
                prediction,
                self.postprocess_config,
                classes=self.allowed_class_ids,
            )

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
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=class_id,
                    bbox=(x1, y1, x2, y2),
                    score=float(row[4]),
                    evidence={
                        "table": table,
                        "detector": "yolo26",
                        "backend": self.backend_name,
                        "weights": str(self.model_path),
                    },
                )
            )
        return detections

    def _decode_end2end_output(self, outputs) -> np.ndarray:
        """Decode ``xyxy, score, class`` output and apply class/score filters."""
        output = np.asarray(outputs[0])
        output = np.squeeze(output)
        if output.ndim != 2:
            raise ValueError(f"unsupported YOLO26 end2end output shape: {output.shape}")
        if output.shape[1] != 6 and output.shape[0] == 6:
            output = output.T
        if output.shape[1] != 6:
            raise ValueError(f"unsupported YOLO26 end2end output shape: {output.shape}")

        output = output.astype(np.float32, copy=False)
        scores = output[:, 4]
        class_ids = output[:, 5].astype(np.int32)
        keep = scores >= self.postprocess_config.conf_thresh
        if self.allowed_class_ids is not None:
            allowed = np.asarray(list(self.allowed_class_ids), dtype=np.int32)
            keep &= (class_ids[:, None] == allowed[None]).any(axis=1)
        output = output[keep]
        if output.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        order = output[:, 4].argsort()[::-1]
        output = output[order[: self.postprocess_config.max_det]]
        return output.astype(np.float32, copy=False)

    def _looks_like_end2end(self, outputs) -> bool:
        """Heuristic for Ultralytics NMS/end-to-end ``(N, 6)`` outputs."""
        if not outputs:
            return False
        output = np.asarray(outputs[0])
        output = np.squeeze(output)
        if output.ndim != 2:
            return False
        if output.shape[1] == 6:
            return output.shape[0] != 4 + len(self.detector_id_to_class)
        if output.shape[0] == 6:
            return output.shape[1] != 4 + len(self.detector_id_to_class)
        return False

    def _resolve_allowed_class_ids(self, classes) -> Iterable[int] | None:
        """Resolve optional class filter from ids or class names."""
        if classes is None:
            return None
        name_to_id = {
            class_name: class_id
            for class_id, class_name in self.detector_id_to_class.items()
        }
        resolved: List[int] = []
        for item in classes:
            if isinstance(item, int):
                resolved.append(item)
                continue
            text = str(item)
            if text.isdigit():
                resolved.append(int(text))
            elif text in name_to_id:
                resolved.append(name_to_id[text])
            else:
                raise ValueError(f"unknown YOLO26 class filter: {item}")
        return resolved

    def _validate_backend_input(self) -> None:
        """If backend exposes a static NCHW input, validate RGB channel count."""
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
        if isinstance(channel, int) and channel != 3:
            raise ValueError(
                f"{self.detector_type} model input channel must be 3, got shape={shape}"
            )


def build_yolo26_detector(config: Dict, class_registry: Dict) -> Yolo26Detector:
    """Build a YOLO26 detector with the configured ONNX/ACL backend."""
    backend_name, model_path = resolve_yolo26_backend(config)
    backend = _build_backend(backend_name, model_path, config)
    return Yolo26Detector(
        config=config,
        class_registry=class_registry,
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
    raise ValueError(f"unsupported resolved YOLO26 backend: {backend_name}")

