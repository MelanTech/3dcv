"""DEIM detector runtime with ONNX/ACL backends.

The migrated 2025 model uses two inputs:

- ``images``: float32 NCHW image tensor;
- ``orig_target_sizes``: int64 ``[height, width]`` tensor.

It returns ``labels``, ``boxes`` and ``scores``. The original deployment feeds
the padded model size as ``orig_target_sizes`` and maps the resulting boxes
back through the letterbox transform; this component preserves that contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from core.components.detector.base import BaseDetector
from core.components.detector.deim.resolver import resolve_deim_backend
from core.infra.inference.backend.base import BaseInferenceBackend
from core.types import Detection, Frame


class DeimDetector(BaseDetector):
    """DEIM detector compatible with fixed-shape ONNX and Ascend OM models."""

    def __init__(
        self,
        config: Dict,
        class_registry: Dict,
        backend: BaseInferenceBackend,
        backend_name: str,
        model_path: Path,
    ):
        self.detector_type = str(config.get("type", "deim"))
        input_size = int(config.get("input_size", 640))
        self.input_width = int(config.get("input_width", input_size))
        self.input_height = int(config.get("input_height", input_size))
        self.pad_value = int(config.get("pad_value", 0))
        self.conf_thresh = float(config.get("conf_thresh", 0.4))
        self.use_nms = bool(config.get("use_nms", True))
        self.nms_iou = float(config.get("nms_iou", 0.8))
        self.max_det = max(1, int(config.get("max_det", 300)))
        self.agnostic_nms = bool(config.get("agnostic_nms", True))
        self.input_color = str(config.get("input_color", "rgb")).strip().lower()
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("deim.input_color must be rgb or bgr")
        self.normalization = str(
            config.get("normalization", "none")
        ).strip().lower()
        if self.normalization not in {"none", "imagenet"}:
            raise ValueError(
                "deim.normalization must be none or imagenet"
            )
        self.normalization_mean = self._channel_values(
            config.get("normalization_mean", [0.485, 0.456, 0.406]),
            "normalization_mean",
        )
        self.normalization_std = self._channel_values(
            config.get("normalization_std", [0.229, 0.224, 0.225]),
            "normalization_std",
        )
        if np.any(self.normalization_std <= 0):
            raise ValueError("deim.normalization_std values must be positive")

        target_dtype_name = str(
            config.get("target_size_dtype", "int64")
        ).strip().lower()
        target_dtypes = {"int32": np.int32, "int64": np.int64}
        if target_dtype_name not in target_dtypes:
            raise ValueError("deim.target_size_dtype must be int32 or int64")
        self.target_size_dtype = target_dtypes[target_dtype_name]

        self.detector_id_to_class = {
            int(class_id): str(class_name)
            for class_id, class_name in class_registry[
                "detector_id_to_class"
            ].items()
        }
        self.allowed_class_ids = self._resolve_allowed_class_ids(
            config.get("classes")
        )
        self.output_order = tuple(
            str(name).strip().lower()
            for name in config.get(
                "output_order",
                ["labels", "boxes", "scores"],
            )
        )
        if sorted(self.output_order) != ["boxes", "labels", "scores"]:
            raise ValueError(
                "deim.output_order must contain labels, boxes, and scores"
            )

        self.backend = backend
        self.backend_name = backend_name
        self.model_path = model_path
        self._validate_backend_inputs()

    def infer(self, frame: Frame, table: int) -> List[Detection]:
        """Preprocess RGB, execute the two-input model, and decode detections."""
        if frame.rgb is None:
            raise ValueError("deim requires frame.rgb")

        image_tensor, target_sizes, transform, original_shape = (
            self._preprocess_image(frame.rgb)
        )
        outputs = self.backend.execute([image_tensor, target_sizes])
        return self._decode_outputs(
            outputs=outputs,
            transform=transform,
            original_shape=original_shape,
            table=table,
        )

    def close(self) -> None:
        self.backend.close()

    def _preprocess_image(
        self,
        rgb: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Tuple[float, int, int],
        Tuple[int, int],
    ]:
        """Letterbox HWC RGB and return both DEIM inputs plus inverse transform."""
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"deim requires HWC 3-channel RGB, got shape={image.shape}"
            )
        if image.dtype != np.uint8:
            values = image.astype(np.float32)
            if values.max(initial=0.0) <= 1.0:
                values *= 255.0
            image = np.clip(values, 0, 255).astype(np.uint8)
        if self.input_color == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        original_shape = image.shape[:2]
        padded, ratio, pad_w, pad_h = self._letterbox(image)
        image_tensor = padded.astype(np.float32) / 255.0
        if self.normalization == "imagenet":
            image_tensor = (
                image_tensor - self.normalization_mean
            ) / self.normalization_std
        image_tensor = np.transpose(image_tensor, (2, 0, 1))[None]
        image_tensor = np.ascontiguousarray(image_tensor, dtype=np.float32)

        # Preserve the 2025 export contract: boxes are emitted in padded-input
        # coordinates and are mapped back by ratio/padding below.
        target_sizes = np.asarray(
            [[self.input_height, self.input_width]],
            dtype=self.target_size_dtype,
        )
        return (
            image_tensor,
            target_sizes,
            (ratio, pad_w, pad_h),
            original_shape,
        )

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int]:
        """Resize with aspect ratio and center-pad using the legacy DEIM rules."""
        height, width = image.shape[:2]
        ratio = min(
            self.input_width / width,
            self.input_height / height,
        )
        resized_width = max(1, int(width * ratio))
        resized_height = max(1, int(height * ratio))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_w = (self.input_width - resized_width) // 2
        pad_h = (self.input_height - resized_height) // 2
        padded = np.full(
            (self.input_height, self.input_width, 3),
            self.pad_value,
            dtype=np.uint8,
        )
        padded[
            pad_h : pad_h + resized_height,
            pad_w : pad_w + resized_width,
        ] = resized
        return padded, ratio, pad_w, pad_h

    def _decode_outputs(
        self,
        outputs: Sequence[np.ndarray],
        transform: Tuple[float, int, int],
        original_shape: Tuple[int, int],
        table: int,
    ) -> List[Detection]:
        labels, boxes, scores = self._normalize_outputs(outputs)
        labels = np.asarray(labels).reshape(-1).astype(np.int32, copy=False)
        scores = np.asarray(scores).reshape(-1).astype(np.float32, copy=False)
        boxes = np.asarray(boxes)
        if boxes.size == 0:
            return []
        if boxes.shape[-1] != 4:
            raise ValueError(
                f"unsupported DEIM boxes output shape: {boxes.shape}"
            )
        boxes = boxes.reshape(-1, 4).astype(np.float32, copy=False)

        count = min(len(labels), len(boxes), len(scores))
        labels = labels[:count]
        boxes = boxes[:count]
        scores = scores[:count]
        keep = np.isfinite(scores) & (scores >= self.conf_thresh)
        keep &= np.isfinite(boxes).all(axis=1)
        if self.allowed_class_ids is not None:
            allowed = np.asarray(self.allowed_class_ids, dtype=np.int32)
            keep &= (labels[:, None] == allowed[None]).any(axis=1)
        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]
        if boxes.size == 0:
            return []

        ratio, pad_w, pad_h = transform
        denominator = max(float(ratio), 1e-6)
        boxes = boxes.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / denominator
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / denominator
        original_height, original_width = original_shape
        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]], 0, original_width
        )
        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]], 0, original_height
        )

        valid_extent = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        labels = labels[valid_extent]
        boxes = boxes[valid_extent]
        scores = scores[valid_extent]
        if boxes.size == 0:
            return []

        indices = np.arange(len(boxes), dtype=np.int32)
        if self.use_nms and len(boxes) > 1:
            indices = self._nms(
                boxes,
                scores,
                labels,
                self.nms_iou,
                self.agnostic_nms,
            )
        if len(indices) > self.max_det:
            order = scores[indices].argsort()[::-1]
            indices = indices[order[: self.max_det]]

        detections: List[Detection] = []
        for index in indices:
            class_id = int(labels[index])
            x1, y1, x2, y2 = (
                int(round(value)) for value in boxes[index]
            )
            detections.append(
                Detection(
                    class_name=self.detector_id_to_class.get(
                        class_id,
                        f"class_{class_id}",
                    ),
                    class_id=class_id,
                    bbox=(x1, y1, x2, y2),
                    score=float(scores[index]),
                    evidence={
                        "table": table,
                        "detector": self.detector_type,
                        "backend": self.backend_name,
                        "weights": str(self.model_path),
                        "normalization": self.normalization,
                    },
                )
            )
        return detections

    @staticmethod
    def _channel_values(values, key: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(f"deim.{key} must contain exactly 3 values")
        return array.reshape((1, 1, 3))

    def _normalize_outputs(
        self,
        outputs: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) != 3:
            raise ValueError(
                f"DEIM expects 3 outputs, got {len(outputs)}"
            )

        try:
            output_infos = self.backend.get_outputs()
        except Exception:
            output_infos = []
        if len(output_infos) == len(outputs):
            by_name = {
                str(info.name).strip().lower(): np.asarray(value)
                for info, value in zip(output_infos, outputs)
            }
            if {"labels", "boxes", "scores"} <= by_name.keys():
                return (
                    by_name["labels"],
                    by_name["boxes"],
                    by_name["scores"],
                )

        # OM output names may be rewritten. Shape/dtype inference handles the
        # standard exported contract before falling back to configured order.
        arrays = [np.asarray(value) for value in outputs]
        box_indices = [
            index
            for index, value in enumerate(arrays)
            if value.ndim >= 2 and value.shape[-1] == 4
        ]
        if len(box_indices) == 1:
            box_index = box_indices[0]
            remaining = [
                index for index in range(len(arrays)) if index != box_index
            ]
            integer = [
                index
                for index in remaining
                if np.issubdtype(arrays[index].dtype, np.integer)
            ]
            if len(integer) == 1:
                label_index = integer[0]
                score_index = next(
                    index for index in remaining if index != label_index
                )
                return (
                    arrays[label_index],
                    arrays[box_index],
                    arrays[score_index],
                )

        by_order = {
            name: arrays[index]
            for index, name in enumerate(self.output_order)
        }
        return (
            by_order["labels"],
            by_order["boxes"],
            by_order["scores"],
        )

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        iou_thresh: float,
        agnostic: bool,
    ) -> np.ndarray:
        """Apply score-ordered NMS, optionally independently per class."""
        if len(boxes) == 0:
            return np.zeros((0,), dtype=np.int32)
        order = scores.argsort()[::-1]
        kept: List[int] = []
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0.0,
            boxes[:, 3] - boxes[:, 1],
        )
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(
                0.0,
                yy2 - yy1,
            )
            union = areas[current] + areas[rest] - intersection + 1e-6
            overlap = intersection / union
            suppress = overlap > iou_thresh
            if not agnostic:
                suppress &= labels[rest] == labels[current]
            order = rest[~suppress]
        return np.asarray(kept, dtype=np.int32)

    def _resolve_allowed_class_ids(
        self,
        classes,
    ) -> Iterable[int] | None:
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
                raise ValueError(f"unknown DEIM class filter: {item}")
        return resolved

    def _validate_backend_inputs(self) -> None:
        try:
            inputs = self.backend.get_inputs()
        except Exception:
            return
        if len(inputs) != 2:
            raise ValueError(
                f"DEIM model must expose 2 inputs, got {len(inputs)}"
            )
        image_shape = list(getattr(inputs[0], "shape", []))
        if len(image_shape) == 4:
            channel = image_shape[1]
            if isinstance(channel, int) and channel != 3:
                raise ValueError(
                    f"DEIM image input channel must be 3, got {image_shape}"
                )
            static_height = image_shape[2]
            static_width = image_shape[3]
            if (
                isinstance(static_height, int)
                and static_height > 0
                and static_height != self.input_height
            ):
                raise ValueError(
                    f"DEIM config input_height={self.input_height} does not "
                    f"match model input shape {image_shape}"
                )
            if (
                isinstance(static_width, int)
                and static_width > 0
                and static_width != self.input_width
            ):
                raise ValueError(
                    f"DEIM config input_width={self.input_width} does not "
                    f"match model input shape {image_shape}"
                )
        target_shape = list(getattr(inputs[1], "shape", []))
        if (
            len(target_shape) == 2
            and isinstance(target_shape[1], int)
            and target_shape[1] != 2
        ):
            raise ValueError(
                "DEIM orig_target_sizes input must end in 2, "
                f"got shape={target_shape}"
            )


def build_deim_detector(
    config: Dict,
    class_registry: Dict,
) -> DeimDetector:
    """Build DEIM with the configured ONNX/ACL backend."""
    backend_name, model_path = resolve_deim_backend(config)
    backend = _build_backend(backend_name, model_path, config)
    return DeimDetector(
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
    raise ValueError(f"unsupported resolved DEIM backend: {backend_name}")
