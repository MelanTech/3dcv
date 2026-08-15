"""RF-DETR raw ONNX/OM inference component.

Official RF-DETR exports contain preprocessing-independent raw outputs:

- ``dets``: normalized ``cxcywh`` boxes with shape ``(B, Q, 4)``;
- ``labels``: unactivated per-class logits with shape ``(B, Q, C + 1)``.

The final logit column is the implicit no-object class. Inference applies
per-class sigmoid, stable top-k selection over all query/class pairs, and then
the confidence threshold, matching the official export helper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from core.components.detector.base import BaseDetector
from core.components.detector.rf_detr.resolver import (
    resolve_rf_detr_backend,
)
from core.infra.inference.backend.base import BaseInferenceBackend
from core.types import Detection, Frame


class RfDetrDetector(BaseDetector):
    """RF-DETR detector compatible with raw detection ONNX and Ascend OM."""

    def __init__(
        self,
        config: Dict,
        class_registry: Dict,
        backend: BaseInferenceBackend,
        backend_name: str,
        model_path: Path,
    ):
        self.detector_type = str(config.get("type", "rf_detr"))
        self.backend = backend
        self.backend_name = backend_name
        self.model_path = model_path

        default_size = int(config.get("input_size", 640))
        self.input_width = int(config.get("input_width", default_size))
        self.input_height = int(config.get("input_height", default_size))
        self.input_color = str(config.get("input_color", "rgb")).strip().lower()
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("rf_detr.input_color must be rgb or bgr")

        self.normalization_mean = self._channel_values(
            config.get("normalization_mean", [0.485, 0.456, 0.406]),
            "normalization_mean",
        )
        self.normalization_std = self._channel_values(
            config.get("normalization_std", [0.229, 0.224, 0.225]),
            "normalization_std",
        )
        if np.any(self.normalization_std <= 0):
            raise ValueError(
                "rf_detr.normalization_std values must be positive"
            )

        self.conf_thresh = float(config.get("conf_thresh", 0.3))
        configured_num_select = config.get("num_select")
        self.num_select = (
            None
            if configured_num_select is None
            else max(0, int(configured_num_select))
        )
        self.drop_background = bool(config.get("drop_background", True))
        self.strict_class_count = bool(
            config.get("strict_class_count", True)
        )
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
            for name in config.get("output_order", ["dets", "labels"])
        )
        if sorted(self.output_order) != ["dets", "labels"]:
            raise ValueError(
                "rf_detr.output_order must contain dets and labels"
            )
        self._validate_backend_contract()

    def infer(self, frame: Frame, table: int) -> List[Detection]:
        if frame.rgb is None:
            raise ValueError("rf_detr requires frame.rgb")
        data, original_shape = self._preprocess_image(frame.rgb)
        outputs = self.backend.execute(data)
        return self._decode_outputs(outputs, original_shape, table)

    def close(self) -> None:
        self.backend.close()

    def _preprocess_image(
        self,
        rgb: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize directly to model H/W and apply ImageNet normalization."""
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "rf_detr requires HWC 3-channel RGB, "
                f"got shape={image.shape}"
            )
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            if image.max(initial=0.0) > 1.0:
                image /= 255.0
            image = np.clip(image, 0.0, 1.0)
        if self.input_color == "bgr":
            image = image[..., ::-1]

        original_shape = image.shape[:2]
        # OpenCV INTER_LINEAR on float32 uses half-pixel bilinear sampling,
        # matching RF-DETR's antialias=False tensor resize convention.
        image = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image = (
            image - self.normalization_mean
        ) / self.normalization_std
        data = np.transpose(image, (2, 0, 1))[None]
        return np.ascontiguousarray(data, dtype=np.float32), original_shape

    def _decode_outputs(
        self,
        outputs: Sequence[np.ndarray],
        original_shape: Tuple[int, int],
        table: int,
    ) -> List[Detection]:
        boxes, logits = self._normalize_outputs(outputs)
        boxes = np.asarray(boxes)
        logits = np.asarray(logits)
        if boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes[0]
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"unsupported RF-DETR dets output shape: {boxes.shape}"
            )
        if logits.ndim != 2 or logits.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"unsupported RF-DETR labels output shape: {logits.shape}; "
                f"dets shape={boxes.shape}"
            )

        boxes = boxes.astype(np.float32, copy=False)
        logits = logits.astype(np.float32, copy=False)
        if self.drop_background:
            if logits.shape[1] < 2:
                raise ValueError(
                    "RF-DETR labels output has no class column before "
                    "the no-object column"
                )
            logits = logits[:, :-1]

        if self.strict_class_count:
            expected_classes = (
                max(self.detector_id_to_class) + 1
                if self.detector_id_to_class
                else 0
            )
            if logits.shape[1] != expected_classes:
                raise ValueError(
                    f"RF-DETR model exposes {logits.shape[1]} classes after "
                    f"background removal, registry expects {expected_classes}"
                )

        clipped_logits = np.clip(logits, -88.0, 88.0)
        scores_all = 1.0 / (1.0 + np.exp(-clipped_logits))
        if self.allowed_class_ids is not None:
            allowed_mask = np.zeros(
                (scores_all.shape[1],),
                dtype=bool,
            )
            for class_id in self.allowed_class_ids:
                if 0 <= class_id < len(allowed_mask):
                    allowed_mask[class_id] = True
            scores_all = np.where(
                allowed_mask[None, :],
                scores_all,
                -np.inf,
            )

        selection_cap = (
            boxes.shape[0]
            if self.num_select is None
            else self.num_select
        )
        scores, class_ids, query_indices = self._select_topk_multiclass(
            scores_all,
            threshold=self.conf_thresh,
            num_select=selection_cap,
        )
        if len(scores) == 0:
            return []

        selected_boxes = boxes[query_indices]
        cx, cy, width, height = selected_boxes.T
        decoded = np.stack(
            (
                cx - np.maximum(width, 0.0) / 2,
                cy - np.maximum(height, 0.0) / 2,
                cx + np.maximum(width, 0.0) / 2,
                cy + np.maximum(height, 0.0) / 2,
            ),
            axis=1,
        )
        original_height, original_width = original_shape
        decoded *= np.asarray(
            [
                original_width,
                original_height,
                original_width,
                original_height,
            ],
            dtype=np.float32,
        )
        decoded[:, [0, 2]] = np.clip(
            decoded[:, [0, 2]],
            0,
            original_width,
        )
        decoded[:, [1, 3]] = np.clip(
            decoded[:, [1, 3]],
            0,
            original_height,
        )

        detections: List[Detection] = []
        for box, score, class_id in zip(decoded, scores, class_ids):
            if not np.isfinite(score) or not np.isfinite(box).all():
                continue
            x1, y1, x2, y2 = (
                int(round(value)) for value in box
            )
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_id)
            detections.append(
                Detection(
                    class_name=self.detector_id_to_class.get(
                        class_id,
                        f"class_{class_id}",
                    ),
                    class_id=class_id,
                    bbox=(x1, y1, x2, y2),
                    score=float(score),
                    evidence={
                        "table": table,
                        "detector": self.detector_type,
                        "backend": self.backend_name,
                        "weights": str(self.model_path),
                    },
                )
            )
        return detections

    def _normalize_outputs(
        self,
        outputs: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(outputs) != 2:
            raise ValueError(
                f"RF-DETR detection model expects 2 outputs, got {len(outputs)}"
            )
        try:
            output_infos = self.backend.get_outputs()
        except Exception:
            output_infos = []
        if len(output_infos) == len(outputs):
            names = [
                str(info.name).strip().lower()
                for info in output_infos
            ]
            boxes_index = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "dets" in name
                ),
                None,
            )
            logits_index = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "labels" in name
                ),
                None,
            )
            if boxes_index is not None and logits_index is not None:
                return (
                    np.asarray(outputs[boxes_index]),
                    np.asarray(outputs[logits_index]),
                )

        # Shape matching is intentionally avoided: with three foreground
        # classes, logits end in 4 and are indistinguishable from cxcywh boxes.
        by_order = {
            name: np.asarray(outputs[index])
            for index, name in enumerate(self.output_order)
        }
        return by_order["dets"], by_order["labels"]

    @staticmethod
    def _select_topk_multiclass(
        scores_all: np.ndarray,
        threshold: float,
        num_select: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stable top-k over flattened query/class pairs, then threshold."""
        if scores_all.ndim != 2:
            raise ValueError(
                f"RF-DETR scores must have shape (Q,C), got {scores_all.shape}"
            )
        flat_scores = scores_all.reshape(-1)
        if num_select <= 0 or flat_scores.size == 0:
            empty = np.empty((0,), dtype=np.int64)
            return flat_scores[:0], empty, empty

        count = min(int(num_select), len(flat_scores))
        flat_indices = np.arange(len(flat_scores), dtype=np.int64)
        sort_scores = np.where(
            np.isnan(flat_scores),
            np.inf,
            flat_scores,
        )
        selected = np.lexsort(
            (flat_indices, -sort_scores)
        )[:count]
        scores = flat_scores[selected]
        class_count = scores_all.shape[1]
        query_indices = selected // class_count
        class_ids = selected % class_count
        keep = scores > threshold
        return (
            scores[keep],
            class_ids[keep],
            query_indices[keep],
        )

    @staticmethod
    def _channel_values(values, key: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(
                f"rf_detr.{key} must contain exactly 3 values"
            )
        return array.reshape((1, 1, 3))

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
                raise ValueError(
                    f"unknown RF-DETR class filter: {item}"
                )
        return resolved

    def _validate_backend_contract(self) -> None:
        try:
            inputs = self.backend.get_inputs()
        except Exception:
            return
        if len(inputs) != 1:
            raise ValueError(
                f"RF-DETR detection model must expose 1 input, got {len(inputs)}"
            )
        shape = list(getattr(inputs[0], "shape", []))
        if len(shape) != 4:
            return
        channel = shape[1]
        if isinstance(channel, int) and channel != 3:
            raise ValueError(
                f"RF-DETR input channel must be 3, got shape={shape}"
            )
        static_height = shape[2]
        static_width = shape[3]
        if (
            isinstance(static_height, int)
            and static_height > 0
            and static_height != self.input_height
        ):
            raise ValueError(
                f"RF-DETR input_height={self.input_height} does not match "
                f"model shape={shape}"
            )
        if (
            isinstance(static_width, int)
            and static_width > 0
            and static_width != self.input_width
        ):
            raise ValueError(
                f"RF-DETR input_width={self.input_width} does not match "
                f"model shape={shape}"
            )


def build_rf_detr_detector(
    config: Dict,
    class_registry: Dict,
) -> RfDetrDetector:
    """Build RF-DETR with the configured ONNX/ACL backend."""
    backend_name, model_path = resolve_rf_detr_backend(config)
    backend = _build_backend(backend_name, model_path, config)
    return RfDetrDetector(
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
    raise ValueError(
        f"unsupported resolved RF-DETR backend: {backend_name}"
    )
