"""PP-LCNet four-way document orientation inference for RGB OCR crops."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from .session import create_paddleocr_session


DOC_ORIENTATION_ANGLES = (0, 90, 180, 270)


class DocOrientationPredictor:
    """Predict and apply the official 0/90/180/270 correction angle."""

    def __init__(
        self,
        model_dir: str,
        use_gpu: bool = False,
        config: Dict | None = None,
    ):
        self.session = create_paddleocr_session(
            model_dir,
            use_gpu,
            dict(config or {}),
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            self.close()
            raise ValueError("document orientation model requires one input and one output")
        self.input_name = inputs[0].name
        self.output_names = [outputs[0].name]
        try:
            self._validate_interface(inputs[0], outputs[0])
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_interface(input_info, output_info) -> None:
        input_shape = list(input_info.shape)
        output_shape = list(output_info.shape)
        if len(input_shape) != 4:
            raise ValueError(f"document orientation input must be NCHW: {input_shape}")
        for actual, expected in zip(input_shape[1:], (3, 224, 224)):
            if isinstance(actual, int) and actual != expected:
                raise ValueError(
                    "document orientation input must be [N,3,224,224], "
                    f"got {input_shape}"
                )
        if len(output_shape) != 2 or (
            isinstance(output_shape[-1], int) and output_shape[-1] != 4
        ):
            raise ValueError(
                f"document orientation output must be [N,4], got {output_shape}"
            )
        if getattr(input_info, "type", "tensor(float)") != "tensor(float)":
            raise ValueError("document orientation input must be float32")
        if getattr(output_info, "type", "tensor(float)") != "tensor(float)":
            raise ValueError("document orientation output must be float32")

    @staticmethod
    def preprocess(rgb: np.ndarray) -> np.ndarray:
        """Apply the official resize-short-256 and center-crop-224 transform."""
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("document orientation input must be HWC RGB")
        height, width = rgb.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("document orientation input must not be empty")

        scale = 256.0 / min(height, width)
        resized_height = round(height * scale)
        resized_width = round(width * scale)
        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        top = (resized_height - 224) // 2
        left = (resized_width - 224) // 2
        image = resized[top : top + 224, left : left + 224]
        if image.shape[:2] != (224, 224):
            raise ValueError("document orientation center crop must be 224x224")

        image = image.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = ((image - mean) / std).transpose((2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(image, dtype=np.float32)

    def predict(self, rgb: np.ndarray) -> Tuple[int, float]:
        outputs = self.session.run(
            self.output_names,
            {self.input_name: self.preprocess(rgb)},
        )
        if not outputs:
            raise RuntimeError("document orientation model returned no output")
        probabilities = np.asarray(outputs[0], dtype=np.float32)
        if probabilities.shape != (1, 4):
            raise ValueError(
                f"document orientation output must be [1,4], got {probabilities.shape}"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("document orientation output contains non-finite values")
        class_id = int(np.argmax(probabilities[0]))
        return DOC_ORIENTATION_ANGLES[class_id], float(probabilities[0, class_id])

    @staticmethod
    def correct_orientation(rgb: np.ndarray, angle: int) -> np.ndarray:
        """Apply the model label as a counter-clockwise correction angle."""
        if angle == 0:
            return rgb
        if angle == 90:
            return cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle == 180:
            return cv2.rotate(rgb, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        raise ValueError(f"unsupported document orientation angle: {angle}")

    def close(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session.close()
            self.session = None
