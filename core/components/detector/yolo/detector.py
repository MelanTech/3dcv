"""YOLO 检测器：串联 预处理 → 推理后端 → 输出解码 三个环节。"""
from __future__ import annotations

from typing import Dict, List

from core.components.detector.base import BaseDetector
from core.components.detector.yolo.decoder.base import BaseYoloDecoder
from core.components.detector.yolo.preprocess import preprocess_image
from core.infra.inference.backend.base import BaseInferenceBackend
from core.types import Detection, Frame


class YoloDetector(BaseDetector):
    """通用 YOLO 检测器：后端与解码器可插拔，本类只负责编排。"""

    def __init__(
        self,
        config: Dict,
        class_registry: Dict,
        backend: BaseInferenceBackend,
        decoder: BaseYoloDecoder,
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
        self.decoder = decoder

    def infer(self, frame: Frame, table: int) -> List[Detection]:
        """预处理 RGB → 后端推理 → 解码为图像坐标下的检测框。"""
        if frame.rgb is None:
            raise ValueError(f"{self.detector_type} requires frame.rgb")

        data, pad, original_shape = preprocess_image(
            frame.rgb,
            self.input_width,
            self.input_height,
        )
        outputs = self.backend.execute(data)
        return self.decoder.decode(
            outputs=outputs,
            pad=pad,
            original_shape=original_shape,
            input_shape=(self.input_height, self.input_width),
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            detector_id_to_class=self.detector_id_to_class,
            table=table,
            backend_name=self.backend.name,
        )

    def close(self) -> None:
        self.backend.close()
