"""PaddleOCR 文字识别：裁剪候选目标 → OCR → 模板模糊匹配为书本类别。

对应旧项目（3dcv_2025）的 OCRBookCropDetector，适配到当前框架的 BaseOcr 接口：
对上游检出的候选框（默认为 ``Book``）逐个裁剪送入 PaddleOCR ONNX 引擎，拼接识别
出的文本，再用 rapidfuzz 与配置的模板串做模糊匹配，命中则产出对应的书本
物品名称检测项。类别与模板来自共享的 class_registry，阈值等来自 ocr 配置。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
from rapidfuzz import process

from core.components.ocr.base import BaseOcr
from core.components.ocr.paddleocr import ONNXPaddleOcr
from core.components.ocr.paddleocr.model_resolver import resolve_engine_config
from core.types import Detection, Frame


class PaddleOcr(BaseOcr):
    """基于 PaddleOCR ONNX 引擎的书本文字识别与分类。"""

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        if class_registry is None:
            raise ValueError("PaddleOcr requires shared class_registry config")

        _backend, engine_config = resolve_engine_config(config.get("engine", {}))
        # 引擎强制关闭 GPU、开启方向分类，与旧项目保持一致。
        self._engine = ONNXPaddleOcr(
            use_angle_cls=bool(config.get("use_angle_cls", True)),
            use_gpu=bool(config.get("use_gpu", False)),
            **engine_config,
        )

        # 只对这些类别的检测框做 OCR（默认书本）。
        self.candidate_classes = set(class_registry.get("ocr_candidate_classes", ["Book"]))
        self.output_classes = list(class_registry.get("ocr_output_classes", []))
        if not self.output_classes:
            raise ValueError("class_registry.ocr_output_classes must not be empty for OCR")

        # 每个输出类别对应一段模板文本，用于模糊匹配。
        ocr_templates = dict(class_registry.get("ocr_templates", {}))
        missing_templates = [
            class_name
            for class_name in self.output_classes
            if class_name not in ocr_templates
        ]
        if missing_templates:
            raise ValueError(
                "class_registry.ocr_templates must define every OCR output class: "
                + ", ".join(missing_templates)
            )
        self.templates = [str(ocr_templates[class_name]) for class_name in self.output_classes]

        self.enlarge = float(config.get("enlarge", 1.0))
        self.min_match_score = float(config.get("min_match_score", 0.0))

    def process(self, frame: Frame, detections: List[Detection], table: int) -> List[Detection]:
        """对候选框做 OCR 并模板匹配，返回命中类别的新增检测。"""
        if frame.rgb is None:
            return []

        results: List[Detection] = []
        for detection in detections:
            if detection.class_name not in self.candidate_classes:
                continue

            text = self._read_text(frame.rgb, detection.bbox)
            if not text:
                continue

            class_name, score = self._classify(text)
            if class_name is None:
                continue

            results.append(
                Detection(
                    class_name=class_name,
                    bbox=detection.bbox,
                    score=1.0,
                    evidence={
                        "source": "ocr",
                        "table": table,
                        "text": text,
                        "match_score": score,
                    },
                )
            )
        return results

    def _read_text(self, rgb, bbox) -> str:
        """裁剪 bbox 区域跑 OCR，把该区域内识别出的所有文本拼接成一个串。"""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        if self.enlarge > 1.0:
            crop = cv2.resize(
                crop,
                dsize=None,
                fx=self.enlarge,
                fy=self.enlarge,
                interpolation=cv2.INTER_LINEAR,
            )

        _boxes, rec_res = self._engine(crop)
        if not rec_res:
            return ""
        return "".join(text for text, _score in rec_res)

    def _classify(self, text: str):
        """用 rapidfuzz 把识别文本模糊匹配到模板，返回 (类别名, 相似度) 或 (None, 0)。"""
        _matched, score, index = process.extractOne(text, self.templates)
        if score > self.min_match_score:
            return self.output_classes[index], float(score)
        return None, 0.0

    def close(self) -> None:
        """释放 PaddleOCR det/rec/cls 后端资源。"""
        close = getattr(self._engine, "close", None)
        if close is not None:
            close()
