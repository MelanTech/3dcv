"""Original PP-OCRv5 business component kept as an independent factory option."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from rapidfuzz import process

from core.components.ocr.base import BaseOcr
from core.components.ocr.paddleocrv5 import ONNXPaddleOcr
from core.components.ocr.paddleocrv5.model_resolver import resolve_engine_config
from core.types import Detection, Frame


class PaddleOcrV5(BaseOcr):
    """Recognize OCR candidates with the original PP-OCRv5 pipeline."""

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        if class_registry is None:
            raise ValueError("PaddleOcrV5 requires shared class_registry config")

        _backend, engine_config = resolve_engine_config(config.get("engine", {}))
        self._engine = ONNXPaddleOcr(
            use_angle_cls=bool(config.get("use_angle_cls", True)),
            use_gpu=bool(config.get("use_gpu", False)),
            **engine_config,
        )

        self.candidate_classes = set(
            class_registry.get("ocr_candidate_classes", ["Book"])
        )
        self.output_classes = list(class_registry.get("ocr_output_classes", []))
        if not self.output_classes:
            raise ValueError(
                "class_registry.ocr_output_classes must not be empty for OCR"
            )

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
        self.templates = [
            str(ocr_templates[class_name]) for class_name in self.output_classes
        ]

        self.enlarge = float(config.get("enlarge", 1.0))
        self.min_match_score = float(config.get("min_match_score", 0.0))

    def process(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
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

    def _read_details(self, rgb, bbox, include_images: bool = False) -> Dict:
        """Expose v5 intermediates to the version comparison tool."""
        x1, y1, x2, y2 = (int(value) for value in bbox)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return {
                "text": "",
                "boxes": [],
                "recognitions": [],
                "ocr_confidence": 0.0,
                "orientation_angle": 0,
                "orientation_confidence": 0.0,
                "orientation_applied": False,
            }
        if self.enlarge > 1.0:
            crop = cv2.resize(
                crop,
                dsize=None,
                fx=self.enlarge,
                fy=self.enlarge,
                interpolation=cv2.INTER_LINEAR,
            )

        boxes, rec_res = self._engine(crop)
        boxes = [] if boxes is None else boxes
        rec_res = [] if rec_res is None else rec_res
        recognitions = [
            {"text": str(text), "confidence": float(score)}
            for text, score in rec_res
        ]
        text = "".join(item["text"] for item in recognitions)
        total_chars = sum(len(item["text"]) for item in recognitions)
        confidence = (
            sum(
                len(item["text"]) * item["confidence"]
                for item in recognitions
            )
            / total_chars
            if total_chars
            else 0.0
        )
        details = {
            "text": text,
            "boxes": [np.asarray(box).tolist() for box in boxes],
            "recognitions": recognitions,
            "ocr_confidence": float(confidence),
            "orientation_angle": 0,
            "orientation_confidence": 0.0,
            "orientation_applied": False,
        }
        if include_images:
            details["original_crop_rgb"] = crop
            details["corrected_crop_rgb"] = crop
        return details

    def _read_text(self, rgb, bbox) -> str:
        return str(self._read_details(rgb, bbox)["text"])

    def _classify(self, text: str):
        _matched, score, index = process.extractOne(text, self.templates)
        if score > self.min_match_score:
            return self.output_classes[index], float(score)
        return None, 0.0

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if close is not None:
            close()
