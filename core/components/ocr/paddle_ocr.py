"""PaddleOCR recognition and template classification for OCR candidates."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from rapidfuzz import process

from core.components.ocr.base import BaseOcr
from core.components.ocr.paddleocr import DocOrientationPredictor, ONNXPaddleOcr
from core.components.ocr.paddleocr.model_resolver import resolve_engine_config
from core.types import Detection, Frame


logger = logging.getLogger(__name__)


class PaddleOcr(BaseOcr):
    """Recognize UnknownOcrCandidate crops without changing the public API."""

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        if class_registry is None:
            raise ValueError("PaddleOcr requires shared class_registry config")

        raw_engine_config = dict(config.get("engine", {}))
        doc_model_path = raw_engine_config.pop("doc_orientation_model_dir", None)
        backend, engine_config = resolve_engine_config(raw_engine_config)
        use_gpu = bool(config.get("use_gpu", False))
        self._engine = ONNXPaddleOcr(
            use_angle_cls=bool(config.get("use_angle_cls", True)),
            use_gpu=use_gpu,
            **engine_config,
        )

        self.version = str(config.get("version", "unspecified"))
        self._doc_orientation = None
        self.doc_orientation_min_confidence = float(
            config.get("doc_orientation_min_confidence", 0.8)
        )
        if not 0.0 <= self.doc_orientation_min_confidence <= 1.0:
            raise ValueError(
                "ocr.doc_orientation_min_confidence must be between 0 and 1"
            )
        if bool(config.get("use_doc_orientation", False)):
            self._initialize_doc_orientation(
                doc_model_path=doc_model_path,
                raw_engine_config=raw_engine_config,
                backend=backend,
                use_gpu=use_gpu,
            )

        self.candidate_classes = set(
            class_registry.get("ocr_candidate_classes", ["Book"])
        )
        self.output_classes = list(class_registry.get("ocr_output_classes", []))
        if not self.output_classes:
            raise ValueError("class_registry.ocr_output_classes must not be empty for OCR")

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
        self.min_text_length = int(config.get("min_text_length", 2))
        if self.min_text_length < 1:
            raise ValueError("ocr.min_text_length must be at least 1")

    def _initialize_doc_orientation(
        self,
        doc_model_path,
        raw_engine_config: Dict,
        backend: str,
        use_gpu: bool,
    ) -> None:
        if not doc_model_path:
            logger.warning(
                "Document orientation is enabled but no model is configured; "
                "continuing without four-way correction"
            )
            return
        try:
            doc_config = dict(raw_engine_config)
            doc_config["backend"] = backend
            doc_config["doc_orientation_model_dir"] = doc_model_path
            _, resolved = resolve_engine_config(
                doc_config,
                model_keys=("doc_orientation_model_dir",),
            )
            self._doc_orientation = DocOrientationPredictor(
                resolved["doc_orientation_model_dir"],
                use_gpu=use_gpu,
                config=resolved,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load document orientation model; using the original "
                "OCR flow: %s",
                exc,
            )

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
            details = self._read_details(frame.rgb, detection.bbox)
            text = details["text"]
            if not text:
                continue
            class_name, match_score = self._classify(text)
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
                        "match_score": match_score,
                        "ocr_version": self.version,
                        "ocr_confidence": details["ocr_confidence"],
                        "orientation_angle": details["orientation_angle"],
                        "orientation_confidence": details[
                            "orientation_confidence"
                        ],
                        "orientation_applied": details["orientation_applied"],
                    },
                )
            )
        return results

    def _prepare_crop(self, rgb: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(value) for value in bbox)
        height, width = rgb.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        crop = rgb[y1:y2, x1:x2]
        return crop if crop.size else None

    def _correct_orientation(self, crop: np.ndarray):
        predictor = self._doc_orientation
        if predictor is None:
            return crop, 0, 0.0, False
        try:
            angle, confidence = predictor.predict(crop)
            applied = confidence >= self.doc_orientation_min_confidence
            corrected = (
                predictor.correct_orientation(crop, angle) if applied else crop
            )
            logger.debug(
                "OCR doc orientation angle=%d confidence=%.6f applied=%s",
                angle,
                confidence,
                applied,
            )
            return corrected, int(angle), float(confidence), bool(applied)
        except Exception as exc:
            logger.warning(
                "Document orientation prediction failed; using original crop: %s",
                exc,
            )
            return crop, 0, 0.0, False

    def _read_details(self, rgb, bbox, include_images: bool = False) -> Dict:
        """Run one OCR pass and retain the intermediate results for diagnostics."""
        original_crop = self._prepare_crop(rgb, bbox)
        if original_crop is None:
            return {
                "text": "",
                "boxes": [],
                "recognitions": [],
                "ocr_confidence": 0.0,
                "orientation_angle": 0,
                "orientation_confidence": 0.0,
                "orientation_applied": False,
            }

        crop, angle, orientation_confidence, orientation_applied = (
            self._correct_orientation(original_crop)
        )
        if self.enlarge > 1.0:
            crop = cv2.resize(
                crop,
                dsize=None,
                fx=self.enlarge,
                fy=self.enlarge,
                interpolation=cv2.INTER_LINEAR,
            )

        # Frame.rgb and candidate crops are RGB; PaddleOCR det/rec expect BGR.
        engine_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        boxes, rec_res = self._engine(engine_crop)
        boxes = [] if boxes is None else boxes
        rec_res = [] if rec_res is None else rec_res
        recognitions = [
            {"text": str(text), "confidence": float(score)}
            for text, score in rec_res
        ]
        text = "".join(item["text"] for item in recognitions)
        total_chars = sum(len(item["text"]) for item in recognitions)
        ocr_confidence = (
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
            "ocr_confidence": float(ocr_confidence),
            "orientation_angle": angle,
            "orientation_confidence": orientation_confidence,
            "orientation_applied": orientation_applied,
        }
        if include_images:
            details["original_crop_rgb"] = original_crop
            details["corrected_crop_rgb"] = crop
        return details

    def _read_text(self, rgb, bbox) -> str:
        """Compatibility helper used by existing diagnostics."""
        return str(self._read_details(rgb, bbox)["text"])

    def _classify(self, text: str):
        """Fuzzy-match complete OCR text, rejecting information-poor strings."""
        text = "".join(str(text).split())
        if len(text) < self.min_text_length:
            return None, 0.0
        _matched, score, index = process.extractOne(text, self.templates)
        if score >= self.min_match_score:
            return self.output_classes[index], float(score)
        return None, 0.0

    def close(self) -> None:
        doc_close = getattr(self._doc_orientation, "close", None)
        if doc_close is not None:
            doc_close()
        close = getattr(self._engine, "close", None)
        if close is not None:
            close()
