from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.components.ocr.paddle_ocr import PaddleOcr
from core.components.ocr.paddleocr.predict_doc_orientation import (
    DocOrientationPredictor,
)
from core.types import Detection, Frame


class _FakeEngine:
    drop_score = 0.5

    def __init__(self, text="语文"):
        self.text = text
        self.crops = []
        self.closed = False

    def __call__(self, crop):
        self.crops.append(crop.copy())
        box = np.asarray(
            [[[0, 0], [crop.shape[1] - 1, 0], [crop.shape[1] - 1, 10], [0, 10]]]
        )
        return box, [(self.text, 0.9)] if self.text else []

    def close(self):
        self.closed = True


class _FakeOrientation:
    def __init__(self, angle=90, confidence=0.95):
        self.angle = angle
        self.confidence = confidence
        self.inputs = []
        self.closed = False

    def predict(self, crop):
        self.inputs.append(crop.copy())
        return self.angle, self.confidence

    def correct_orientation(self, crop, angle):
        return DocOrientationPredictor.correct_orientation(crop, angle)

    def close(self):
        self.closed = True


def build_ocr(text="语文"):
    ocr = object.__new__(PaddleOcr)
    ocr._engine = _FakeEngine(text)
    ocr.version = "test"
    ocr._doc_orientation = None
    ocr.doc_orientation_min_confidence = 0.8
    ocr.candidate_classes = {"UnknownOcrCandidate"}
    ocr.output_classes = ["W001", "W002", "W003", "W004"]
    ocr.templates = [
        "语文阅读写作",
        "高等数学线性代数",
        "英语外语单词语法",
        "自然物理化学科学",
    ]
    ocr.enlarge = 1.0
    ocr.min_match_score = 60.0
    ocr.min_text_length = 2
    return ocr


class PaddleOcrUpgradeTest(unittest.TestCase):
    def test_disabled_orientation_keeps_old_flow(self):
        config = {
            "version": "test",
            "use_doc_orientation": False,
            "engine": {},
        }
        registry = {
            "ocr_candidate_classes": ["UnknownOcrCandidate"],
            "ocr_output_classes": ["W001"],
            "ocr_templates": {"W001": "语文阅读写作"},
        }
        fake_engine = _FakeEngine()
        with patch(
            "core.components.ocr.paddle_ocr.resolve_engine_config",
            return_value=("onnx", {}),
        ), patch(
            "core.components.ocr.paddle_ocr.ONNXPaddleOcr",
            return_value=fake_engine,
        ), patch(
            "core.components.ocr.paddle_ocr.DocOrientationPredictor"
        ) as orientation_class:
            ocr = PaddleOcr(config, registry)

        self.assertIsNone(ocr._doc_orientation)
        orientation_class.assert_not_called()

    def test_orientation_load_failure_falls_back(self):
        config = {
            "version": "test",
            "use_doc_orientation": True,
            "engine": {"doc_orientation_model_dir": "missing.onnx"},
        }
        registry = {
            "ocr_candidate_classes": ["UnknownOcrCandidate"],
            "ocr_output_classes": ["W001"],
            "ocr_templates": {"W001": "语文阅读写作"},
        }
        with patch(
            "core.components.ocr.paddle_ocr.resolve_engine_config",
            side_effect=[("onnx", {}), FileNotFoundError("missing model")],
        ), patch(
            "core.components.ocr.paddle_ocr.ONNXPaddleOcr",
            return_value=_FakeEngine(),
        ), self.assertLogs(
            "core.components.ocr.paddle_ocr", level="WARNING"
        ) as logs:
            ocr = PaddleOcr(config, registry)

        self.assertIsNone(ocr._doc_orientation)
        self.assertIn("using the original OCR flow", " ".join(logs.output))

    def test_single_character_is_rejected(self):
        self.assertEqual((None, 0.0), build_ocr()._classify("语"))

    def test_complete_text_is_classified_with_new_mapping(self):
        self.assertEqual("W001", build_ocr()._classify("语文")[0])
        self.assertEqual("W002", build_ocr()._classify("数学")[0])

    def test_candidate_is_oriented_then_converted_once_to_bgr(self):
        ocr = build_ocr()
        orientation = _FakeOrientation(angle=90)
        ocr._doc_orientation = orientation
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        rgb[:, :, 0] = 11
        rgb[:, :, 1] = 22
        rgb[:, :, 2] = 33

        details = ocr._read_details(rgb, (0, 0, 30, 20))

        expected = cv2.cvtColor(np.rot90(rgb, 1), cv2.COLOR_RGB2BGR)
        np.testing.assert_array_equal(expected, ocr._engine.crops[0])
        self.assertEqual(90, details["orientation_angle"])
        self.assertTrue(details["orientation_applied"])

    def test_low_confidence_keeps_original_crop(self):
        ocr = build_ocr()
        ocr._doc_orientation = _FakeOrientation(angle=90, confidence=0.79)
        rgb = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        ocr._read_details(rgb, (0, 0, 30, 20))
        np.testing.assert_array_equal(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), ocr._engine.crops[0]
        )

    def test_process_preserves_detection_contract(self):
        ocr = build_ocr()
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        frame = Frame("frame", rgb, None, 0.0)
        candidate = Detection(
            class_name="UnknownOcrCandidate",
            bbox=(0, 0, 30, 20),
            score=0.8,
        )
        results = ocr.process(frame, [candidate], table=1)
        self.assertEqual(1, len(results))
        self.assertEqual("W001", results[0].class_name)
        self.assertEqual(candidate.bbox, results[0].bbox)
        self.assertEqual("语文", results[0].evidence["text"])

    def test_non_candidate_does_not_run_orientation_or_ocr(self):
        ocr = build_ocr()
        orientation = _FakeOrientation(angle=90)
        ocr._doc_orientation = orientation
        frame = Frame("frame", np.zeros((20, 30, 3), dtype=np.uint8), None, 0.0)
        known_detection = Detection(
            class_name="Cup",
            bbox=(0, 0, 30, 20),
            score=0.8,
        )

        self.assertEqual([], ocr.process(frame, [known_detection], table=1))
        self.assertEqual([], orientation.inputs)
        self.assertEqual([], ocr._engine.crops)


if __name__ == "__main__":
    unittest.main()
