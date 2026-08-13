from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort

from core.config_loader import load_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "ppocrv6"


@unittest.skipUnless(
    (MODEL_DIR / "det.onnx").is_file() and (MODEL_DIR / "rec.onnx").is_file(),
    "PP-OCRv6 small ONNX models are absent",
)
class PpOcrV6ModelsTest(unittest.TestCase):
    def test_detection_interface(self):
        session = ort.InferenceSession(
            str(MODEL_DIR / "det.onnx"), providers=["CPUExecutionProvider"]
        )
        model_input = session.get_inputs()[0]
        model_output = session.get_outputs()[0]
        self.assertEqual("tensor(float)", model_input.type)
        self.assertEqual(4, len(model_input.shape))
        self.assertEqual(3, model_input.shape[1])
        output = session.run(
            [model_output.name],
            {model_input.name: np.zeros((1, 3, 32, 64), dtype=np.float32)},
        )[0]
        self.assertEqual((1, 1, 32, 64), output.shape)

    def test_recognition_interface_and_dictionary(self):
        session = ort.InferenceSession(
            str(MODEL_DIR / "rec.onnx"), providers=["CPUExecutionProvider"]
        )
        model_input = session.get_inputs()[0]
        model_output = session.get_outputs()[0]
        self.assertEqual([3, 48], model_input.shape[1:3])
        self.assertEqual(18710, model_output.shape[2])
        dictionary = (MODEL_DIR / "ppocrv6_dict.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(18708, len(dictionary))

    def test_active_config_really_selects_v6(self):
        config = load_config(str(ROOT / "config" / "ocr" / "paddle.yaml"))["ocr"]
        self.assertEqual("ppocrv6-small", config["version"])
        self.assertTrue(config["use_doc_orientation"])
        self.assertIn("ppocrv6", config["engine"]["det_model_dir"])
        self.assertIn("ppocrv6", config["engine"]["rec_model_dir"])
        self.assertEqual(0.2, config["engine"]["det_db_thresh"])
        self.assertEqual(0.45, config["engine"]["det_db_box_thresh"])
        self.assertEqual(1.4, config["engine"]["det_db_unclip_ratio"])


if __name__ == "__main__":
    unittest.main()
