from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.components.ocr.paddleocrv6.predict_doc_orientation import (
    DOC_ORIENTATION_ANGLES,
    DocOrientationPredictor,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "doc_orientation" / "inference.onnx"


class _TensorInfo:
    def __init__(self, name, shape, tensor_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _FakeSession:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)
        self.inputs = [_TensorInfo("x", ["batch", 3, 224, 224])]
        self.outputs = [_TensorInfo("fetch_name_0", ["batch", 4])]
        self.closed = False

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, _output_names, _input_feed):
        return [self.output]

    def close(self):
        self.closed = True


def predictor(output):
    session = _FakeSession(output)
    with patch(
        "core.components.ocr.paddleocrv6.predict_doc_orientation."
        "create_paddleocr_session",
        return_value=session,
    ):
        instance = DocOrientationPredictor("unused.onnx")
    return instance, session


class DocOrientationTest(unittest.TestCase):
    def test_official_rgb_preprocess(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        tensor = DocOrientationPredictor.preprocess(rgb)
        self.assertEqual((1, 3, 224, 224), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)
        self.assertAlmostEqual((1 - 0.485) / 0.229, tensor[0, 0, 0, 0], 5)
        self.assertAlmostEqual((0 - 0.456) / 0.224, tensor[0, 1, 0, 0], 5)

    def test_class_mapping(self):
        rgb = np.zeros((32, 64, 3), dtype=np.uint8)
        for class_id, expected in enumerate(DOC_ORIENTATION_ANGLES):
            values = np.full((1, 4), 0.1, dtype=np.float32)
            values[0, class_id] = 0.7
            instance, _session = predictor(values)
            angle, confidence = instance.predict(rgb)
            self.assertEqual(expected, angle)
            self.assertAlmostEqual(0.7, confidence, 6)

    def test_rotation_semantics(self):
        rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        self.assertIs(rgb, DocOrientationPredictor.correct_orientation(rgb, 0))
        np.testing.assert_array_equal(
            np.rot90(rgb, 1),
            DocOrientationPredictor.correct_orientation(rgb, 90),
        )
        np.testing.assert_array_equal(
            np.rot90(rgb, 2),
            DocOrientationPredictor.correct_orientation(rgb, 180),
        )
        np.testing.assert_array_equal(
            np.rot90(rgb, 3),
            DocOrientationPredictor.correct_orientation(rgb, 270),
        )

    @unittest.skipUnless(MODEL_PATH.is_file(), "orientation ONNX is absent")
    def test_real_model_interface(self):
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        self.assertEqual("x", session.get_inputs()[0].name)
        self.assertEqual([3, 224, 224], session.get_inputs()[0].shape[1:])
        self.assertEqual(4, session.get_outputs()[0].shape[-1])


if __name__ == "__main__":
    unittest.main()
