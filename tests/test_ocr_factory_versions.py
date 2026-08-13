from __future__ import annotations

import unittest
from pathlib import Path

from core.components.ocr.builder import build_ocr
from core.components.ocr.paddleocrv5.component import PaddleOcrV5
from core.components.ocr.paddleocrv6.component import PaddleOcrV6
from core.config_loader import load_config


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_config(
    str(ROOT / "config" / "class_registry" / "default.yaml")
)["class_registry"]


class OcrFactoryVersionsTest(unittest.TestCase):
    def test_factory_builds_independent_v5_component(self):
        config = load_config(
            str(ROOT / "config" / "ocr" / "paddleocrv5.yaml")
        )["ocr"]
        ocr = build_ocr(config, REGISTRY)
        try:
            self.assertIsInstance(ocr, PaddleOcrV5)
            self.assertEqual(
                "core.components.ocr.paddleocrv5.paddleocr",
                ocr._engine.__class__.__module__,
            )
        finally:
            ocr.close()

    def test_factory_builds_independent_v6_component(self):
        config = load_config(
            str(ROOT / "config" / "ocr" / "paddleocrv6.yaml")
        )["ocr"]
        ocr = build_ocr(config, REGISTRY)
        try:
            self.assertIsInstance(ocr, PaddleOcrV6)
            self.assertEqual(
                "core.components.ocr.paddleocrv6.paddleocr",
                ocr._engine.__class__.__module__,
            )
            self.assertIsNotNone(ocr._doc_orientation)
        finally:
            ocr.close()

    def test_legacy_paddle_type_remains_available(self):
        config = load_config(
            str(ROOT / "config" / "ocr" / "paddle.yaml")
        )["ocr"]
        ocr = build_ocr(config, REGISTRY)
        try:
            self.assertEqual("PaddleOcr", type(ocr).__name__)
            self.assertEqual(
                "core.components.ocr.paddleocr.paddleocr",
                ocr._engine.__class__.__module__,
            )
        finally:
            ocr.close()


if __name__ == "__main__":
    unittest.main()
