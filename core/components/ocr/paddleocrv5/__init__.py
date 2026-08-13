"""Self-contained PP-OCRv5 ONNX/ACL engine.

    engine = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False, det_model_dir=..., ...)
    boxes, rec_res = engine(bgr_image)   # rec_res: [(text, score), ...]
"""
from .paddleocr import ONNXPaddleOcr

__all__ = ["ONNXPaddleOcr"]
