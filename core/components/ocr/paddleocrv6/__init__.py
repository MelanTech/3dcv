"""Self-contained PP-OCRv6 ONNX/ACL engine and orientation predictor.

    engine = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False, det_model_dir=..., ...)
    boxes, rec_res = engine(bgr_image)   # rec_res: [(text, score), ...]
"""
from .paddleocr import ONNXPaddleOcr
from .predict_doc_orientation import DocOrientationPredictor

__all__ = ["DocOrientationPredictor", "ONNXPaddleOcr"]
