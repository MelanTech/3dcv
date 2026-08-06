"""PaddleOCR v5 的 ONNX 移植版（纯 onnxruntime 推理，自 3dcv_2025 迁移）。

本子包是一份自包含的第三方 OCR 引擎，仅依赖 onnxruntime/numpy/opencv/PIL/
shapely/pyclipper，不依赖本项目其它模块。入口为 ``ONNXPaddleOcr``：

    engine = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False, det_model_dir=..., ...)
    boxes, rec_res = engine(bgr_image)   # rec_res: [(text, score), ...]

业务层封装见 core/components/ocr/paddle_ocr.py。
"""
from .paddleocr import ONNXPaddleOcr

__all__ = ["ONNXPaddleOcr"]
