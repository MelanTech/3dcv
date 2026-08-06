class PredictBase(object):
    def __init__(self):
        pass

    def get_onnx_session(self, model_dir, use_gpu):
        """按 args.backend 创建 ONNX Runtime 或 ACL session。

        保留方法名是为了兼容 PaddleOCR 原始代码里的调用点。
        """
        from .session import create_paddleocr_session

        args = getattr(self, "args", None)
        return create_paddleocr_session(model_dir, use_gpu, vars(args) if args is not None else {})


    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    @staticmethod
    def fixed_input_shape(session):
        """返回静态 NCHW 输入 shape；动态维度存在时返回 None。"""
        inputs = session.get_inputs()
        if not inputs:
            return None
        shape = list(inputs[0].shape)
        if len(shape) != 4 or not all(isinstance(dim, int) and dim > 0 for dim in shape):
            return None
        return shape
