"""昇腾 ACL 推理后端（香橙派 AiPro 等 NPU 平台，运行 .om 模型）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.infra.inference.backend.base import BaseInferenceBackend, TensorInfo


class AclBackend(BaseInferenceBackend):
    """基于 AclLite 运行昇腾 .om 模型的推理后端。"""

    name = "acl"

    def __init__(self, model_path: Path, _config: Dict):
        import acl
        from core.infra.inference.acl.acllite.acllite_model import AclLiteModel
        from core.infra.inference.acl.resource_manager import AclResourceManager

        self._acl = acl
        self.resource_manager = AclResourceManager.instance()
        self.resource_manager.initialize()
        self.model = AclLiteModel(str(model_path))
        self._inputs = self._collect_tensor_infos(kind="input")
        self._outputs = self._collect_tensor_infos(kind="output")

    def execute(self, data: np.ndarray) -> List[np.ndarray]:
        if self.model is None:
            raise RuntimeError("ACL model is not initialized")
        return self.model.execute([data])

    def close(self) -> None:
        if self.model is not None:
            if os.environ.get("CV3D_ACL_FINALIZE", "0").strip().lower() in {"1", "true", "yes", "on"}:
                destroy = getattr(self.model, "destroy", None)
                if destroy is not None:
                    destroy()
            else:
                try:
                    self.model._is_destroyed = True
                except Exception:
                    pass
                try:
                    from core.infra.inference.acl.acllite.acllite_resource_list import resource_list

                    resource_list.unregister(self.model)
                except Exception:
                    pass
            del self.model
            self.model = None

    def get_inputs(self) -> List[TensorInfo]:
        return self._inputs

    def get_outputs(self) -> List[TensorInfo]:
        return self._outputs

    def _collect_tensor_infos(self, kind: str) -> List[TensorInfo]:
        infos: List[TensorInfo] = []
        if kind == "input":
            count = self.model._input_num
            get_dims = self._acl.mdl.get_input_dims
        else:
            count = self.model._output_size
            get_dims = self._acl.mdl.get_output_dims

        for index in range(count):
            dims, _ret = get_dims(self.model._model_desc, index)
            infos.append(
                TensorInfo(
                    name=dims.get("name") or f"{kind}_{index}",
                    shape=list(dims.get("dims", [])),
                )
            )
        return infos
