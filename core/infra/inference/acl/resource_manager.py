"""ACL 资源管理器：进程级单例，统一初始化昇腾运行时与 DVPP。"""
from __future__ import annotations

import atexit
import os
from threading import Lock
from typing import Optional


class AclResourceManager:
    """昇腾 ACL 运行时的线程安全单例，保证资源只初始化一次、退出时释放。"""

    _instance: Optional["AclResourceManager"] = None
    _instance_lock = Lock()

    def __init__(self):
        self._lock = Lock()
        self._initialized = False
        self.resource = None
        self.dvpp = None

    @classmethod
    def instance(cls) -> "AclResourceManager":
        """获取全局单例，并在首次创建时注册退出清理。"""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
                atexit.register(cls._instance.close)
            return cls._instance

    def initialize(self) -> None:
        """初始化 ACL 资源与 DVPP（幂等，重复调用无副作用）。"""
        with self._lock:
            if self._initialized:
                return

            from core.infra.inference.acl.acllite.acllite_imageproc import AclLiteImageProc
            from core.infra.inference.acl.acllite.acllite_resource import AclLiteResource

            self.resource = AclLiteResource()
            self.resource.init()
            self.dvpp = AclLiteImageProc(self.resource)
            self._initialized = True

    def close(self) -> None:
        """关闭 ACL runtime；默认跳过 native finalize 以避开 OpenNI 共存时的析构崩溃。"""
        with self._lock:
            if not self._should_finalize_native():
                self._detach_without_native_finalize()
                self._initialized = False
                return

            if self.dvpp is not None:
                del self.dvpp
                self.dvpp = None
            if self.resource is not None:
                del self.resource
                self.resource = None
            self._initialized = False

    @staticmethod
    def _should_finalize_native() -> bool:
        value = os.environ.get("CV3D_ACL_FINALIZE", "0").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _detach_without_native_finalize(self) -> None:
        """让 Python 放弃 ACL/DVPP 对象引用，但不执行易崩的 native 析构。"""
        try:
            from core.infra.inference.acl.acllite.acllite_resource_list import (
                UNREGISTER,
                resource_list,
            )

            for item in resource_list.resources:
                item["status"] = UNREGISTER
                try:
                    item["resource"]._is_destroyed = True
                except Exception:
                    pass
        except Exception:
            pass

        if self.dvpp is not None:
            try:
                self.dvpp._is_destroyed = True
            except Exception:
                pass
            self.dvpp = None

        if self.resource is not None:
            try:
                self.resource._skip_destroy = True
            except Exception:
                pass
            self.resource = None
