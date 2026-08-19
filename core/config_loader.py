"""YAML 配置加载：支持本地 ``!include`` 与旧式 mixin 合并。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


class IncludeLoader(yaml.SafeLoader if yaml is not None else object):
    """自定义 YAML 加载器：以当前文件所在目录为根来解析 ``!include`` 的相对路径。"""

    def __init__(self, stream, root: Path):
        super().__init__(stream)
        self.root = root


def _construct_include(loader: IncludeLoader, node):
    """把 ``!include path/to/file.yaml`` 节点展开为该文件对应的映射内容。"""
    relative_path = loader.construct_scalar(node)
    include_path = (loader.root / relative_path).resolve()
    return _load_yaml_file(include_path)


if yaml is not None:
    IncludeLoader.add_constructor("!include", _construct_include)


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for config !include support. "
            "Install it with: python3 -m pip install PyYAML"
        ) from YAML_IMPORT_ERROR


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """读取单个 YAML 文件，并确保其顶层是一个映射（dict）。"""
    _require_yaml()
    if not path.exists():
        raise ValueError(f"config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    data = yaml.load(text, Loader=lambda stream: IncludeLoader(stream, path.parent))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return data


def _unwrap_same_key_includes(value):
    """Allow ``detector: !include detector/foo.yaml`` with legacy wrapped files."""
    if isinstance(value, list):
        return [_unwrap_same_key_includes(item) for item in value]
    if not isinstance(value, dict):
        return value

    unwrapped = {}
    for key, child in value.items():
        child = _unwrap_same_key_includes(child)
        if isinstance(child, dict) and set(child) == {key}:
            unwrapped[key] = _unwrap_same_key_includes(child[key])
        else:
            unwrapped[key] = child
    return unwrapped


def load_config(path: str) -> Dict[str, Any]:
    """加载根配置，兼容旧 ``mixins`` 与新显式组件 include 风格。"""
    config_path = Path(path).expanduser().resolve()
    root_config = _load_yaml_file(config_path)

    mixins = root_config.pop("mixins", [])
    if mixins is None:
        mixins = []
    if not isinstance(mixins, list):
        raise ValueError("top-level 'mixins' must be a list")

    merged: Dict[str, Any] = dict(root_config)
    for index, mixin in enumerate(mixins):
        if not isinstance(mixin, dict):
            raise ValueError(f"mixin #{index + 1} must resolve to a mapping")
        if len(mixin) != 1:
            keys = ", ".join(sorted(str(key) for key in mixin.keys()))
            raise ValueError(
                f"mixin #{index + 1} must define exactly one top-level domain, got: {keys}"
            )

        domain = next(iter(mixin.keys()))
        if domain in merged:
            raise ValueError(
                f"duplicate config domain '{domain}'. "
                "Each !include mixin must own a distinct top-level domain."
            )
        merged[domain] = mixin[domain]

    return _unwrap_same_key_includes(merged)
