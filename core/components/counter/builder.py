"""计数器工厂：根据配置里的 type 字段创建对应计数器实现。"""
from __future__ import annotations

from typing import Dict, Optional

from core.components.counter.base import BaseCounter


def build_counter(config: dict, class_registry: Optional[Dict] = None) -> BaseCounter:
    """按 config['type'] 构建计数器；目前仅支持贝叶斯计数器。"""
    counter_type = config["type"]

    if counter_type == "bayesian":
        from core.components.counter.bayesian_counter import BayesianCounter

        return BayesianCounter(config, class_registry)

    if counter_type == "tracker":
        from core.components.counter.tracker_counter import TrackerCounter

        return TrackerCounter(config, class_registry)

    if counter_type == "consensus":
        from core.components.counter.consensus_counter import ConsensusCounter

        return ConsensusCounter(config, class_registry)

    raise ValueError(f"unsupported counter type: {counter_type}")
