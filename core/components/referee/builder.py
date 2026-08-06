"""裁判客户端工厂：从 referee/team 配置构建 socket 客户端。"""
from __future__ import annotations

import os

from core.components.referee.base import BaseRefereeClient
from core.components.referee.socket_client import RefereeSocketClient
from core.infra.logging.event_logger import EventLogger


def build_referee_client(
    config: dict,
    team_config: dict,
    logger: EventLogger,
    class_registry: dict | None = None,
) -> BaseRefereeClient:
    """构建基于 TCP socket 的裁判客户端；结果目录可被环境变量覆盖。"""
    class_registry = class_registry or {}
    return RefereeSocketClient(
        enabled=config["enabled"],
        ip=config["ip"],
        port=config["port"],
        team_id=team_config["team_id"],
        result_base_dir=os.environ.get("3DCV_RESULT_DIR", config["result_base_dir"]),
        result_file_prefix=team_config["file_prefix"],
        connect_retry_sec=config["connect_retry_sec"],
        connect_timeout_sec=config["connect_timeout_sec"],
        logger=logger,
        result_class_to_goal_id=class_registry.get("result_class_to_goal_id", {}),
    )
