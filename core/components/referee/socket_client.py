"""裁判盒 TCP socket 客户端：结果文件生成与按协议收发信号。

协议采用 ``>ii`` 大端头部（数据类型 + 长度）加 UTF-8 负载。
enabled=False 时进入“空跑”模式：只写文件、不真正联网，便于本地调试。
"""
from __future__ import annotations

import os
import socket
import struct
import time
from pathlib import Path
from typing import Dict, Iterable
from typing import Optional

from core.components.referee.base import BaseRefereeClient
from core.infra.logging.event_logger import EventLogger
from core.types import RecognitionItem


class RefereeSocketClient(BaseRefereeClient):
    """通过 TCP socket 与裁判盒通信的客户端实现。"""

    def __init__(
        self,
        enabled: bool,
        ip: str,
        port: int,
        team_id: str,
        result_base_dir: str,
        result_file_prefix: str,
        connect_retry_sec: float,
        connect_timeout_sec: float,
        logger: EventLogger,
        result_class_to_goal_id: Dict[str, str] | None = None,
    ):
        self.enabled = enabled
        self.ip = ip
        self.port = int(port)
        self.team_id = team_id
        self.result_base_dir = Path(result_base_dir).expanduser()
        self.result_base_dir.mkdir(parents=True, exist_ok=True)
        self.result_file_prefix = result_file_prefix
        self.connect_retry_sec = float(connect_retry_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.logger = logger
        self.socket: Optional[socket.socket] = None
        self.result_class_to_goal_id = dict(result_class_to_goal_id or {})

    def result_path(self, round_number: int) -> Path:
        return self.result_base_dir / f"{self.result_file_prefix}-R{round_number}.txt"

    def render_result(self, items: Iterable[RecognitionItem]) -> str:
        """渲染成 START/END 包裹的结果文本，跳过数量为 0 的条目。"""
        lines = ["START"]
        for item in items:
            if item.num <= 0:
                continue
            goal_id = self.result_class_to_goal_id.get(item.goal_id, item.goal_id)
            lines.append(f"Goal_ID={goal_id};Num={item.num};Table={item.table}")
        lines.append("END")
        return "\n".join(lines) + "\n"

    def write_result(self, round_number: int, items: Iterable[RecognitionItem]) -> Path:
        """先写临时文件再原子替换，避免裁判读到写了一半的结果。"""
        path = self.result_path(round_number)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(self.render_result(items), encoding="utf-8")
        os.replace(tmp_path, path)
        return path

    def connect(self) -> bool:
        """在超时时间内反复尝试建立 TCP 连接；未启用时直接视为成功。"""
        if not self.enabled:
            self.logger.event("referee_connect_skipped", enabled=False)
            return True

        deadline = time.monotonic() + self.connect_timeout_sec
        while time.monotonic() <= deadline:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.ip, self.port))
                self.logger.event("referee_connected", ip=self.ip, port=self.port)
                return True
            except OSError as exc:
                self.logger.event("referee_connect_failed", error=str(exc))
                self.socket = None
                time.sleep(self.connect_retry_sec)

        return False

    def pack_data(self, data_type: int, data_content: str) -> bytes:
        """按协议打包：大端 (数据类型, 负载长度) 头部 + UTF-8 负载。"""
        if data_type not in (0, 1, 2, 3):
            raise ValueError(f"unsupported referee socket data_type: {data_type}")
        data_bytes = data_content.encode("utf-8")
        header = struct.pack(">ii", data_type, len(data_bytes))
        return header + data_bytes

    def _send(self, data_type: int, content: str, event: str) -> bool:
        """统一的发送逻辑：未启用时空跑，未连接则失败，异常记录后返回 False。"""
        if not self.enabled:
            self.logger.event(event, enabled=False, bytes=len(content.encode("utf-8")))
            return True
        if self.socket is None:
            self.logger.event(f"{event}_failed", error="referee socket is not connected")
            return False
        try:
            self.socket.sendall(self.pack_data(data_type, content))
            self.logger.event(event, enabled=True, data_type=data_type)
            return True
        except OSError as exc:
            self.logger.event(f"{event}_failed", error=str(exc))
            return False

    def send_start(self) -> bool:
        """发送开始信号（数据类型 0，负载为队伍 ID）。"""
        return self._send(0, str(self.team_id), "referee_start_sent")

    def send_rotate(self) -> bool:
        """发送旋转/换桌信号（数据类型 3）。"""
        return self._send(3, "0000", "referee_rotate_sent")

    def send_result_file(self, path: Path) -> bool:
        """把结果文件内容作为数据类型 1 发送。"""
        return self._send(1, path.read_text(encoding="utf-8"), "referee_result_sent")

    def close(self) -> None:
        """关闭 socket 连接（若已建立）。"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
            self.logger.event("referee_closed")
