"""进程级"暂停时钟"：提供一个把可视化暂停时长扣除后的单调时间。

Depth filter 的可视化窗口暂停时，会阻塞整个流水线，但轮次计时用的是
wall-clock（time.monotonic）的绝对 deadline，暂停太久就会把轮次直接跑超时退出。
本模块提供一个共享的单调时钟：暂停期间时间"冻结"，因此基于它的 deadline 会
在暂停后自动顺延，从而让 step/暂停观察不再触发计时退出。

用法：
- 计时方（frame_collector）用 ``now()`` 取时间、算 deadline，替代 time.monotonic()。
- 暂停方（depth_filter 可视化）在进入/退出暂停时调用 ``begin_pause()`` /
  ``end_pause()``（可重入，用计数保护）。
"""
from __future__ import annotations

import time
import threading


_lock = threading.Lock()
_paused_depth = 0
_pause_started_at = 0.0
_accumulated_pause = 0.0


def now() -> float:
    """返回扣除累计暂停时长后的单调时间（秒）。"""
    with _lock:
        base = time.monotonic() - _accumulated_pause
        if _paused_depth > 0:
            # 暂停进行中：把当前这段暂停也一并扣掉，使时间在暂停期间冻结。
            base -= time.monotonic() - _pause_started_at
        return base


def begin_pause() -> None:
    """标记进入暂停；可重入（嵌套调用只在最外层真正开始计时）。"""
    global _paused_depth, _pause_started_at
    with _lock:
        if _paused_depth == 0:
            _pause_started_at = time.monotonic()
        _paused_depth += 1


def end_pause() -> None:
    """标记退出暂停；最外层退出时把这段暂停累加进总暂停时长。"""
    global _paused_depth, _accumulated_pause
    with _lock:
        if _paused_depth == 0:
            return
        _paused_depth -= 1
        if _paused_depth == 0:
            _accumulated_pause += time.monotonic() - _pause_started_at
