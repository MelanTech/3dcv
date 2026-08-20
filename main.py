#!/usr/bin/env python3
"""命令行入口：运行某一个比赛轮次（round1 / round2）。"""
from __future__ import annotations

import argparse
import os
import signal
import sys

from core.app import run_round
from core.warmup import run_warmup


def _raise_keyboard_interrupt(signum, _frame) -> None:
    """把终止信号（SIGTERM 等）转成 KeyboardInterrupt，走和 Ctrl+C 一样的清理流程。"""
    raise KeyboardInterrupt(f"received signal {signum}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数：必选的轮次选择和可选的运行时配置文件路径。"""
    parser = argparse.ArgumentParser(description="3DCV 2026 competition runtime skeleton")
    parser.add_argument(
        "--round",
        choices=("round1", "round2"),
        required=True,
        help="competition round to run",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="path to runtime config",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="initialize model resources and exit without starting a round",
    )
    return parser.parse_args()


def main() -> int:
    """运行选定轮次，并把运行结果/异常翻译成进程退出码（0 成功，130 中断，1 失败）。"""
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    args = parse_args()
    os.environ.setdefault("CV3D_FAST_EXIT_AFTER_STOP", "1")
    try:
        if args.warmup:
            run_warmup(config_path=args.config, round_name=args.round)
            print(f"Warmup finished for {args.round}.")
            return 0
        result_path = run_round(config_path=args.config, round_name=args.round)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Runtime failed: {exc}", file=sys.stderr)
        return 1

    print(f"Finished {args.round}. Result: {result_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code == 0:
        os._exit(0)
    raise SystemExit(exit_code)
