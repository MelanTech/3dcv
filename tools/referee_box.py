#!/usr/bin/env python3
"""Local referee-box replica for 3DCV round evaluation."""

from __future__ import annotations

import argparse
import re
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml


HEADER = struct.Struct(">ii")
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
GOAL_ID_RE = re.compile(r"(?:C[ABCD]|W)\d{3}")
TEAM_ID_RE = re.compile(r"[A-Za-z0-9]+")
RESULT_LINE_RE = re.compile(
    r"Goal_ID=((?:C[ABCD]|W)\d{3});Num=([0-9]+);Table=([0-9]+)"
)

Key = Tuple[int, str]


@dataclass(frozen=True)
class RoundRules:
    min_time: float
    max_time: float
    min_prop: float
    table_count: int


ROUND_RULES = {
    "round1": RoundRules(min_time=30.0, max_time=75.0, min_prop=0.3, table_count=1),
    "round2": RoundRules(min_time=75.0, max_time=170.0, min_prop=0.3, table_count=3),
}


@dataclass(frozen=True)
class ItemScore:
    table: int
    goal_id: str
    truth: int
    prediction: int
    score: float
    reason: str


@dataclass(frozen=True)
class ScoreReport:
    item_scores: Tuple[ItemScore, ...]
    measure_score: float
    full_score: float
    average_score: float
    total_time: float
    time_weight: float
    time_score: float
    raw_total_score: float
    deduction_rate: float
    total_score: float


@dataclass(frozen=True)
class SessionResult:
    team_id: Optional[str]
    predictions: Optional[Dict[Key, int]]
    total_time: Optional[float]
    rotate_times: Tuple[float, ...]
    source: str
    error: Optional[str]
    exit_code: int


class ProtocolError(RuntimeError):
    """The client violated the referee-box wire or result protocol."""


class RoundTimeout(TimeoutError):
    """The client did not submit a result before MaxTime."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3DCV local referee-box replica")
    parser.add_argument("--round", choices=tuple(ROUND_RULES), required=True)
    parser.add_argument("--case", type=Path, help="absolute counts directory for round1")
    parser.add_argument(
        "--cases",
        nargs=3,
        type=Path,
        metavar=("TABLE1", "TABLE2", "TABLE3"),
        help="three absolute counts directories for round2",
    )
    parser.add_argument("--host", default="0.0.0.0", help="IPv4 address to bind")
    parser.add_argument("--port", type=int, default=6666, help="TCP port to bind")
    parser.add_argument(
        "--fallback-result",
        type=Path,
        help="local result file used after MaxTime, with a 10%% deduction",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be in the range 1..65535")

    if args.round == "round1":
        if args.case is None or args.cases is not None:
            parser.error("round1 requires --case and does not accept --cases")
        case_paths = [args.case]
    else:
        if args.cases is None or args.case is not None:
            parser.error("round2 requires exactly three --cases and does not accept --case")
        case_paths = list(args.cases)

    for case_path in case_paths:
        if not case_path.is_absolute():
            parser.error(f"case path must be absolute: {case_path}")

    args.case_paths = tuple(path.resolve() for path in case_paths)
    if args.fallback_result is not None:
        args.fallback_result = args.fallback_result.expanduser().resolve()
    return args


def load_counts(case_path: Path) -> Dict[str, int]:
    counts_path = case_path / "counts.yaml"
    if not case_path.is_dir():
        raise ValueError(f"case directory does not exist: {case_path}")
    if not counts_path.is_file():
        raise ValueError(f"counts.yaml does not exist: {counts_path}")

    try:
        raw = yaml.safe_load(counts_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to load {counts_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ValueError(f"{counts_path} must contain a YAML mapping")

    counts: Dict[str, int] = {}
    for raw_goal_id, raw_count in raw.items():
        goal_id = str(raw_goal_id)
        if GOAL_ID_RE.fullmatch(goal_id) is None:
            raise ValueError(f"invalid goal ID in {counts_path}: {goal_id!r}")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(
                f"count for {goal_id} in {counts_path} must be a non-negative integer"
            )
        counts[goal_id] = raw_count
    return counts


def load_truth(case_paths: Iterable[Path]) -> Dict[Key, int]:
    truth: Dict[Key, int] = {}
    for table, case_path in enumerate(case_paths, start=1):
        for goal_id, count in load_counts(case_path).items():
            if count > 0:
                truth[(table, goal_id)] = count
    if not truth:
        raise ValueError("truth contains no object category with a positive count")
    return truth


def parse_result_text(text: str) -> Dict[Key, int]:
    lines = text.splitlines()
    if len(lines) < 2 or lines[0] != "START" or lines[-1] != "END":
        raise ProtocolError("result must start with START and end with END")
    if any(not line for line in lines):
        raise ProtocolError("result must not contain blank lines")

    predictions: Dict[Key, int] = {}
    for line_number, line in enumerate(lines[1:-1], start=2):
        match = RESULT_LINE_RE.fullmatch(line)
        if match is None:
            raise ProtocolError(f"invalid result line {line_number}: {line!r}")
        goal_id, raw_num, raw_table = match.groups()
        num = int(raw_num)
        table = int(raw_table)
        if num <= 0:
            raise ProtocolError(f"Num must be positive on result line {line_number}")
        if table <= 0:
            raise ProtocolError(f"Table must be positive on result line {line_number}")
        key = (table, goal_id)
        if key in predictions:
            raise ProtocolError(
                f"duplicate result for Goal_ID={goal_id}, Table={table}"
            )
        predictions[key] = num
    return predictions


def read_result_file(path: Path) -> Dict[Key, int]:
    try:
        return parse_result_text(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"failed to read fallback result {path}: {exc}") from exc


def score_predictions(
    truth: Mapping[Key, int],
    predictions: Mapping[Key, int],
    rules: RoundRules,
    total_time: float,
    deduction_rate: float = 0.0,
) -> ScoreReport:
    item_scores = []
    measure_score = 0.0

    for key in sorted(truth):
        table, goal_id = key
        truth_num = truth[key]
        prediction = predictions.get(key, 0)
        if prediction == truth_num:
            score = 3.0
            reason = "exact"
        elif 0 < prediction < truth_num:
            score = prediction / truth_num * 3.0
            reason = "undercount"
        elif prediction > truth_num:
            score = 0.0
            reason = "overcount"
        else:
            score = 0.0
            reason = "missing"
        measure_score += score
        item_scores.append(
            ItemScore(table, goal_id, truth_num, prediction, score, reason)
        )

    for key in sorted(set(predictions) - set(truth)):
        table, goal_id = key
        score = -3.0
        measure_score += score
        item_scores.append(
            ItemScore(table, goal_id, 0, predictions[key], score, "false_detection")
        )

    full_score = len(truth) * 3.0
    average_score = measure_score / len(truth)
    score_prop = measure_score / full_score
    time_weight = score_prop if score_prop >= rules.min_prop else 0.0

    if total_time <= rules.min_time:
        time_score = 3.0 * time_weight
    elif total_time >= rules.max_time:
        time_score = 0.0
    else:
        time_ratio = (rules.max_time - total_time) / (
            rules.max_time - rules.min_time
        )
        time_score = time_ratio * 3.0 * time_weight

    raw_total_score = average_score + time_score
    total_score = raw_total_score * (1.0 - deduction_rate)
    return ScoreReport(
        item_scores=tuple(item_scores),
        measure_score=measure_score,
        full_score=full_score,
        average_score=average_score,
        total_time=total_time,
        time_weight=time_weight,
        time_score=time_score,
        raw_total_score=raw_total_score,
        deduction_rate=deduction_rate,
        total_score=total_score,
    )


def recv_exact(
    client: socket.socket,
    size: int,
    deadline: Optional[float],
) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if deadline is None:
            client.settimeout(None)
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RoundTimeout("MaxTime reached before a result was received")
            client.settimeout(remaining)
        try:
            chunk = client.recv(size - len(chunks))
        except socket.timeout as exc:
            raise RoundTimeout("MaxTime reached before a result was received") from exc
        if not chunk:
            raise ProtocolError("client disconnected before submitting a result")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_packet(
    client: socket.socket,
    deadline: Optional[float],
) -> Tuple[int, bytes]:
    header = recv_exact(client, HEADER.size, deadline)
    data_type, data_length = HEADER.unpack(header)
    if data_length < 0 or data_length > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"invalid DataLength: {data_length}")
    payload = recv_exact(client, data_length, deadline)
    return data_type, payload


def decode_utf8(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"{label} payload is not valid UTF-8") from exc


def run_session(
    client: socket.socket,
    rules: RoundRules,
    round_name: str,
    fallback_result: Optional[Path],
) -> SessionResult:
    team_id: Optional[str] = None
    started_at: Optional[float] = None
    rotate_times = []

    try:
        while True:
            deadline = (
                None if started_at is None else started_at + rules.max_time
            )
            data_type, payload = recv_packet(client, deadline)

            if data_type not in (0, 1, 2, 3):
                raise ProtocolError(f"unsupported DataType: {data_type}")

            if data_type == 0:
                candidate_team_id = decode_utf8(payload, "team ID")
                if TEAM_ID_RE.fullmatch(candidate_team_id) is None:
                    raise ProtocolError(
                        "team ID must contain only ASCII letters and digits"
                    )
                if started_at is None:
                    team_id = candidate_team_id
                    started_at = time.monotonic()
                    print(f"[START] team_id={team_id}", flush=True)
                else:
                    print("[WARN] repeated DataType=0 ignored", flush=True)
                continue

            if started_at is None:
                raise ProtocolError(f"DataType={data_type} received before DataType=0")

            elapsed = time.monotonic() - started_at
            if data_type == 1:
                predictions = parse_result_text(decode_utf8(payload, "result"))
                return SessionResult(
                    team_id=team_id,
                    predictions=predictions,
                    total_time=elapsed,
                    rotate_times=tuple(rotate_times),
                    source="network",
                    error=None,
                    exit_code=0,
                )

            if data_type == 2:
                return SessionResult(
                    team_id=team_id,
                    predictions=None,
                    total_time=elapsed,
                    rotate_times=tuple(rotate_times),
                    source="industrial_measurement_packet",
                    error="DataType=2 submitted in a 3D recognition round",
                    exit_code=1,
                )

            if round_name == "round1":
                raise ProtocolError("DataType=3 is not valid in round1")
            if len(rotate_times) >= 2:
                raise ProtocolError("round2 accepts at most two DataType=3 packets")
            rotate_times.append(elapsed)
            print(
                f"[ROTATE] index={len(rotate_times)} elapsed={elapsed:.3f}s",
                flush=True,
            )
    except RoundTimeout as exc:
        if fallback_result is None:
            return SessionResult(
                team_id=team_id,
                predictions=None,
                total_time=rules.max_time,
                rotate_times=tuple(rotate_times),
                source="timeout",
                error=f"{exc}; no --fallback-result was provided",
                exit_code=1,
            )
        try:
            predictions = read_result_file(fallback_result)
        except ProtocolError as fallback_exc:
            return SessionResult(
                team_id=team_id,
                predictions=None,
                total_time=rules.max_time,
                rotate_times=tuple(rotate_times),
                source="invalid_fallback",
                error=str(fallback_exc),
                exit_code=1,
            )
        return SessionResult(
            team_id=team_id,
            predictions=predictions,
            total_time=rules.max_time,
            rotate_times=tuple(rotate_times),
            source="fallback_result",
            error="MaxTime reached; fallback result used with 10% deduction",
            exit_code=0,
        )
    except ProtocolError as exc:
        elapsed = None if started_at is None else time.monotonic() - started_at
        return SessionResult(
            team_id=team_id,
            predictions=None,
            total_time=elapsed,
            rotate_times=tuple(rotate_times),
            source="protocol_error",
            error=str(exc),
            exit_code=1,
        )


def print_report(
    round_name: str,
    case_paths: Sequence[Path],
    session: SessionResult,
    report: Optional[ScoreReport],
) -> None:
    print()
    print("=== 3DCV Referee Report ===")
    print(f"Round: {round_name}")
    print(f"Team ID: {session.team_id or '<not received>'}")
    for table, case_path in enumerate(case_paths, start=1):
        print(f"Table {table} case: {case_path}")
    print(f"Result source: {session.source}")
    if session.total_time is not None:
        print(f"Total time: {session.total_time:.3f}s")
    else:
        print("Total time: N/A")
    if round_name == "round2":
        rotations = ", ".join(f"{value:.3f}s" for value in session.rotate_times)
        print(f"Rotate signals: {len(session.rotate_times)}/2")
        print(f"Rotate times: {rotations or 'none'}")
        if len(session.rotate_times) < 2:
            print("Warning: fewer than two rotate signals were received")
    if session.error:
        print(f"Notice: {session.error}")

    if report is None:
        print("Recognition total: 0.000000")
        print("Recognition average: 0.000000")
        print("Time score: 0.000000")
        print("Round total: 0.000000")
        return

    print()
    print("Table  Goal_ID  Truth  Pred  Score      Reason")
    print("-----  -------  -----  ----  ---------  ----------------")
    for item in report.item_scores:
        print(
            f"{item.table:>5}  {item.goal_id:<7}  {item.truth:>5}  "
            f"{item.prediction:>4}  {item.score:>9.6f}  {item.reason}"
        )

    print()
    print(f"Recognition total: {report.measure_score:.6f}/{report.full_score:.6f}")
    print(f"Recognition average: {report.average_score:.6f}")
    print(f"Time weight: {report.time_weight:.6f}")
    print(f"Time score: {report.time_score:.6f}")
    print(f"Raw round total: {report.raw_total_score:.6f}")
    if report.deduction_rate:
        print(f"Manual fallback deduction: {report.deduction_rate * 100:.0f}%")
    print(f"Round total: {report.total_score:.6f}")


def serve(args: argparse.Namespace) -> int:
    rules = ROUND_RULES[args.round]
    truth = load_truth(args.case_paths)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            f"[LISTEN] {args.host}:{args.port} round={args.round} "
            f"truth_categories={len(truth)}",
            flush=True,
        )
        client, address = server.accept()
        with client:
            print(f"[CONNECTED] client={address[0]}:{address[1]}", flush=True)
            session = run_session(
                client=client,
                rules=rules,
                round_name=args.round,
                fallback_result=args.fallback_result,
            )

    report = None
    if session.predictions is not None and session.total_time is not None:
        deduction_rate = 0.1 if session.source == "fallback_result" else 0.0
        report = score_predictions(
            truth=truth,
            predictions=session.predictions,
            rules=rules,
            total_time=session.total_time,
            deduction_rate=deduction_rate,
        )
    print_report(args.round, args.case_paths, session, report)
    return session.exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        return serve(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"Referee box failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
