"""Independent-process detector stage for overlapping ACL detector and OCR work."""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from typing import Any, Dict, List, Optional

from core.components.detector.builder import build_detector
from core.infra.concurrency.shared_frame_slots import (
    SharedFrameReference,
    SharedFrameSlots,
    frame_from_shared_reference,
)
from core.orchestration.pipeline.detector_stage import (
    BaseDetectorStage,
    DetectedFrame,
)
from core.types import Detection, Frame


def _close_timeout(env_name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(env_name, default)))
    except (TypeError, ValueError):
        return default


def _detector_worker(
    requests: mp.Queue,
    responses: mp.Queue,
    detector_config: Dict[str, Any],
    class_registry: Optional[Dict[str, Any]],
    round_name: str,
) -> None:
    """Own one ACL runtime/context/model and serve serialized inference calls."""
    detector = None
    shared_blocks = {}
    try:
        detector = build_detector(detector_config, round_name, class_registry)
        responses.put(("ready", None, None))
        while True:
            message = requests.get()
            if message[0] == "close":
                return

            message_type, request_id, payload, table = message
            try:
                if message_type == "infer_shared":
                    frame = frame_from_shared_reference(payload, shared_blocks)
                elif message_type == "infer":
                    frame = payload
                else:
                    raise ValueError(
                        f"unsupported detector worker message: {message_type!r}"
                    )
                responses.put(("result", request_id, detector.infer(frame, table)))
            except Exception:
                responses.put(("error", request_id, traceback.format_exc()))
    except Exception:
        responses.put(("startup_error", None, traceback.format_exc()))
    finally:
        close = getattr(detector, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
        for block in shared_blocks.values():
            block.close()


class ProcessDetectorStage(BaseDetectorStage):
    """One-in-flight detector process with bounded shared-memory frame transport."""

    def __init__(
        self,
        detector_config: Dict[str, Any],
        class_registry: Optional[Dict[str, Any]],
        round_name: str,
    ):
        context = mp.get_context("spawn")
        self._requests: mp.Queue = context.Queue(maxsize=1)
        self._responses: mp.Queue = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_detector_worker,
            args=(self._requests, self._responses, detector_config, class_registry, round_name),
            name="detector-worker",
        )
        self._next_request_id = 0
        self._frame_slots: Optional[SharedFrameSlots] = None
        self._pending: Optional[tuple[Frame, int, int]] = None
        self._closed = False
        self._process.start()
        kind, _, payload = self._receive(timeout_sec=60.0)
        if kind != "ready":
            self.close()
            raise RuntimeError(
                "detector worker failed to start:\n"
                f"{payload or 'worker exited before initialization'}"
            )

    def accept(self, frame: Frame, table: int) -> Optional[DetectedFrame]:
        self._ensure_running()
        request_id = self._submit(frame, table)
        previous = self._pending
        self._pending = (frame, table, request_id)
        if previous is None:
            return None

        previous_frame, previous_table, previous_request_id = previous
        return DetectedFrame(
            frame=previous_frame,
            table=previous_table,
            detections=self._result(previous_request_id),
        )

    def flush(self) -> Optional[DetectedFrame]:
        pending = self._pending
        if pending is None:
            return None
        self._pending = None
        frame, table, request_id = pending
        return DetectedFrame(
            frame=frame,
            table=table,
            detections=self._result(request_id),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                self._requests.put(
                    ("close",),
                    timeout=_close_timeout("CV3D_DETECTOR_WORKER_CLOSE_PUT_TIMEOUT_SEC", 0.2),
                )
            except queue.Full:
                pass
            self._process.join(
                timeout=_close_timeout("CV3D_DETECTOR_WORKER_CLOSE_JOIN_TIMEOUT_SEC", 0.5)
            )
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(
                    timeout=_close_timeout("CV3D_DETECTOR_WORKER_TERMINATE_JOIN_TIMEOUT_SEC", 0.5)
                )
        self._requests.close()
        self._responses.close()
        if self._frame_slots is not None:
            self._frame_slots.close()
            self._frame_slots = None

    def _submit(self, frame: Frame, table: int) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        reference = self._write_shared_frame(frame, request_id)
        message = (
            ("infer", request_id, frame, table)
            if reference is None
            else ("infer_shared", request_id, reference, table)
        )
        try:
            self._requests.put(message, timeout=10.0)
        except queue.Full as exc:
            self._ensure_running()
            raise RuntimeError("detector worker request queue remained full") from exc
        return request_id

    def _write_shared_frame(
        self,
        frame: Frame,
        request_id: int,
    ) -> Optional[SharedFrameReference]:
        try:
            if self._frame_slots is None:
                self._frame_slots = SharedFrameSlots(frame)
            return self._frame_slots.write(frame, slot=request_id % 2)
        except (TypeError, ValueError):
            return None

    def _result(self, request_id: int) -> List[Detection]:
        kind, result_id, payload = self._receive(timeout_sec=None)
        if kind == "result" and result_id == request_id:
            return payload
        if kind == "error":
            raise RuntimeError(
                f"detector worker inference failed for request {result_id}:\n{payload}"
            )
        raise RuntimeError(
            f"detector worker returned unexpected response {kind!r} "
            f"for request {result_id}"
        )

    def _ensure_running(self) -> None:
        if self._closed:
            raise RuntimeError("detector process stage is closed")
        if not self._process.is_alive():
            raise RuntimeError(
                f"detector worker exited unexpectedly with exit code {self._process.exitcode}"
            )

    def _receive(self, timeout_sec: Optional[float]):
        if timeout_sec is not None:
            try:
                return self._responses.get(timeout=timeout_sec)
            except queue.Empty as exc:
                self._ensure_running()
                raise RuntimeError("detector worker did not respond in time") from exc

        while True:
            try:
                return self._responses.get(timeout=1.0)
            except queue.Empty:
                self._ensure_running()
