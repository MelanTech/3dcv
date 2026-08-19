"""运行时装配：根据配置构建一整套组件并交给对应轮次的状态机执行。"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, TypeVar

from core.components.frame_source.builder import build_frame_source
from core.config_loader import load_config
from core.components.counter.builder import build_counter
from core.components.detector.builder import build_detector
from core.components.filter.builder import build_filter
from core.components.referee.builder import build_referee_client
from core.infra.logging.event_logger import EventLogger
from core.infra import pause_clock
from core.components.ocr.builder import build_ocr
from core.components.unknown_merger.builder import build_unknown_merger
from core.orchestration.pipeline.builder import build_detector_stage
from core.orchestration.pipeline.frame_pipeline import FramePipeline
from core.orchestration.state_machine.round1_state_machine import Round1StateMachine
from core.orchestration.state_machine.round2_state_machine import Round2StateMachine
from core.orchestration.state_machine.state_logger import StateLogger
from core.components.table_locator.builder import build_table_locator
from core.infra.visualization.count_gui_builder import build_count_gui
from core.infra.visualization.builder import build_visualizer


RoundName = str
T = TypeVar("T")


def _build_component(logger: EventLogger, name: str, factory: Callable[[], T]) -> T:
    """构建单个可插拔组件，并记录其初始化耗时。"""
    started = time.perf_counter()
    logger.event("component_init_started", component=name)
    value = factory()
    logger.event(
        "component_init_finished",
        component=name,
        elapsed_sec=round(time.perf_counter() - started, 3),
    )
    return value


def _round_number(round_name: RoundName) -> int:
    if round_name == "round1":
        return 1
    if round_name == "round2":
        return 2
    raise ValueError(f"unsupported round name: {round_name}")


def _send_start_signal(logger: EventLogger, referee_client) -> None:
    """Connect to the referee and send START before expensive model initialization."""
    with StateLogger(logger, "CONNECT_REFEREE"):
        if not referee_client.connect():
            raise RuntimeError("failed to connect referee box")

    with StateLogger(logger, "SEND_START"):
        if not referee_client.send_start():
            raise RuntimeError("failed to send start signal")


def _finalize_before_state_machine(
    logger: EventLogger,
    referee_client,
    round_number: int,
    exc: BaseException,
) -> None:
    """If startup already began the round, submit an empty fallback result."""
    logger.event(
        "round_interrupted_before_state_machine",
        round_number=round_number,
        error=str(exc),
        exc_type=type(exc).__name__,
    )
    try:
        result_path = referee_client.write_result(round_number, [])
        logger.event(
            "startup_fallback_result_saved",
            round_number=round_number,
            path=str(result_path),
        )
        if bool(getattr(referee_client, "enabled", True)):
            sent = referee_client.send_result_file(result_path)
            logger.event(
                "startup_fallback_result_sent",
                round_number=round_number,
                path=str(result_path),
                sent=bool(sent),
            )
    except Exception as finalize_exc:
        logger.event(
            "startup_fallback_result_failed",
            round_number=round_number,
            error=str(finalize_exc),
            exc_type=type(finalize_exc).__name__,
        )


def _close_component(logger: EventLogger, name: str, component) -> None:
    close = getattr(component, "close", None)
    if close is None:
        return
    try:
        close()
        logger.event("startup_resource_closed", resource=name)
    except Exception as exc:
        logger.event(
            "startup_resource_close_failed",
            resource=name,
            error=str(exc),
            exc_type=type(exc).__name__,
        )


def _filter_intrinsic_config(filter_config):
    if isinstance(filter_config, dict):
        if set(filter_config) == {"filter"}:
            return _filter_intrinsic_config(filter_config["filter"])
        return filter_config.get("intrinsic")
    return None


def run_round(config_path: str, round_name: RoundName) -> Path:
    """加载配置、按依赖顺序装配所有组件，然后把控制权交给轮次状态机。"""
    round_started_at = pause_clock.now()
    config = load_config(config_path)
    logging_config = config["logging"]
    round_number = _round_number(round_name)

    with EventLogger(
        logging_config["base_dir"],
        round_name,
        console=logging_config.get("console", True),
    ) as logger:
        logger.event("config_loaded", path=config_path)
        referee_client = _build_component(
            logger,
            "referee",
            lambda: build_referee_client(
                config["referee"],
                config["team"],
                logger,
                config.get("class_registry"),
            ),
        )
        start_signal_sent = False
        detector = None
        ocr = None
        unknown_merger = None
        frame_source = None
        table_filter = None
        visualizer = None
        count_gui = None
        detector_stage = None
        pipeline = None

        try:
            _send_start_signal(logger, referee_client)
            start_signal_sent = True

            # 先初始化所有 ACL 模型，再打开 OpenNI 相机，避免 native 库初始化顺序冲突。
            detector = _build_component(
                logger,
                "detector",
                lambda: build_detector(config["detector"], round_name, config.get("class_registry")),
            )
            ocr = _build_component(
                logger,
                "ocr",
                lambda: build_ocr(config["ocr"], config.get("class_registry")),
            )
            unknown_merger = _build_component(
                logger,
                "unknown_merger",
                lambda: build_unknown_merger(
                    config.get("unknown_merger"),
                    config.get("class_registry"),
                ),
            )
            frame_source = _build_component(
                logger,
                "frame_source",
                lambda: build_frame_source(config["frame_source"], round_name),
            )
            table_locator = _build_component(
                logger,
                "table_locator",
                lambda: build_table_locator(config["table_locator"]),
            )
            table_filter = _build_component(
                logger,
                "filter",
                lambda: build_filter(
                    config["filter"],
                    intrinsic_override=frame_source.get_filter_intrinsic(),
                ),
            )
            filter_intrinsic = frame_source.get_filter_intrinsic()
            logger.event(
                "filter_intrinsic_selected",
                source=frame_source.get_filter_intrinsic_source(),
                intrinsic=filter_intrinsic or _filter_intrinsic_config(config["filter"]),
            )
            filter_intrinsic_error = getattr(frame_source, "filter_intrinsic_error", None)
            if filter_intrinsic_error:
                logger.event(
                    "filter_intrinsic_runtime_read_failed",
                    error=filter_intrinsic_error,
                    fallback="config",
                )
            counter = _build_component(
                logger,
                "counter",
                lambda: build_counter(config["counter"], config.get("class_registry")),
            )
            visualizer = _build_component(
                logger,
                "visualizer",
                lambda: build_visualizer(config.get("visualization"), config.get("class_registry")),
            )
            count_gui = _build_component(
                logger,
                "count_gui",
                lambda: build_count_gui(
                    config.get("visualization", {}).get("count_gui"),
                    config.get("class_registry"),
                    logger,
                ),
            )
            detector_stage = _build_component(
                logger,
                "detector_stage",
                lambda: build_detector_stage(
                    detector=detector,
                    config=config.get("pipeline", {}).get("async_detector"),
                    detector_config=config["detector"],
                    class_registry=config.get("class_registry"),
                    round_name=round_name,
                    logger=logger,
                ),
            )
            pipeline = _build_component(
                logger,
                "pipeline",
                lambda: FramePipeline(
                    detector=detector,
                    table_locator=table_locator,
                    table_filter=table_filter,
                    ocr=ocr,
                    counter=counter,
                    visualizer=visualizer,
                    count_gui=count_gui,
                    logger=logger,
                    unknown_merger=unknown_merger,
                    log_per_frame=logging_config.get("per_frame", False),
                    ignored_by_counter=config.get("class_registry", {}).get("ignored_by_counter", []),
                    round_started_at=round_started_at,
                    detector_stage=detector_stage,
                ),
            )

            if round_name == "round1":
                machine = Round1StateMachine(
                    config=config,
                    frame_source=frame_source,
                    pipeline=pipeline,
                    referee_client=referee_client,
                    logger=logger,
                    round_started_at=round_started_at,
                    start_signal_sent=start_signal_sent,
                )
            elif round_name == "round2":
                machine = Round2StateMachine(
                    config=config,
                    frame_source=frame_source,
                    pipeline=pipeline,
                    referee_client=referee_client,
                    logger=logger,
                    round_started_at=round_started_at,
                    start_signal_sent=start_signal_sent,
                )
            else:
                raise ValueError(f"unsupported round name: {round_name}")

            # 状态机负责轮次特有的流程编排与结果提交。
            return machine.run()
        except BaseException as exc:
            if pipeline is None:
                if start_signal_sent:
                    _finalize_before_state_machine(
                        logger,
                        referee_client,
                        round_number,
                        exc,
                    )
                for name, component in (
                    ("detector_stage", detector_stage),
                    ("count_gui", count_gui),
                    ("visualizer", visualizer),
                    ("filter", table_filter),
                    ("frame_source", frame_source),
                    ("unknown_merger", unknown_merger),
                    ("ocr", ocr),
                    ("detector", detector),
                    ("referee_client", referee_client),
                ):
                    if component is not None:
                        _close_component(logger, name, component)
            raise
