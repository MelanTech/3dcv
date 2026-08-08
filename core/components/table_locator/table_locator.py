def process(self, detections: List[Detection], table: int) -> List[Detection]:
    # ---- 强制打印，不依赖 logger ----
    print(f"[DIAG] table={table}, detections_len={len(detections)}", flush=True)
def process(self, detections: List[Detection], table: int) -> List[Detection]:
    print(f"[DIAG] table={table}, detections_len={len(detections)}", flush=True)
    # ... 其余代码不变 ...
def process(self, detections: List[Detection], table: int) -> List[Detection]:
    """更新候选跟踪；检测到换桌时主动重置状态，并基于检测连续性锁定。"""
    import logging
    logger = logging.getLogger(__name__)
    
    # ---- 1. 换桌检测：完全重置 ----
    if table != self._last_table_id:
        self._reset_state()
        self._last_table_id = table
        self.force_default(f"table_changed_to_{table}")
        self._frames_since_switch = 0
        self._last_seen_bbox = None
        self._last_seen_frame = None
        self._frame_count_since_switch = 0
        self._total_detections_since_switch = 0
        self._valid_bbox_count_since_switch = 0

    table_detections = [
        {
            "bbox": self._to_bbox(detection.bbox),
            "score": float(detection.score),
        }
        for detection in detections
        if detection.class_name == "Table"
    ]

    if not hasattr(self, "_frames_since_switch"):
        self._frames_since_switch = 0
    if not hasattr(self, "_last_seen_bbox"):
        self._last_seen_bbox = None
        self._last_seen_frame = None
    if not hasattr(self, "_frame_count_since_switch"):
        self._frame_count_since_switch = 0
        self._total_detections_since_switch = 0
        self._valid_bbox_count_since_switch = 0

    if self._last_table_id == table:
        self._frames_since_switch += 1
    else:
        self._frames_since_switch = 1

    self.current_frame += 1
    self._update_candidates(table_detections)

    # ---- 诊断日志：每帧打印 table_detections 数量 ----
    raw_count = len(table_detections)
    valid_count = sum(1 for d in table_detections if self._is_valid_bbox(d["bbox"]))
    
    if table in [2, 3]:
        logger.info(
            f"[TABLE_LOCATOR_DIAG] table={table}, "
            f"frame_since_switch={self._frames_since_switch}, "
            f"raw_detections={raw_count}, "
            f"valid_detections={valid_count}, "
            f"candidates_count={len(self.candidates)}, "
            f"is_localized={self._is_localized}"
        )
    
    self._frame_count_since_switch += 1
    self._total_detections_since_switch += raw_count
    self._valid_bbox_count_since_switch += valid_count

    if table_detections:
        best_det = max(table_detections, key=lambda d: d["score"])
        if self._is_valid_bbox(best_det["bbox"]):
            self._last_seen_bbox = best_det["bbox"]
            self._last_seen_frame = self.current_frame

    # ---- 2. 基于检测连续性的主动锁定 ----
    if not self._is_localized:
        for candidate in self.candidates.values():
            history = candidate["history"]
            if len(history) >= 2:
                self.target_bbox = history[-1]["bbox"]
                self.using_default_bbox = False
                self._is_localized = True
                self._is_stable = True
                self.fallback_reason = None
                self.last_update_frame = self.current_frame
                logger.info(
                    f"[TABLE_LOCATOR_DIAG] LOCKED table={table}, "
                    f"total_frames={self._frame_count_since_switch}, "
                    f"total_detections={self._total_detections_since_switch}, "
                    f"valid_detections={self._valid_bbox_count_since_switch}, "
                    f"reason='candidate_history_>=2'"
                )
                return self._replace_table_detection(detections)

        if self._frames_since_switch >= 5 and self._last_seen_bbox is not None:
            self.target_bbox = self._last_seen_bbox
            self.using_default_bbox = False
            self._is_localized = True
            self._is_stable = True
            self.fallback_reason = "forced_last_seen_detection"
            self.last_update_frame = self.current_frame
            logger.info(
                f"[TABLE_LOCATOR_DIAG] LOCKED table={table}, "
                f"total_frames={self._frame_count_since_switch}, "
                f"total_detections={self._total_detections_since_switch}, "
                f"valid_detections={self._valid_bbox_count_since_switch}, "
                f"reason='forced_last_seen_detection'"
            )
            return self._replace_table_detection(detections)

    # ---- 3. 如果使用默认框但检测到候选 ----
    if self.using_default_bbox and table_detections:
        for candidate in self.candidates.values():
            history = candidate["history"]
            if len(history) >= 2:
                last_bbox = history[-1]["bbox"]
                if last_bbox and self._is_valid_bbox(last_bbox):
                    self.target_bbox = last_bbox
                    self.using_default_bbox = False
                    self._is_localized = True
                    self._is_stable = True
                    self.fallback_reason = None
                    self.last_update_frame = self.current_frame
                    return self._replace_table_detection(detections)

        first_det = table_detections[0]
        if self._is_valid_bbox(first_det["bbox"]):
            self.target_bbox = first_det["bbox"]
            self.using_default_bbox = False
            self._is_localized = True
            self._is_stable = True
            self.fallback_reason = None
            self.last_update_frame = self.current_frame
            return self._replace_table_detection(detections)

    # ---- 4. 已定位时正常更新 ----
    if self._is_localized and self.target_bbox is not None:
        if self.using_default_bbox:
            selected = self._select_best_candidate()
            if selected is not None and self._candidate_is_stable(selected[1]):
                candidate_id, best_candidate = selected
                self.target_bbox = best_candidate["history"][-1]["bbox"]
                self.target_candidate_id = candidate_id
                self.using_default_bbox = False
                self.last_update_frame = self.current_frame
                self._is_stable = self._candidate_is_stable(best_candidate)
        elif self.current_frame - self.last_update_frame >= self.update_interval:
            self._try_update_bbox(table_detections)
            self.last_update_frame = self.current_frame

        if self.target_candidate_id in self.candidates:
            selected = self._select_best_candidate()
            target_candidate = self.candidates[self.target_candidate_id]
            self._is_stable = (
                selected is not None
                and selected[0] == self.target_candidate_id
                and self._candidate_is_stable(target_candidate)
            )

        return self._replace_table_detection(detections)

    # ---- 5. 未定位：初始化阶段 ----
    selected = None
    if self.current_frame >= self.init_frames:
        selected = self._select_best_candidate()
        if selected is not None and self._candidate_is_stable(selected[1]):
            candidate_id, best_candidate = selected
            self.target_bbox = best_candidate["history"][-1]["bbox"]
            self.target_candidate_id = candidate_id
            self.using_default_bbox = False
            self._is_stable = True
            self._is_localized = True
            self.last_update_frame = self.current_frame
            return self._replace_table_detection(detections)

    # ---- 6. 强制锁定 ----
    if self.current_frame >= self.init_frames * 2:
        if selected is None:
            selected = self._select_best_candidate()
        if selected is not None:
            candidate_id, best_candidate = selected
            if best_candidate["history"]:
                self.target_bbox = best_candidate["history"][-1]["bbox"]
                self.target_candidate_id = candidate_id
                self.using_default_bbox = False
                self._is_stable = True
                self._is_localized = True
                self.fallback_reason = "forced_after_timeout"
                self.last_update_frame = self.current_frame
                return self._replace_table_detection(detections)

    # ---- 7. 最后兜底 ----
    if self._last_seen_bbox is not None:
        self.target_bbox = self._last_seen_bbox
        self.using_default_bbox = False
        self._is_localized = True
        self._is_stable = True
        self.fallback_reason = "forced_single_sighting"
        self.last_update_frame = self.current_frame
        logger.info(
            f"[TABLE_LOCATOR_DIAG] LOCKED table={table}, "
            f"total_frames={self._frame_count_since_switch}, "
            f"total_detections={self._total_detections_since_switch}, "
            f"valid_detections={self._valid_bbox_count_since_switch}, "
            f"reason='forced_single_sighting'"
        )
        return self._replace_table_detection(detections)

    # ---- 8. 锁定失败警告 ----
    if table in [2, 3] and self._frame_count_since_switch >= 5:
        logger.warning(
            f"[TABLE_LOCATOR_DIAG] WARNING: table={table} failed to acquire, "
            f"total_frames={self._frame_count_since_switch}, "
            f"total_detections={self._total_detections_since_switch}, "
            f"valid_detections={self._valid_bbox_count_since_switch}, "
            f"reason='no_valid_detection_seen'"
        )

    self._is_localized = False
    self._is_stable = False
    self.fallback_reason = "waiting_for_stable_candidate"

    return detections