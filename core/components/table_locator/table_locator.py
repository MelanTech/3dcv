"""桌面定位器实现：多候选跟踪 + 稳定性打分，稳定后锁定桌面框。

对每个检测到的桌子维护一个候选轨迹，综合出现频率、中心抖动、尺寸方差和
置信度打分，选出最稳定的候选作为桌面；锁定后周期性微调，锁定超时可回退
到配置的默认框。process() 会把最终桌面框以标准 Detection 形式返回。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from core.components.table_locator.base import BaseTableLocator
from core.types import Detection


class TableLocator(BaseTableLocator):
    """基于候选稳定性打分的桌面定位器。"""

    def __init__(self, config: Dict):
        self.init_frames = int(config["init_frames"])
        self.iou_threshold = float(config["iou_threshold"])
        self.min_area = int(config["min_area"])
        self.update_interval = int(config["update_interval"])
        self.update_iou_threshold = float(config["update_iou_threshold"])
        self.default_bbox = self._to_bbox(config["default_bbox"])
        self.min_stable_frames = int(config.get("min_stable_frames", 8))
        self.max_center_shift_px = float(config.get("max_center_shift_px", 8.0))
        self.max_size_change_ratio = float(config.get("max_size_change_ratio", 0.05))
        self.max_area_trend_ratio = float(config.get("max_area_trend_ratio", 0.08))
        self.max_missing_frames = int(config.get("max_missing_frames", 5))

        if not self._is_valid_bbox(self.default_bbox):
            raise ValueError("invalid default_bbox")

        self._reset_state()

    @property
    def is_localized(self) -> bool:
        return self._is_localized

    @property
    def is_stable(self) -> bool:
        return self._is_stable

    @property
    def is_acquired(self) -> bool:
        return self._is_stable

    def handle_acquire_timeout(self, reason: str) -> None:
        self.force_default(reason)

    def _reset_state(self) -> None:
        self.current_frame = 0
        self._is_localized = False
        self._is_stable = False
        self.target_bbox: Optional[Tuple[int, int, int, int]] = None
        self.target_candidate_id: Optional[str] = None
        self.last_update_frame = -self.update_interval
        self.using_default_bbox = False
        self.fallback_reason: Optional[str] = None
        self.candidates: Dict[str, Dict] = {}
        self.next_id = 0

    def clear(self) -> None:
        self._reset_state()

    @staticmethod
    def _to_bbox(bbox) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2), int(y2)

    def _is_valid_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        return x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) >= self.min_area

    def _iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个框的交并比（IoU），面积过小的框视为无效返回 0。"""
        x1, y1, x2, y2 = bbox1
        a1, b1, a2, b2 = bbox2

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2 - b1)
        if area1 < self.min_area or area2 < self.min_area:
            return 0.0

        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _calculate_stability_score(self, history: List[Dict]) -> float:
        """综合出现频率、中心位置抖动、尺寸方差和置信度，给候选打稳定性分。"""
        n = len(history)
        if n == 0 or self.current_frame <= 0:
            return 0.0

        freq_score = (n / self.current_frame) * 100.0 * 0.3

        centers = []
        for item in history:
            x1, y1, x2, y2 = item["bbox"]
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        avg_x = sum(center[0] for center in centers) / n
        avg_y = sum(center[1] for center in centers) / n
        std_x = sum((center[0] - avg_x) ** 2 for center in centers) / n
        std_y = sum((center[1] - avg_y) ** 2 for center in centers) / n
        pos_score = max(0.0, 30.0 - (std_x + std_y) / 2.0)

        areas = [
            (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])
            for item in history
        ]
        avg_area = sum(areas) / n
        area_var = sum((area - avg_area) ** 2 for area in areas) / n
        size_score = max(0.0, 20.0 - area_var / 1000.0)

        conf_score = (sum(item["score"] for item in history) / n) * 100.0 * 0.2
        return freq_score + pos_score + size_score + conf_score

    def _select_best_candidate(self) -> Optional[Tuple[str, Dict]]:
        """给所有候选重新打分并选出分数最高者（需超过阈值）。"""
        if not self.candidates:
            return None

        for candidate in self.candidates.values():
            candidate["score"] = self._calculate_stability_score(candidate["history"])

        candidate_id, best_candidate = max(
            self.candidates.items(),
            key=lambda item: item[1]["score"],
        )
        return (candidate_id, best_candidate) if best_candidate["score"] > 30.0 else None

    def _candidate_is_stable(self, candidate: Dict) -> bool:
        """判断候选最近若干帧是否足够稳定（中心位移、尺寸变化、面积趋势均在阈值内）。"""
        history = candidate["history"][-self.min_stable_frames:]
        if len(history) < self.min_stable_frames:
            return False

        centers = []
        widths = []
        heights = []
        areas = []
        for item in history:
            x1, y1, x2, y2 = item["bbox"]
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            width = float(x2 - x1)
            height = float(y2 - y1)
            widths.append(width)
            heights.append(height)
            areas.append(width * height)

        for previous, current in zip(centers, centers[1:]):
            dx = current[0] - previous[0]
            dy = current[1] - previous[1]
            if (dx * dx + dy * dy) ** 0.5 > self.max_center_shift_px:
                return False

        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)
        if avg_width <= 0.0 or avg_height <= 0.0:
            return False

        max_width_delta = max(abs(width - avg_width) / avg_width for width in widths)
        max_height_delta = max(abs(height - avg_height) / avg_height for height in heights)
        avg_area = sum(areas) / len(areas)
        if avg_area <= 0.0:
            return False

        area_trend_ratio = abs(areas[-1] - areas[0]) / avg_area
        return (
            max_width_delta <= self.max_size_change_ratio
            and max_height_delta <= self.max_size_change_ratio
            and area_trend_ratio <= self.max_area_trend_ratio
        )

    def force_default(self, reason: str) -> None:
        """回退到配置的默认桌面框，并直接标记为已定位/已稳定。"""
        self.target_bbox = self.default_bbox
        self.target_candidate_id = None
        self.using_default_bbox = True
        self.fallback_reason = reason
        self._is_localized = True
        self._is_stable = True

    def _try_update_bbox(self, table_detections: List[Dict]) -> bool:
        """用与当前目标框 IoU 最高且达标的新检测更新目标框。"""
        if not table_detections or self.target_bbox is None:
            return False

        best_iou = 0.0
        best_bbox = None
        for detection in table_detections:
            iou = self._iou(self.target_bbox, detection["bbox"])
            if iou > best_iou and iou > self.update_iou_threshold:
                best_iou = iou
                best_bbox = detection["bbox"]

        if best_bbox is not None:
            self.target_bbox = best_bbox
            return True
        return False

    def _update_candidates(self, table_detections: List[Dict]) -> None:
        """把本帧桌子检测按 IoU 关联到已有候选，未匹配的新建候选，并清理消失候选。"""
        if not table_detections:
            self._cleanup_missing_candidates()
            return

        matched_ids: Set[str] = set()
        for detection in table_detections:
            best_iou = 0.0
            best_id = None

            for candidate_id, candidate in self.candidates.items():
                last_bbox = candidate["history"][-1]["bbox"]
                iou = self._iou(detection["bbox"], last_bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_id = candidate_id

            if best_id is not None:
                self.candidates[best_id]["history"].append(detection)
                self.candidates[best_id]["last_seen_frame"] = self.current_frame
                matched_ids.add(best_id)
            else:
                self.candidates[f"Table_{self.next_id}"] = {
                    "history": [detection],
                    "score": 0.0,
                    "first_seen_frame": self.current_frame,
                    "last_seen_frame": self.current_frame,
                }
                self.next_id += 1
        self._cleanup_missing_candidates()

    def _cleanup_missing_candidates(self) -> None:
        """移除超过最大丢失帧数的候选；若目标候选丢失则标记为不稳定。"""
        expired_ids = [
            candidate_id
            for candidate_id, candidate in self.candidates.items()
            if self.current_frame - candidate.get("last_seen_frame", 0) > self.max_missing_frames
        ]
        for candidate_id in expired_ids:
            self.candidates.pop(candidate_id, None)
            if self.target_candidate_id == candidate_id:
                self.target_candidate_id = None
                self._is_stable = False

    def _replace_table_detection(self, detections: List[Detection]) -> List[Detection]:
        """用当前锁定的桌面框替换原始 Table 检测。"""
        if self.target_bbox is None:
            return detections
        non_table = [detection for detection in detections if detection.class_name != "Table"]
        return [
            *non_table,
            Detection(
                class_name="Table",
                class_id=0,
                bbox=self.target_bbox,
                score=1.0,
                evidence={"source": "table_locator"},
            ),
        ]

    def process(self, detections: List[Detection], _table: int) -> List[Detection]:
        """更新候选跟踪；已定位时用锁定的桌面框替换/补充 "Table" 检测后返回。"""
        table_detections = [
            {
                "bbox": self._to_bbox(detection.bbox),
                "score": float(detection.score),
            }
            for detection in detections
            if detection.class_name == "Table"
        ]

        self.current_frame += 1
        self._update_candidates(table_detections)

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

        if self.current_frame >= self.init_frames:
            selected = self._select_best_candidate()
            if selected is not None and self._candidate_is_stable(selected[1]):
                candidate_id, best_candidate = selected
                self.target_bbox = best_candidate["history"][-1]["bbox"]
                self.target_candidate_id = candidate_id
                self.using_default_bbox = False
                self._is_stable = self._candidate_is_stable(best_candidate)
                self._is_localized = True
                self.last_update_frame = self.current_frame
                return self._replace_table_detection(detections)

            self._is_localized = False
            self._is_stable = False
            self.fallback_reason = "waiting_for_stable_candidate"

        return detections
