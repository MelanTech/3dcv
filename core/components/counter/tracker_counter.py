"""追踪计数器：利用 2D 边框 IoU 追踪实例，利用置信度软投票进行稳健分类。

核心思路：把每一帧视为对桌面物理实体的一次“观测”。
通过 2D 坐标的重合度（IoU）跨帧关联目标，建立独立的生命周期（Track）。
对同一 Track 生命周期内的类别置信度进行累加（软投票），彻底消除闪烁，并应对临时遮挡。
"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from core.components.counter.base import BaseCounter
from core.types import Detection


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """计算两个 2D 边界框的交并比 (IoU)。"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    if inter_area == 0:
        return 0.0

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    return float(inter_area) / float(union_area) if union_area > 0 else 0.0


def compute_iom(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """计算两个框的交集面积占较小框面积的比例 (Intersection over Min Area / 包含度)。"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    if inter_area == 0:
        return 0.0

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    min_area = min(area1, area2)
    return float(inter_area) / float(min_area) if min_area > 0 else 0.0


class TrackState(Enum):
    """实体生命周期状态"""
    TENTATIVE = auto()  # 观察期：新出现的检测，尚未确认是否为稳定物理实体
    ACTIVE = auto()     # 活跃态：已确认的物理实体，当前正在视野中稳定跟踪
    LOST = auto()       # 休眠态：已确认的物理实体，暂时离开视野或被遮挡，保留档案待唤醒


class Track:
    """具备完整状态机与置信度积分累积的物理实体 Track。"""

    _id_counter = 0

    def __init__(self, detection: Detection):
        self.id = Track._id_counter
        Track._id_counter += 1

        self.bbox = detection.bbox
        self.class_votes: Dict[str, float] = defaultdict(float)
        self.hits = 1
        self.misses = 0
        self.state = TrackState.TENTATIVE
        self.total_score = float(detection.score)

        # 初始化第一次投票
        self.class_votes[detection.class_name] += float(detection.score)

    def update(self, detection: Detection) -> None:
        """用新的观测更新状态。"""
        self.bbox = detection.bbox
        score = float(detection.score)
        self.class_votes[detection.class_name] += score
        self.total_score += score
        self.hits += 1
        self.misses = 0
        if self.state == TrackState.LOST:
            self.state = TrackState.ACTIVE

    def mark_miss(self) -> None:
        """记录漏检。"""
        self.misses += 1

    def get_best_class_and_score(self) -> Tuple[str, float]:
        """获取目前累积票数最高的类别及其总积分。"""
        if not self.class_votes:
            return "Unknown", 0.0
        best_class = max(self.class_votes.items(), key=lambda item: item[1])
        return best_class[0], best_class[1]


class TrackerCounter(BaseCounter):
    """工业级休眠-唤醒追踪计数器。"""

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        if class_registry is None:
            raise ValueError("TrackerCounter requires shared class_registry config")

        self.class_names = list(class_registry.get("result_classes", class_registry.get("classes", [])))
        if not self.class_names:
            raise ValueError("class_registry.result_classes must not be empty")
        self.class_set = set(self.class_names)

        # 未知/OCR 类别（例如书本 W001~W004），不参与常规几何追踪，按出现最大值处理
        self.unknown_classes = set(
            class_registry.get(
                "unknown_classes",
                class_registry.get("ocr_output_classes", []),
            )
        )
        self.max_unknown_detections = {name: 0 for name in self.class_names if name in self.unknown_classes}

        self.max_per_class = int(config.get("max_object_count", 5))
        self.total_max = int(config.get("total_max", 15))

        # 核心追踪与确认参数
        self.iou_threshold = float(config.get("iou_threshold", 0.3))
        self.center_dist_threshold = float(config.get("center_dist_threshold", 80.0))
        self.confirm_hits = int(config.get("confirm_hits", 3))
        self.min_score_sum = float(config.get("min_score_sum", 1.2))
        self.max_lost_frames = int(config.get("max_lost_frames", 10))

        # 所有 Track 列表（包含 Tentative, Active, Lost）
        self.tracks: List[Track] = []
        self.final_counts = {name: 0 for name in self.class_names}

    def update(self, detections: List[Detection]) -> None:
        """接收当前帧检测并驱动追踪状态机流转。"""
        # 1. 未知类别（OCR）独立统计
        unknown_detections = [d for d in detections if d.class_name in self.unknown_classes]
        current_unknown_counts = defaultdict(int)
        for d in unknown_detections:
            current_unknown_counts[d.class_name] += 1
        for name, count in current_unknown_counts.items():
            self.max_unknown_detections[name] = max(self.max_unknown_detections[name], count)

        # 2. 过滤常规有效检测并进行同类嵌套冗余框抑制 (IoM / IoU)
        raw_valid = [
            d for d in detections
            if d.class_name in self.class_set and d.class_name not in self.unknown_classes
        ]
        # 按置信度从高到低排序，优先保留置信度最高的主框，剔除被嵌套的碎片小框
        sorted_raw = sorted(raw_valid, key=lambda d: d.score, reverse=True)
        valid_detections: List[Detection] = []
        for d in sorted_raw:
            is_nested = False
            for kept in valid_detections:
                if kept.class_name == d.class_name:
                    if compute_iom(kept.bbox, d.bbox) > 0.55 or compute_iou(kept.bbox, d.bbox) > 0.5:
                        is_nested = True
                        break
            if not is_nested:
                valid_detections.append(d)

        # 3. 区分 Active / Tentative Tracks 和 Lost Tracks
        active_and_tentative = [t for t in self.tracks if t.state in (TrackState.ACTIVE, TrackState.TENTATIVE)]
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]

        unmatched_detections = list(range(len(valid_detections)))
        unmatched_active_tracks = list(range(len(active_and_tentative)))

        # 阶段一：与活跃态及观察态 Track 进行全域同类优先匹配
        matches = []
        if valid_detections and active_and_tentative:
            match_matrix = []
            for t_idx, track in enumerate(active_and_tentative):
                tx_c = (track.bbox[0] + track.bbox[2]) / 2.0
                ty_c = (track.bbox[1] + track.bbox[3]) / 2.0
                best_class, _ = track.get_best_class_and_score()

                for d_idx, det in enumerate(valid_detections):
                    dx_c = (det.bbox[0] + det.bbox[2]) / 2.0
                    dy_c = (det.bbox[1] + det.bbox[3]) / 2.0

                    iou = compute_iou(track.bbox, det.bbox)
                    dist = ((tx_c - dx_c) ** 2 + (ty_c - dy_c) ** 2) ** 0.5
                    same_class = (best_class == det.class_name)

                    if same_class:
                        # 同类别赋予极高基础分，按距离优先匹配最近的同类 Track（防止运动导致分裂）
                        score = 100.0 + iou - (dist / 1000.0)
                        match_matrix.append((score, t_idx, d_idx))
                    elif iou >= self.iou_threshold:
                        # 异类别仅在空间高度重叠时允许匹配（软投票抑制局部误检）
                        score = iou - (dist / 1000.0)
                        match_matrix.append((score, t_idx, d_idx))

            match_matrix.sort(key=lambda x: x[0], reverse=True)
            for score, t_idx, d_idx in match_matrix:
                if t_idx in unmatched_active_tracks and d_idx in unmatched_detections:
                    matches.append((t_idx, d_idx))
                    unmatched_active_tracks.remove(t_idx)
                    unmatched_detections.remove(d_idx)

        # 更新阶段一匹配成功的 Track
        for t_idx, d_idx in matches:
            active_and_tentative[t_idx].update(valid_detections[d_idx])

        # 阶段二：处理未匹配检测框（休眠唤醒 vs 新建实体）
        for d_idx in unmatched_detections:
            det = valid_detections[d_idx]

            # 优先在休眠池中寻找可唤醒的同类档案
            matching_lost = None
            for l_track in lost_tracks:
                if l_track.state == TrackState.LOST:
                    l_best_class, _ = l_track.get_best_class_and_score()
                    if l_best_class == det.class_name:
                        matching_lost = l_track
                        break

            if matching_lost is not None:
                # 唤醒并复用休眠档案，转盘转回正面
                matching_lost.update(det)
            else:
                # 只有当同类活跃实体数确实不足、且不重叠时，才新建 Tentative 档案
                is_duplicate = False
                for t in self.tracks:
                    if t.state != TrackState.LOST and compute_iou(t.bbox, det.bbox) > 0.4:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    new_track = Track(det)
                    self.tracks.append(new_track)

        # 阶段三：更新未命中 Track 状态机流转
        for t_idx in unmatched_active_tracks:
            track = active_and_tentative[t_idx]
            track.mark_miss()
            if track.state == TrackState.ACTIVE:
                if track.misses >= self.max_lost_frames:
                    track.state = TrackState.LOST
            elif track.state == TrackState.TENTATIVE:
                # 观察期目标如果连续 3 帧丢失，判定为虚假噪点，标记销毁
                if track.misses >= 3:
                    track.misses = 999  # 待清理

        # 阶段四：实体确认晋升与噪点清理
        survived_tracks = []
        for track in self.tracks:
            # 晋升判定：命中数 >= confirm_hits 且总置信度达标
            if track.state == TrackState.TENTATIVE:
                if track.hits >= self.confirm_hits and track.total_score >= self.min_score_sum:
                    track.state = TrackState.ACTIVE
                elif track.misses >= 3:
                    continue  # 丢弃未确认的噪点
            survived_tracks.append(track)
        self.tracks = survived_tracks

        # 汇总
        self._aggregate_counts()

    def _aggregate_counts(self) -> None:
        """汇总所有已确认实体（包括 ACTIVE 和 LOST），应用软投票和赛制上限约束。"""
        raw_counts = {name: 0 for name in self.class_names}
        track_scores = defaultdict(list)

        # 1. 统计所有已确认实体（ACTIVE 与 LOST）
        for track in self.tracks:
            if track.state in (TrackState.ACTIVE, TrackState.LOST):
                best_class, best_score = track.get_best_class_and_score()
                if best_class in self.class_set:
                    raw_counts[best_class] += 1
                    track_scores[best_class].append(best_score)

        # 2. 统计未知类别（OCR）
        for name, max_count in self.max_unknown_detections.items():
            raw_counts[name] = max_count

        # 3. 单类上限裁剪 (max_per_class)
        for class_name in self.class_names:
            if raw_counts[class_name] > self.max_per_class:
                raw_counts[class_name] = self.max_per_class

        # 4. 总数上限裁剪 (total_max)
        total = sum(raw_counts.values())
        if total > self.total_max:
            excess = total - self.total_max
            candidates = []
            for class_name, count in raw_counts.items():
                if count > 0:
                    if class_name not in self.unknown_classes:
                        scores = sorted(track_scores[class_name])
                        for s in scores:
                            candidates.append((s, class_name))
                    else:
                        for _ in range(count):
                            candidates.append((1.0, class_name))

            candidates.sort(key=lambda x: x[0])
            for _, class_name in candidates:
                if excess <= 0:
                    break
                if raw_counts[class_name] > 0:
                    raw_counts[class_name] -= 1
                    excess -= 1

        self.final_counts = raw_counts

    def get_counts(self) -> Dict[str, int]:
        """返回赛制规定的非零计数条目。"""
        return {
            class_name: count
            for class_name, count in self.final_counts.items()
            if count > 0
        }

    def clear(self) -> None:
        """重置内部状态，彻底清空，防止跨桌位数据泄漏。"""
        self.tracks.clear()
        self.final_counts = {name: 0 for name in self.class_names}
        for name in self.max_unknown_detections:
            self.max_unknown_detections[name] = 0
        Track._id_counter = 0