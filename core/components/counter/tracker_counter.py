"""追踪计数器：利用 2D 边框 IoU 追踪实例，利用置信度软投票进行稳健分类。

核心思路：把每一帧视为对桌面物理实体的一次“观测”。
通过 2D 坐标的重合度（IoU）跨帧关联目标，建立独立的生命周期（Track）。
对同一 Track 生命周期内的类别置信度进行累加（软投票），彻底消除闪烁，并应对临时遮挡。
"""
from __future__ import annotations

from collections import defaultdict
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


class Track:
    """独立的物理实例追踪记录。"""
    
    _id_counter = 0

    def __init__(self, detection: Detection):
        self.id = Track._id_counter
        Track._id_counter += 1

        self.bbox = detection.bbox
        self.class_votes: Dict[str, float] = defaultdict(float)
        self.hits = 1
        self.misses = 0

        # 初始化第一次投票
        self.class_votes[detection.class_name] += detection.score

    def update(self, detection: Detection):
        """用新的观测更新状态。"""
        # 对于旋转台上的移动物体，不能做平滑延迟，否则边框滞后会导致下一帧 IoU 匹配失败
        # 直接使用最新的观测框
        self.bbox = detection.bbox
        
        # 增加软投票分数
        self.class_votes[detection.class_name] += detection.score
        
        self.hits += 1
        self.misses = 0

    def mark_miss(self):
        """记录漏检。"""
        self.misses += 1
        
    def get_best_class_and_score(self) -> Tuple[str, float]:
        """获取目前累积票数最高的类别及其总票数。"""
        if not self.class_votes:
            return "Unknown", 0.0
        best_class = max(self.class_votes.items(), key=lambda item: item[1])
        return best_class[0], best_class[1]


class TrackerCounter(BaseCounter):
    """基于实例追踪的计数器。
    
    解决贝叶斯计数器缺乏空间隔离、容易因类别闪烁导致总数崩溃的问题。
    """

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        if class_registry is None:
            raise ValueError("TrackerCounter requires shared class_registry config")

        self.class_names = list(class_registry.get("result_classes", class_registry.get("classes", [])))
        if not self.class_names:
            raise ValueError("class_registry.result_classes must not be empty")
        self.class_set = set(self.class_names)

        # 未知/OCR 类别：不进行追踪，因为 OCR 往往不是每帧都出结果，直接用历史最大值
        self.unknown_classes = set(
            class_registry.get(
                "unknown_classes",
                class_registry.get("ocr_output_classes", []),
            )
        )
        self.max_unknown_detections = {name: 0 for name in self.class_names if name in self.unknown_classes}

        self.max_per_class = int(config.get("max_object_count", 5))
        self.total_max = int(config.get("total_max", 15))

        # 追踪参数
        self.iou_threshold = float(config.get("iou_threshold", 0.3))
        self.center_dist_threshold = float(config.get("center_dist_threshold", 50.0))
        self.confirm_hits = int(config.get("confirm_hits", 3))
        self.max_lost_frames = int(config.get("max_lost_frames", 10))

        self.tracks: List[Track] = []
        self.final_counts = {name: 0 for name in self.class_names}

    def update(self, detections: List[Detection]) -> None:
        """输入当前帧的所有检测，进行目标关联和追踪更新。"""
        
        # 1. 专门处理 OCR 等“未知类别” (按原有的 MAX 逻辑)
        unknown_detections = [d for d in detections if d.class_name in self.unknown_classes]
        current_unknown_counts = defaultdict(int)
        for d in unknown_detections:
            current_unknown_counts[d.class_name] += 1
            
        for name, count in current_unknown_counts.items():
            self.max_unknown_detections[name] = max(self.max_unknown_detections[name], count)
        
        # 2. 过滤出需要追踪的正常类别
        valid_detections = [
            d for d in detections 
            if d.class_name in self.class_set and d.class_name not in self.unknown_classes
        ]
        
        unmatched_detections = list(range(len(valid_detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # 贪心匹配 (IoU 为主，中心点距离为辅)
        matches = []
        if len(valid_detections) > 0 and len(self.tracks) > 0:
            match_matrix = []
            for t_idx, track in enumerate(self.tracks):
                tx_c = (track.bbox[0] + track.bbox[2]) / 2.0
                ty_c = (track.bbox[1] + track.bbox[3]) / 2.0
                
                for d_idx, det in enumerate(valid_detections):
                    dx_c = (det.bbox[0] + det.bbox[2]) / 2.0
                    dy_c = (det.bbox[1] + det.bbox[3]) / 2.0
                    
                    iou = compute_iou(track.bbox, det.bbox)
                    dist = ((tx_c - dx_c) ** 2 + (ty_c - dy_c) ** 2) ** 0.5
                    
                    best_class, _ = track.get_best_class_and_score()
                    same_class = (best_class == det.class_name)
                    
                    # 综合判定：IoU达标(允许跨类抖动)，或中心点达标(仅限同类抢救，严格限制物理距离)
                    if iou >= self.iou_threshold or (dist <= self.center_dist_threshold and same_class):
                        score = iou - (dist / 1000.0)
                        if same_class:
                            score += 10.0
                            
                        match_matrix.append((score, t_idx, d_idx))
            
            # 按综合分数从大到小排序，优先匹配最吻合的
            match_matrix.sort(key=lambda x: x[0], reverse=True)
            
            for score, t_idx, d_idx in match_matrix:
                if t_idx in unmatched_tracks and d_idx in unmatched_detections:
                    matches.append((t_idx, d_idx))
                    unmatched_tracks.remove(t_idx)
                    unmatched_detections.remove(d_idx)
                    
        # 1. 更新成功匹配的 Track
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(valid_detections[d_idx])
            
        # 2. 为未匹配的检测创建新 Track，并过滤掉高度重合的 YOLO 冗余框
        for d_idx in unmatched_detections:
            det = valid_detections[d_idx]
            is_duplicate = False
            for track in self.tracks:
                if compute_iou(track.bbox, det.bbox) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                self.tracks.append(Track(det))
            
        # 3. 惩罚未匹配的 Track
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_miss()
            
        # 4. 清理丢失太久的 Track
        self.tracks = [t for t in self.tracks if t.misses <= self.max_lost_frames]
        
        # 5. 汇总数量
        self._aggregate_counts()

    def _aggregate_counts(self) -> None:
        """从活着的 Track 汇总最终计数，并应用上限裁剪。"""
        raw_counts = {name: 0 for name in self.class_names}
        track_scores = defaultdict(list)
        
        # 加入正常类别的追踪计数
        for track in self.tracks:
            # 只有连续命中达到要求，才算是真实存在的物体
            if track.hits >= self.confirm_hits:
                best_class, best_score = track.get_best_class_and_score()
                if best_class in self.class_set:
                    raw_counts[best_class] += 1
                    track_scores[best_class].append(best_score)
                    
        # 加入 OCR 类别的最大保留计数
        for name, max_count in self.max_unknown_detections.items():
            raw_counts[name] = max_count
            
        # 处理 max_per_class
        for class_name in self.class_names:
            if raw_counts[class_name] > self.max_per_class:
                raw_counts[class_name] = self.max_per_class
                
        # 处理 total_max (削减那些软投票分数最低的，OCR 类别不参与削减)
        total = sum(raw_counts.values())
        if total > self.total_max:
            excess = total - self.total_max
            # 收集所有类别的所有可削减样本的分数
            candidates = []
            for class_name, count in raw_counts.items():
                if count > 0:
                    if class_name not in self.unknown_classes:
                        scores = sorted(track_scores[class_name])
                        for s in scores:
                            candidates.append((s, class_name))
                    else:
                        # 对于未知类别（OCR书本），我们赋予一个中等偏上的固定假想分数(比如 1.0)。
                        # 这样如果假书本泛滥成灾导致总数超过 15，它们也会被削减，而不是拿真香肠献祭。
                        for _ in range(count):
                            candidates.append((1.0, class_name))
                        
            candidates.sort(key=lambda x: x[0]) # 分数越低越容易被削减
            
            for _, class_name in candidates:
                if excess <= 0:
                    break
                if raw_counts[class_name] > 0:
                    raw_counts[class_name] -= 1
                    excess -= 1
                    
        self.final_counts = raw_counts

    def get_counts(self) -> Dict[str, int]:
        """返回最终计数，只保留数量大于 0 的类别。"""
        return {
            class_name: count
            for class_name, count in self.final_counts.items()
            if count > 0
        }

    def clear(self) -> None:
        """重置内部状态，回到初始状态。"""
        self.tracks.clear()
        self.final_counts = {name: 0 for name in self.class_names}
        for name in self.max_unknown_detections:
            self.max_unknown_detections[name] = 0
        Track._id_counter = 0
