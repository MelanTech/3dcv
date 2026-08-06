"""贝叶斯计数器：用滑动窗口 + 贝叶斯后验估计每类目标最可能的数量。

核心思路：把“每帧观测到的数量”看作真实数量在漏检/误检噪声下的带噪观测，
逐帧用似然更新每个类别数量的后验分布，最后取后验最大者作为该类计数，
并在超过总数上限时按置信度削减。
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from statistics import median
from typing import DefaultDict, Dict, List, Optional

import numpy as np

from core.components.counter.base import BaseCounter
from core.types import Detection


class BayesianCounter(BaseCounter):
    """基于贝叶斯后验的计数器。

    区分两类类别：
    - normal_classes：用贝叶斯后验估计数量；
    - unknown_classes：无法可靠建模，退化为取窗口内观测到的最大值。
    """

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        if class_registry is None:
            raise ValueError("BayesianCounter requires shared class_registry config")

        self.class_names = list(class_registry.get("result_classes", class_registry.get("classes", [])))
        if not self.class_names:
            raise ValueError("class_registry.result_classes must not be empty")
        self.class_set = set(self.class_names)
        self._validate_class_registry()

        self.max_per_class = int(config["max_object_count"])
        self.total_max = int(config["total_max"])

        self.unknown_classes = set(
            class_registry.get(
                "unknown_classes",
                class_registry.get("ocr_output_classes", []),
            )
        )
        self.unknown_active = [name for name in self.class_names if name in self.unknown_classes]
        self.normal_classes = [name for name in self.class_names if name not in self.unknown_classes]

        self.smooth_window = int(config.get("smooth_window", 5))
        self.min_positive_frames = int(config.get("min_positive_frames", 3))
        self.selection_threshold = float(config.get("selection_threshold", 0.5))
        if self.min_positive_frames > self.smooth_window:
            self.min_positive_frames = self.smooth_window

        self.miss_rate = float(config["miss_rate"])
        self.false_positive_rate = float(config["false_positive_rate"])

        self.detection_history: DefaultDict[str, List[int]] = defaultdict(list)
        self.history_window = {name: deque(maxlen=self.smooth_window) for name in self.class_names}
        self.max_detections = {name: 0 for name in self.unknown_active}
        self.prior = {
            name: np.ones(self.max_per_class + 1, dtype=float) / (self.max_per_class + 1)
            for name in self.normal_classes
        }
        self.final_counts = {name: 0 for name in self.class_names}

        self._precompute_likelihood_terms()

    def _validate_class_registry(self) -> None:
        if len(self.class_names) != len(self.class_set):
            raise ValueError("class_registry.result_classes contains duplicates")

    def _precompute_likelihood_terms(self) -> None:
        """预计算与观测无关的似然项（漏检/误检幂次），避免逐帧重复计算。"""
        max_count = self.max_per_class
        self._n_range = np.arange(max_count + 1, dtype=int)

        self._one_minus_miss_pow_n = np.power(1.0 - self.miss_rate, self._n_range)
        self._one_minus_fp_pow_max_minus_n = np.power(
            1.0 - self.false_positive_rate,
            max_count - self._n_range,
        )
        self._one_minus_miss_pow_k_arr = self._one_minus_miss_pow_n
        self._one_minus_fp_pow_max_minus_k_arr = np.power(
            1.0 - self.false_positive_rate,
            max_count - self._n_range,
        )

    def _likelihood_vectorized(self, observed_count: int) -> np.ndarray:
        """给定观测数量，计算真实数量取各个值的似然 P(观测 | 真实=n)。"""
        observed_count = int(max(0, min(observed_count, self.max_per_class)))
        n_range = self._n_range
        likelihood = np.zeros(self.max_per_class + 1, dtype=float)

        one_minus_miss_pow_n = self._one_minus_miss_pow_n
        one_minus_fp_pow_max_minus_n = self._one_minus_fp_pow_max_minus_n
        one_minus_miss_pow_k = self._one_minus_miss_pow_k_arr[observed_count]
        one_minus_fp_pow_max_minus_k = self._one_minus_fp_pow_max_minus_k_arr[observed_count]

        likelihood[observed_count] = (
            one_minus_miss_pow_n[observed_count]
            * one_minus_fp_pow_max_minus_n[observed_count]
        )

        mask_less = n_range < observed_count
        if np.any(mask_less):
            exponents = observed_count - n_range[mask_less]
            fp_pow = np.power(self.false_positive_rate, exponents)
            likelihood[mask_less] = (
                one_minus_miss_pow_n[mask_less]
                * fp_pow
                * one_minus_fp_pow_max_minus_k
            )

        mask_greater = n_range > observed_count
        if np.any(mask_greater):
            exponents = n_range[mask_greater] - observed_count
            miss_pow = np.power(self.miss_rate, exponents)
            likelihood[mask_greater] = (
                miss_pow
                * one_minus_miss_pow_k
                * one_minus_fp_pow_max_minus_n[mask_greater]
            )

        return likelihood

    def _robust_count_from_window(self, class_name: str) -> int:
        """从滑动窗口取稳健观测值：正样本帧数不足则记 0，否则取窗口中位数。"""
        window = self.history_window[class_name]
        if not window:
            return 0

        positive_frames = sum(1 for count in window if count > 0)
        if positive_frames < self.min_positive_frames:
            return 0

        robust_count = int(round(median(window)))
        return max(0, min(robust_count, self.max_per_class))

    def update(self, detections: List[Detection]) -> None:
        """用当前帧检测更新窗口，并对正常类别做一步贝叶斯后验更新。"""
        current_counts = Counter(
            detection.class_name
            for detection in detections
            if detection.class_name in self.class_set
        )

        for class_name in self.class_names:
            raw_count = current_counts.get(class_name, 0)
            capped_count = max(0, min(raw_count, self.max_per_class))
            self.detection_history[class_name].append(raw_count)
            self.history_window[class_name].append(capped_count)

            if class_name in self.max_detections:
                self.max_detections[class_name] = max(
                    self.max_detections[class_name],
                    capped_count,
                )

        for class_name in self.normal_classes:
            observed_count = self._robust_count_from_window(class_name)
            likelihood = self._likelihood_vectorized(observed_count)
            posterior = likelihood * self.prior[class_name]
            posterior_sum = posterior.sum()
            if posterior_sum <= 0.0:
                posterior[:] = 1.0 / posterior.size
            else:
                posterior /= posterior_sum
            self.prior[class_name] = posterior

        self._apply_total_constraint()

    def _apply_total_constraint(self) -> None:
        """由各类后验得到候选计数，若总数超过上限则按置信度削减到上限内。"""
        counts: Dict[str, int] = {}

        for class_name in self.unknown_active:
            counts[class_name] = min(
                self.max_detections.get(class_name, 0),
                self.max_per_class,
            )

        for class_name in self.normal_classes:
            posterior = self.prior[class_name]
            best_count = int(np.argmax(posterior))
            best_probability = float(posterior[best_count])
            counts[class_name] = (
                best_count
                if best_probability >= self.selection_threshold
                else 0
            )

        total = sum(counts.values())
        if total > self.total_max:
            self._decrease_counts(counts, total - self.total_max)
        else:
            self.final_counts = counts

    def _decrease_counts(self, counts: Dict[str, int], excess: int) -> None:
        """优先削减“减 1 后置信度损失最小”的类别，直到总数不超上限。"""
        if excess <= 0:
            self.final_counts = counts
            return

        candidates = []
        for class_name in self.normal_classes:
            current_count = counts.get(class_name, 0)
            if current_count > 0:
                probability = float(self.prior[class_name][current_count - 1])
                candidates.append((class_name, -probability, current_count))

        candidates.sort(key=lambda item: item[1])

        index = 0
        while excess > 0 and index < len(candidates):
            class_name, _, current_count = candidates[index]
            if current_count > 0:
                counts[class_name] = current_count - 1
                excess -= 1
            index += 1

        self.final_counts = counts

    def get_counts(self) -> Dict[str, int]:
        """返回最终计数，只保留数量大于 0 的类别。"""
        return {
            class_name: count
            for class_name, count in self.final_counts.items()
            if count > 0
        }

    def clear(self) -> None:
        """重置所有历史、窗口和后验，回到均匀先验的初始状态。"""
        self.detection_history = defaultdict(list)
        self.history_window = {name: deque(maxlen=self.smooth_window) for name in self.class_names}
        self.max_detections = {name: 0 for name in self.unknown_active}
        self.prior = {
            name: np.ones(self.max_per_class + 1, dtype=float) / (self.max_per_class + 1)
            for name in self.normal_classes
        }
        self.final_counts = {name: 0 for name in self.class_names}
