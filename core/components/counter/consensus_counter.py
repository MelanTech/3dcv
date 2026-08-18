"""Consensus counter: Bayesian normal classes with persistent OCR evidence."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.components.counter.bayesian_counter import BayesianCounter


class ConsensusCounter(BayesianCounter):
    """Bayesian counter with temporal consensus for sparse OCR classes.

    Normal object classes keep the existing Bayesian posterior behavior. OCR
    output classes are different: they are sparse and expensive, so the old
    max-over-time rule is recall-friendly but lets a single bad OCR frame create
    a final W001/W002/W003/W004 count. This counter requires repeated OCR
    evidence for each OCR count level.
    """

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        super().__init__(config, class_registry)
        self.ocr_min_positive_frames = max(
            1,
            int(config.get("ocr_min_positive_frames", 2)),
        )

    def _apply_total_constraint(self) -> None:
        """Apply Bayesian counts plus persistent OCR counts, then cap total."""
        counts: Dict[str, int] = {}

        for class_name in self.unknown_active:
            counts[class_name] = self._robust_unknown_count(class_name)

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

    def _robust_unknown_count(self, class_name: str) -> int:
        history = self.detection_history.get(class_name, [])
        if not history:
            return 0
        max_observed = min(max(history), self.max_per_class)
        for count in range(max_observed, 0, -1):
            positive_frames = sum(1 for observed in history if observed >= count)
            if positive_frames >= self.ocr_min_positive_frames:
                return count
        return 0
