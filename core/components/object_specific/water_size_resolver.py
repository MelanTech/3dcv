"""Resolve 350 ml versus 550 ml water bottles from detector and depth evidence."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np

from core.types import Detection, Frame


class DisabledWaterSizeResolver:
    """No-op implementation used when water-size resolving is disabled."""

    def process(
        self,
        _frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        del table
        return detections


class WaterSizeResolver:
    """Use reliable bottle length only to resolve ambiguous detector decisions."""

    def __init__(
        self,
        config: Dict,
        class_registry: Dict,
        geometry_provider,
    ):
        self.small_class = str(config.get("small_class", "Water350ml"))
        self.large_class = str(config.get("large_class", "Water550ml"))
        if self.small_class == self.large_class:
            raise ValueError("water_size_resolver classes must be different")
        self.water_classes = {self.small_class, self.large_class}
        ambiguity = config.get("detector_ambiguity", {})
        self.paired_score_margin_max = float(
            ambiguity.get("paired_score_margin_max", 0.2)
        )
        self.winner_score_max = float(ambiguity.get("winner_score_max", 0.45))
        if not 0.0 <= self.paired_score_margin_max <= 1.0:
            raise ValueError(
                "water_size_resolver.detector_ambiguity."
                "paired_score_margin_max must be in [0, 1]"
            )
        if not 0.0 <= self.winner_score_max <= 1.0:
            raise ValueError(
                "water_size_resolver.detector_ambiguity."
                "winner_score_max must be in [0, 1]"
            )

        ranges = config.get("length_ranges_m", {})
        self.small_range = self._read_range(
            ranges,
            self.small_class,
            (0.145, 0.195),
        )
        self.large_range = self._read_range(
            ranges,
            self.large_class,
            (0.2, 0.245),
        )

        geometry = config.get("geometry", {})
        self.min_points = int(geometry.get("min_points", 40))
        self.min_valid_depth_ratio = float(
            geometry.get("min_valid_depth_ratio", 0.03)
        )
        self.min_linearity_ratio = float(
            geometry.get("min_linearity_ratio", 2.5)
        )
        self.min_axis_coverage_ratio = float(
            geometry.get("min_axis_coverage_ratio", 0.6)
        )
        self.min_occupied_bin_ratio = float(
            geometry.get("min_occupied_bin_ratio", 0.5)
        )
        self.max_endpoint_gap_ratio = float(
            geometry.get("max_endpoint_gap_ratio", 0.2)
        )
        self.min_endpoint_points = int(geometry.get("min_endpoint_points", 3))
        self.min_endpoint_support_ratio = float(
            geometry.get("min_endpoint_support_ratio", 0.2)
        )
        self.require_plane_fit = bool(geometry.get("require_plane_fit", True))
        self.length_valid_margin_m = float(
            geometry.get("length_valid_margin_m", 0.02)
        )
        self.measurement_options = {
            "sample_stride": int(geometry.get("sample_stride", 2)),
            "max_points": int(geometry.get("max_points", 2000)),
            "min_object_height_m": float(
                geometry.get("min_object_height_m", 0.008)
            ),
            "max_object_height_m": float(
                geometry.get("max_object_height_m", 0.35)
            ),
            "quantile_low": float(geometry.get("quantile_low", 2.0)),
            "quantile_high": float(geometry.get("quantile_high", 98.0)),
            "coverage_bins": int(geometry.get("coverage_bins", 10)),
        }
        self.geometry_provider = geometry_provider

        id_to_class = {
            int(class_id): str(class_name)
            for class_id, class_name in class_registry.get(
                "detector_id_to_class", {}
            ).items()
        }
        self.class_to_id = {
            class_name: class_id for class_id, class_name in id_to_class.items()
        }

    @staticmethod
    def _read_range(
        ranges: Mapping,
        class_name: str,
        default: tuple[float, float],
    ) -> tuple[float, float]:
        values = ranges.get(class_name, default)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(
                f"water_size_resolver.length_ranges_m.{class_name} "
                "must contain [min, max]"
            )
        lower, upper = float(values[0]), float(values[1])
        if lower <= 0.0 or upper <= lower:
            raise ValueError(
                f"invalid water bottle length range for {class_name}: {values}"
            )
        return lower, upper

    def process(
        self,
        frame: Frame,
        detections: List[Detection],
        table: int,
    ) -> List[Detection]:
        del table
        resolved: List[Detection] = []
        for detection in detections:
            if detection.class_name not in self.water_classes:
                resolved.append(detection)
                continue
            resolved.append(self._resolve_detection(frame, detection))
        return resolved

    def _resolve_detection(self, frame: Frame, detection: Detection) -> Detection:
        detector_evidence = self._detector_ambiguity(detection)
        geometry = self._measure_geometry(frame, detection)
        invalid_reasons = self._geometry_invalid_reasons(geometry)
        geometry_valid = not invalid_reasons

        geometry_class = None
        geometry_abstain_reason = None
        if geometry_valid:
            geometry_class, geometry_abstain_reason = self._geometry_class(
                float(geometry["long_axis_length_m"])
            )

        override_applied = bool(
            geometry_valid
            and geometry_class is not None
            and detector_evidence["ambiguous"]
            and geometry_class != detection.class_name
        )
        class_name = geometry_class if override_applied else detection.class_name
        if override_applied:
            decision_reason = "geometry_override"
        elif not geometry_valid:
            decision_reason = "geometry_invalid"
        elif geometry_class is None:
            decision_reason = "geometry_abstained"
        elif not detector_evidence["ambiguous"]:
            decision_reason = "detector_confident"
        else:
            decision_reason = "geometry_agrees"

        evidence = dict(detection.evidence)
        evidence["water_size_resolver"] = {
            "source_class": detection.class_name,
            "final_class": class_name,
            "decision_reason": decision_reason,
            "override_applied": override_applied,
            "detector": detector_evidence,
            "geometry_valid": geometry_valid,
            "geometry_invalid_reasons": invalid_reasons,
            "geometry_class": geometry_class,
            "geometry_abstain_reason": geometry_abstain_reason,
            "geometry": geometry,
        }
        return Detection(
            class_name=class_name,
            class_id=self.class_to_id.get(class_name, detection.class_id),
            bbox=detection.bbox,
            score=detection.score,
            evidence=evidence,
        )

    def _measure_geometry(
        self,
        frame: Frame,
        detection: Detection,
    ) -> Dict:
        measure = getattr(
            self.geometry_provider,
            "measure_object_long_axis",
            None,
        )
        if frame.depth is None or measure is None:
            return {
                "available": False,
                "reason": "geometry_provider_unavailable",
            }
        try:
            result = measure(
                frame.depth,
                detection.bbox,
                **self.measurement_options,
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return {
                "available": False,
                "reason": "geometry_measurement_failed",
                "error": str(exc),
            }
        if not isinstance(result, dict):
            return {
                "available": False,
                "reason": "geometry_measurement_unavailable",
            }
        return result

    def _geometry_invalid_reasons(self, geometry: Mapping) -> List[str]:
        reasons: List[str] = []
        if not bool(geometry.get("available", False)):
            return [str(geometry.get("reason", "measurement_unavailable"))]

        table_model_source = str(geometry.get("table_model_source", ""))
        if self.require_plane_fit and not table_model_source.startswith("plane_fit"):
            reasons.append("table_plane_unavailable")

        length = float(geometry.get("long_axis_length_m", 0.0))
        min_length = min(self.small_range[0], self.large_range[0])
        min_length -= self.length_valid_margin_m
        max_length = max(self.small_range[1], self.large_range[1])
        max_length += self.length_valid_margin_m
        if not np.isfinite(length) or not min_length <= length <= max_length:
            reasons.append("length_out_of_range")
        if int(geometry.get("foreground_point_count", 0)) < self.min_points:
            reasons.append("insufficient_points")
        if (
            float(geometry.get("foreground_depth_ratio", 0.0))
            < self.min_valid_depth_ratio
        ):
            reasons.append("insufficient_depth")
        if float(geometry.get("linearity_ratio", 0.0)) < self.min_linearity_ratio:
            reasons.append("weak_long_axis")
        if (
            float(geometry.get("axis_coverage_ratio", 0.0))
            < self.min_axis_coverage_ratio
        ):
            reasons.append("incomplete_axis_coverage")
        if (
            float(geometry.get("occupied_bin_ratio", 0.0))
            < self.min_occupied_bin_ratio
        ):
            reasons.append("fragmented_axis")
        if (
            float(geometry.get("lower_endpoint_gap_ratio", 1.0))
            > self.max_endpoint_gap_ratio
            or float(geometry.get("upper_endpoint_gap_ratio", 1.0))
            > self.max_endpoint_gap_ratio
        ):
            reasons.append("endpoint_not_visible")
        if (
            int(geometry.get("lower_endpoint_point_count", 0))
            < self.min_endpoint_points
            or int(geometry.get("upper_endpoint_point_count", 0))
            < self.min_endpoint_points
            or float(geometry.get("endpoint_support_ratio", 0.0))
            < self.min_endpoint_support_ratio
        ):
            reasons.append("weak_endpoint_support")
        return reasons

    def _geometry_class(
        self,
        length_m: float,
    ) -> tuple[Optional[str], Optional[str]]:
        in_small_range = self.small_range[0] <= length_m <= self.small_range[1]
        in_large_range = self.large_range[0] <= length_m <= self.large_range[1]
        if in_small_range and in_large_range:
            return None, "overlapping_class_ranges"
        if in_small_range:
            return self.small_class, None
        if in_large_range:
            return self.large_class, None
        return None, "outside_class_ranges"

    def _detector_ambiguity(
        self,
        detection: Detection,
    ) -> Dict:
        for key in ("detector_class_scores", "class_scores"):
            scores = detection.evidence.get(key)
            pair = self._normalized_score_pair(scores)
            if pair is not None:
                return self._paired_ambiguity_evidence(pair, key)

        detector_evidence = detection.evidence.get("detector")
        if isinstance(detector_evidence, Mapping):
            for key in ("class_scores", "scores"):
                pair = self._normalized_score_pair(
                    detector_evidence.get(key)
                )
                if pair is not None:
                    return self._paired_ambiguity_evidence(
                        pair,
                        f"detector.{key}",
                    )

        raw_score = float(detection.score)
        winner_score = (
            float(np.clip(raw_score, 0.0, 1.0))
            if np.isfinite(raw_score)
            else 1.0
        )
        return {
            "source": "winning_class_score",
            "ambiguous": bool(winner_score <= self.winner_score_max),
            "winner_score": winner_score,
            "winner_score_max": self.winner_score_max,
            "small_probability": None,
            "large_probability": None,
            "paired_score_margin": None,
        }

    def _normalized_score_pair(
        self,
        scores,
    ) -> Optional[tuple[float, float]]:
        if not isinstance(scores, Mapping):
            return None
        try:
            small_score = float(scores[self.small_class])
            large_score = float(scores[self.large_class])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite([small_score, large_score])):
            return None
        small_score = max(0.0, small_score)
        large_score = max(0.0, large_score)
        total = small_score + large_score
        if total <= 0.0:
            return None
        return small_score / total, large_score / total

    def _paired_ambiguity_evidence(
        self,
        pair: tuple[float, float],
        source: str,
    ) -> Dict:
        small_probability, large_probability = pair
        margin = abs(small_probability - large_probability)
        return {
            "source": source,
            "ambiguous": bool(margin <= self.paired_score_margin_max),
            "winner_score": None,
            "winner_score_max": self.winner_score_max,
            "small_probability": small_probability,
            "large_probability": large_probability,
            "paired_score_margin": margin,
            "paired_score_margin_max": self.paired_score_margin_max,
        }


def build_water_size_resolver(
    config: Optional[Dict],
    class_registry: Optional[Dict],
    geometry_provider,
):
    """Build the configured water-size resolver."""
    config = config or {}
    if not bool(config.get("enabled", False)):
        return DisabledWaterSizeResolver()
    return WaterSizeResolver(
        config=config,
        class_registry=class_registry or {},
        geometry_provider=geometry_provider,
    )
