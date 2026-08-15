"""深度过滤器：利用深度图把检测投影到 3D，只保留落在桌面范围内的目标。

流程概述：
1. 用相机内参把每个检测框中心 + 深度反投影成相机坐标系下的 3D 点；
2. 优先从桌子框内的深度点拟合桌面平面，转换到“桌面坐标系”；
3. 根据平面内点估计桌面的 3D 包围盒，剔除盒外的目标（可选保留桌子本身）。
平面拟合失败时回退到固定旋转 + 桌面中心平移的旧逻辑。
可选依赖 open3d 做点云可视化，仅在 visualize=True 时启用。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.components.filter.base import BaseFilter
from core.infra import pause_clock
from core.types import Detection, Frame


class DepthFilter(BaseFilter):
    """基于深度与桌面 3D 范围的空间过滤器。"""

    def __init__(self, config: Dict, intrinsic_override: Optional[Dict] = None):
        intrinsic = intrinsic_override or config["intrinsic"]
        filter_config = config["depth_filter"]

        object_depth_config = filter_config.get("object_depth", {})
        visualization_config = filter_config.get("visualization", {})
        filtering_config = filter_config.get("filtering", {})
        plane_fit_config = filter_config.get("plane_fit", {})
        footprint_config = filter_config.get("footprint", {})
        smoothing_config = filter_config.get("smoothing", {})
        fallback_config = filter_config.get("fallback", {})

        def read(section: Dict, key: str, default, legacy_key: Optional[str] = None):
            if key in section:
                return section[key]
            return filter_config.get(legacy_key or key, default)

        window = visualization_config.get("window", filter_config.get("window", {}))
        self.window_width = int(window.get("width", intrinsic["width"]))
        self.window_height = int(window.get("height", intrinsic["height"]))

        self.fx = float(intrinsic["fx"])
        self.fy = float(intrinsic["fy"])
        self.cx = float(intrinsic["cx"])
        self.cy = float(intrinsic["cy"])
        self.intrinsic_config = {
            "width": int(intrinsic["width"]),
            "height": int(intrinsic["height"]),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

        self.dynamic_scale = float(
            read(fallback_config, "dynamic_scale", 0.8)
        )
        self.coord_sample_num = int(
            read(object_depth_config, "coord_sample_num", 20)
        )
        self.object_depth_min_valid_pixels = max(
            1,
            int(read(object_depth_config, "min_valid_pixels", 5, "object_depth_min_valid_pixels")),
        )
        expand_ratios = read(
            object_depth_config,
            "expand_ratios",
            [0.15, 0.3],
            "object_depth_expand_ratios",
        )
        self.object_depth_expand_ratios = tuple(
            max(0.0, float(ratio)) for ratio in expand_ratios
        )
        self.object_depth_table_plane_fallback = bool(
            read(
                object_depth_config,
                "table_plane_fallback",
                True,
                "object_depth_table_plane_fallback",
            )
        )
        self.object_depth_plane_offset_m = float(
            read(
                object_depth_config,
                "plane_offset_m",
                0.0,
                "object_depth_plane_offset_m",
            )
        )
        self.use_support_point = bool(
            read(filtering_config, "use_support_point", True)
        )
        # 支撑点落地判定：脚点真实深度比桌面平面交点远出该阈值(米)时，
        # 认为物体站在紧邻桌子的地面而非桌面上，改用真实脚点参与过滤。
        # <=0 时关闭该校验，退回纯平面投影（旧行为）。
        self.support_point_ground_reject_gap_m = float(
            read(filtering_config, "ground_reject_gap_m", 0.0)
        )
        # 脚点深度采样：从 bbox 底部该比例高度的横带内取深度（兼容斜放物体）。
        self.support_point_band_ratio = float(
            read(filtering_config, "ground_reject_band_ratio", 0.15)
        )
        # 用该较近分位数代表脚点深度，抑制透出背景的远端噪声。
        self.support_point_near_percentile = float(
            read(filtering_config, "ground_reject_near_percentile", 25.0)
        )
        # gap 合理性上限(米)：超过则视为深度读取失败，不据此过滤。<=0 关闭。
        self.support_point_ground_reject_max_gap_m = float(
            read(filtering_config, "ground_reject_max_gap_m", 0.25)
        )
        # 方案 A：support_point 交点落桌外时，用物体真实深度反投影的中心复核落脚位置，
        # 纠正高/边缘物体因射线角度导致的横向漂移误过滤。
        self.support_point_reproject_recover = bool(
            read(filtering_config, "reproject_recover", True)
        )
        # 复核只对高物体启用：真实中心高出桌面平面超过该阈值(米)才复核，
        # 矮物体射线漂移小、无需复核，可避免误救桌外物品。<=0 表示不限高度。
        self.support_point_reproject_min_height_m = float(
            read(filtering_config, "reproject_min_height_m", 0.08)
        )
        self.visualize = bool(
            read(visualization_config, "enabled", False, "visualize")
        )
        self.keep_table = bool(read(filtering_config, "keep_table", True))
        self.depth_trunc_m = float(
            read(visualization_config, "depth_trunc_m", 2.0)
        )
        self.point_cloud_stride = int(
            read(visualization_config, "point_cloud_stride", 2)
        )
        self.plane_fit_enabled = bool(
            read(plane_fit_config, "enabled", True, "plane_fit_enabled")
        )
        self.plane_sample_stride = max(
            1,
            int(read(plane_fit_config, "sample_stride", 4, "plane_sample_stride")),
        )
        self.plane_max_points = max(
            3,
            int(read(plane_fit_config, "max_points", 3000, "plane_max_points")),
        )
        self.plane_min_points = max(
            3,
            int(read(plane_fit_config, "min_points", 80, "plane_min_points")),
        )
        self.plane_ransac_iterations = max(
            1,
            int(
                read(
                    plane_fit_config,
                    "ransac_iterations",
                    80,
                    "plane_ransac_iterations",
                )
            ),
        )
        self.plane_update_interval_frames = max(
            1,
            int(
                read(
                    plane_fit_config,
                    "update_interval_frames",
                    1,
                    "plane_update_interval_frames",
                )
            ),
        )
        self.plane_early_stop_inlier_ratio = float(
            read(
                plane_fit_config,
                "early_stop_inlier_ratio",
                0.75,
                "plane_early_stop_inlier_ratio",
            )
        )
        self.plane_distance_thresh_m = float(
            read(plane_fit_config, "distance_thresh_m", 0.015, "plane_distance_thresh_m")
        )
        self.plane_min_inlier_ratio = float(
            read(plane_fit_config, "min_inlier_ratio", 0.35, "plane_min_inlier_ratio")
        )
        self.plane_use_unmasked_points_for_orientation = bool(
            read(plane_fit_config, "use_unmasked_points_for_orientation", True)
        )
        self.table_range_percentile = float(
            read(footprint_config, "range_percentile", 2.0, "table_range_percentile")
        )
        self.table_size_scale = float(
            read(footprint_config, "size_scale", 1.0, "table_size_scale")
        )
        self.table_range_margin_m = float(
            read(footprint_config, "range_margin_m", 0.02, "table_range_margin_m")
        )
        vertical_expansion_config = footprint_config.get("vertical_expansion", {})
        self.footprint_vertical_expansion_enabled = bool(
            read(vertical_expansion_config, "enabled", False)
        )
        self.footprint_vertical_expansion_angle_deg = float(
            read(vertical_expansion_config, "angle_deg", 0.0)
        )
        if not 0.0 <= self.footprint_vertical_expansion_angle_deg < 90.0:
            raise ValueError("depth_filter.footprint.vertical_expansion.angle_deg must be in [0, 90)")
        self.footprint_vertical_expansion_tan = float(
            np.tan(np.deg2rad(self.footprint_vertical_expansion_angle_deg))
        )
        self.footprint_vertical_expansion_max_extra_margin_m = max(
            0.0,
            float(read(vertical_expansion_config, "max_extra_margin_m", 0.0)),
        )
        self.footprint_vertical_expansion_min_height_m = max(
            0.0,
            float(read(vertical_expansion_config, "min_height_m", 0.0)),
        )
        self.table_footprint_mode = str(
            read(footprint_config, "mode", "auto", "table_footprint_mode")
        ).strip().lower()
        if self.table_footprint_mode not in ("auto", "rectangle", "ellipse", "by_table"):
            raise ValueError(
                "depth_filter.table_footprint_mode must be auto, rectangle, ellipse, or by_table"
            )
        table_modes = footprint_config.get("table_modes", {})
        self.table_footprint_by_table = {
            int(table): str(mode).strip().lower()
            for table, mode in table_modes.items()
        }
        for table, mode in self.table_footprint_by_table.items():
            if mode not in ("rectangle", "ellipse"):
                raise ValueError(
                    f"depth_filter.footprint.table_modes[{table}] must be rectangle or ellipse"
                )
        self.round_table_fill_ratio_max = float(
            read(
                footprint_config,
                "round_fill_ratio_max",
                0.88,
                "round_table_fill_ratio_max",
            )
        )
        self.round_table_corner_ratio_max = float(
            read(
                footprint_config,
                "round_corner_ratio_max",
                0.03,
                "round_table_corner_ratio_max",
            )
        )
        self.footprint_switch_min_frames = max(
            1,
            int(read(footprint_config, "switch_min_frames", 3)),
        )
        self.table_roi_shrink_ratio = float(
            read(plane_fit_config, "roi_shrink_ratio", 0.04, "table_roi_shrink_ratio")
        )
        self.table_box_height_m = float(
            read(visualization_config, "table_box_height_m", 0.06)
        )
        self.table_model_smoothing_alpha = max(
            0.0,
            min(
                1.0,
                float(
                    read(
                        smoothing_config,
                        "alpha",
                        0.25,
                        "table_model_smoothing_alpha",
                    )
                ),
            ),
        )
        self.table_model_reset_distance_m = float(
            read(
                smoothing_config,
                "reset_distance_m",
                0.35,
                "table_model_reset_distance_m",
            )
        )
        self.table_model_reject_distance_m = float(
            read(
                smoothing_config,
                "reject_distance_m",
                0.12,
                "table_model_reject_distance_m",
            )
        )
        self.table_model_reject_angle_deg = float(
            read(
                smoothing_config,
                "reject_angle_deg",
                18.0,
                "table_model_reject_angle_deg",
            )
        )
        self.table_model_reject_size_ratio = float(
            read(
                smoothing_config,
                "reject_size_ratio",
                1.8,
                "table_model_reject_size_ratio",
            )
        )
        self.table_model_hold_on_failure = bool(
            read(
                smoothing_config,
                "hold_on_failure",
                True,
                "table_model_hold_on_failure",
            )
        )
        self.table_model_max_hold_frames = max(
            0,
            int(
                read(
                    smoothing_config,
                    "max_hold_frames",
                    8,
                    "table_model_max_hold_frames",
                )
            ),
        )

        self.visualizer = None
        self.intrinsic = None
        self.table_bbox: Optional[object] = None
        self.point_cloud = None
        self.coordinate_frame = None
        self.markers_pcd = None
        self.detection_boxes = None
        self.show_detection_boxes = bool(
            read(visualization_config, "show_detection_boxes", True)
        )
        self.viz_point_size = float(
            read(visualization_config, "point_size", 3.0)
        )
        self.viz_color_by_keep = bool(
            read(visualization_config, "color_by_keep", True)
        )
        self.viz_line_width = float(
            read(visualization_config, "line_width", 3.0)
        )
        self.viz_show_labels = bool(
            read(visualization_config, "show_labels", True)
        )
        self.detection_box_depth_mode = read(
            visualization_config,
            "detection_box_depth_mode",
            "median",
        )
        if self.detection_box_depth_mode not in ("median", "center"):
            raise ValueError("depth_filter.detection_box_depth_mode must be median or center")

        # O3D GUI 相关句柄。
        self._o3d = None
        self._gui = None
        self._rendering = None
        self.window = None
        self.scene_widget = None
        self._pause_button = None
        self._labels = []
        self.link_lines = None
        self._geom_added = set()
        self._line_material = None
        self._point_material = None
        self._marker_material = None
        self._viz_paused = False
        self._viz_step = False

        if self.visualize:
            self._init_o3d_visualizer()

        self.first_frame = True
        rot_z = self._rotation_matrix_z(np.pi)
        rot_x = self._rotation_matrix_x(np.pi / 4.0)
        self.fallback_rotation_matrix = (rot_x @ rot_z).astype(np.float64)
        self.rotation_matrix = self.fallback_rotation_matrix.copy()
        self.translation_vector = np.zeros(3, dtype=np.float64)
        self.transform_matrix = np.eye(4, dtype=np.float64)
        self.transform_matrix[:3, :3] = self.rotation_matrix
        self.current_table_range: Optional[Dict] = None
        self.table_model_source = "fallback"
        self.smoothed_rotation_matrix: Optional[np.ndarray] = None
        self.smoothed_translation_vector: Optional[np.ndarray] = None
        self.smoothed_table_range: Optional[Dict] = None
        self.table_model_missing_frames = 0
        self.frame_index = 0
        self.current_table = 0
        self.last_plane_fit_frame = -10**9
        self.locked_footprint: Optional[str] = None
        self.pending_footprint: Optional[str] = None
        self.pending_footprint_count = 0
        self.ray_shape: Optional[Tuple[int, int]] = None
        self.ray_x_map: Optional[np.ndarray] = None
        self.ray_y_map: Optional[np.ndarray] = None
        self.random = np.random.default_rng(2026)

    @staticmethod
    def _rotation_matrix_x(angle: float) -> np.ndarray:
        cos_v = np.cos(angle)
        sin_v = np.sin(angle)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_v, -sin_v],
                [0.0, sin_v, cos_v],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_matrix_z(angle: float) -> np.ndarray:
        cos_v = np.cos(angle)
        sin_v = np.sin(angle)
        return np.array(
            [
                [cos_v, -sin_v, 0.0],
                [sin_v, cos_v, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _is_table(detection: Detection) -> bool:
        return detection.class_id == 0 or detection.class_name == "Table"

    def _ensure_ray_maps(self, width: int, height: int) -> None:
        """预计算每个像素的归一化相机射线，避免每帧重复算内参反投影。"""
        shape = (int(height), int(width))
        if self.ray_shape == shape and self.ray_x_map is not None and self.ray_y_map is not None:
            return

        xs = np.arange(width, dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        self.ray_x_map = (grid_x - self.cx) / self.fx
        self.ray_y_map = (grid_y - self.cy) / self.fy
        self.ray_shape = shape

    @staticmethod
    def _clip_bbox(
        bbox: Tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height - 1, int(y1)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _depth_points_from_bbox(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        exclude_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Optional[np.ndarray]:
        """从桌子框内采样深度点并反投影到相机坐标系，用于拟合桌面平面。"""
        height, width = depth_image.shape[:2]
        self._ensure_ray_maps(width, height)
        clipped = self._clip_bbox(bbox, width, height)
        if clipped is None:
            return None

        x1, y1, x2, y2 = clipped
        margin_x = int((x2 - x1) * self.table_roi_shrink_ratio)
        margin_y = int((y2 - y1) * self.table_roi_shrink_ratio)
        x1 += margin_x
        x2 -= margin_x
        y1 += margin_y
        y2 -= margin_y
        if x2 <= x1 or y2 <= y1:
            return None

        xs = np.arange(x1, x2, self.plane_sample_stride, dtype=np.int32)
        ys = np.arange(y1, y2, self.plane_sample_stride, dtype=np.int32)
        if xs.size == 0 or ys.size == 0:
            return None

        grid_x, grid_y = np.meshgrid(xs, ys)
        depth_mm = depth_image[np.ix_(ys, xs)]
        depth_m = depth_mm.astype(np.float64) / 1000.0
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        if self.depth_trunc_m > 0:
            valid &= depth_m <= self.depth_trunc_m
        for exclude_bbox in exclude_bboxes or []:
            excluded = self._clip_bbox(exclude_bbox, width, height)
            if excluded is None:
                continue
            ex1, ey1, ex2, ey2 = excluded
            valid &= ~(
                (grid_x >= ex1)
                & (grid_x < ex2)
                & (grid_y >= ey1)
                & (grid_y < ey2)
            )
        if not np.any(valid):
            return None

        z = depth_m[valid]
        ray_x = self.ray_x_map[np.ix_(ys, xs)]
        ray_y = self.ray_y_map[np.ix_(ys, xs)]
        x = ray_x[valid] * z
        y = ray_y[valid] * z
        points = np.column_stack((x, y, z)).astype(np.float64, copy=False)
        if points.shape[0] > self.plane_max_points:
            indices = np.linspace(
                0,
                points.shape[0] - 1,
                self.plane_max_points,
                dtype=np.int32,
            )
            points = points[indices]
        return points if points.shape[0] >= self.plane_min_points else None

    def _depth_points_from_bbox_with_unmasked(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        exclude_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """一次采样同时返回 masked 点和 unmasked 点，避免同一 ROI 重复反投影。"""
        height, width = depth_image.shape[:2]
        self._ensure_ray_maps(width, height)
        clipped = self._clip_bbox(bbox, width, height)
        if clipped is None:
            return None, None

        x1, y1, x2, y2 = clipped
        margin_x = int((x2 - x1) * self.table_roi_shrink_ratio)
        margin_y = int((y2 - y1) * self.table_roi_shrink_ratio)
        x1 += margin_x
        x2 -= margin_x
        y1 += margin_y
        y2 -= margin_y
        if x2 <= x1 or y2 <= y1:
            return None, None

        xs = np.arange(x1, x2, self.plane_sample_stride, dtype=np.int32)
        ys = np.arange(y1, y2, self.plane_sample_stride, dtype=np.int32)
        if xs.size == 0 or ys.size == 0:
            return None, None

        grid_x, grid_y = np.meshgrid(xs, ys)
        depth_mm = depth_image[np.ix_(ys, xs)]
        depth_m = depth_mm.astype(np.float64) / 1000.0
        base_valid = np.isfinite(depth_m) & (depth_m > 0.0)
        if self.depth_trunc_m > 0:
            base_valid &= depth_m <= self.depth_trunc_m

        masked_valid = base_valid.copy()
        for exclude_bbox in exclude_bboxes or []:
            excluded = self._clip_bbox(exclude_bbox, width, height)
            if excluded is None:
                continue
            ex1, ey1, ex2, ey2 = excluded
            masked_valid &= ~(
                (grid_x >= ex1)
                & (grid_x < ex2)
                & (grid_y >= ey1)
                & (grid_y < ey2)
            )

        ray_x = self.ray_x_map[np.ix_(ys, xs)]
        ray_y = self.ray_y_map[np.ix_(ys, xs)]

        def build_points(valid: np.ndarray) -> Optional[np.ndarray]:
            if not np.any(valid):
                return None
            z = depth_m[valid]
            x = ray_x[valid] * z
            y = ray_y[valid] * z
            points = np.column_stack((x, y, z)).astype(np.float64, copy=False)
            if points.shape[0] > self.plane_max_points:
                indices = np.linspace(
                    0,
                    points.shape[0] - 1,
                    self.plane_max_points,
                    dtype=np.int32,
                )
                points = points[indices]
            return points if points.shape[0] >= self.plane_min_points else None

        return build_points(masked_valid), build_points(base_valid)

    def _fit_plane_ransac(
        self,
        points: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """用 RANSAC 在桌子 ROI 深度点中找最大平面，返回法向量/中心/内点。"""
        point_count = points.shape[0]
        if point_count < self.plane_min_points:
            return None

        best_mask = None
        best_count = 0
        for _ in range(self.plane_ransac_iterations):
            sample_indices = self.random.choice(point_count, size=3, replace=False)
            p0, p1, p2 = points[sample_indices]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue

            normal = normal / norm
            distance = np.abs((points - p0) @ normal)
            mask = distance <= self.plane_distance_thresh_m
            inlier_count = int(np.count_nonzero(mask))
            if inlier_count > best_count:
                best_count = inlier_count
                best_mask = mask
                if best_count / float(point_count) >= self.plane_early_stop_inlier_ratio:
                    break

        if best_mask is None:
            return None

        inlier_ratio = best_count / float(point_count)
        if best_count < self.plane_min_points:
            return None
        if inlier_ratio < self.plane_min_inlier_ratio:
            return None

        inliers = points[best_mask]
        center = np.mean(inliers, axis=0)
        centered = inliers - center
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        normal = vh[-1]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            return None
        normal = normal / normal_norm

        # 让桌面法向量朝向相机，桌面上方的物体在 table-y 方向为正。
        if np.dot(normal, -center) < 0:
            normal = -normal

        return normal, center, inliers, inlier_ratio

    def _points_near_plane(
        self,
        points: np.ndarray,
        normal: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """从候选点中取出贴近当前桌面平面的点。"""
        if points.shape[0] == 0:
            return points
        distance = np.abs((points - center) @ normal)
        return points[distance <= self.plane_distance_thresh_m]

    @staticmethod
    def _initial_table_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """根据桌面法向量生成临时平面坐标轴，供 PCA 估计桌面主方向使用。"""
        y_axis = normal / np.linalg.norm(normal)
        camera_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = camera_x - np.dot(camera_x, y_axis) * y_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            camera_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            x_axis = camera_z - np.dot(camera_z, y_axis) * y_axis
            x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / x_norm
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        return x_axis, y_axis, z_axis

    def _table_rotation_from_plane_points(
        self,
        normal: np.ndarray,
        center: np.ndarray,
        inliers: np.ndarray,
    ) -> np.ndarray:
        """用桌面平面内点的 PCA 主方向估计桌子长宽方向。"""
        x_axis, y_axis, z_axis = self._initial_table_axes(normal)
        plane_points = inliers - center
        plane_coords = np.column_stack((plane_points @ x_axis, plane_points @ z_axis))
        if plane_coords.shape[0] >= 3:
            covariance = np.cov(plane_coords, rowvar=False)
            try:
                eigen_values, eigen_vectors = np.linalg.eigh(covariance)
                principal = eigen_vectors[:, int(np.argmax(eigen_values))]
                x_axis = principal[0] * x_axis + principal[1] * z_axis
                x_axis = x_axis / np.linalg.norm(x_axis)
                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / np.linalg.norm(z_axis)
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
            except np.linalg.LinAlgError:
                pass

        # 固定方向符号，避免连续帧坐标轴来回翻转。
        if np.dot(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64)) < 0:
            x_axis = -x_axis
            z_axis = -z_axis
        return np.vstack((x_axis, y_axis, z_axis)).astype(np.float64)

    @staticmethod
    def _orthonormalize_rotation(matrix: np.ndarray) -> np.ndarray:
        """把线性插值后的矩阵拉回最近的正交旋转矩阵。"""
        u, _s, vh = np.linalg.svd(matrix)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vh
        return rotation.astype(np.float64)

    @staticmethod
    def _camera_origin_from_transform(
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        return -rotation.T @ translation

    @staticmethod
    def _align_rotation_to_previous(
        rotation: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> np.ndarray:
        """让 PCA 轴方向与上一帧一致，避免等价坐标轴正负号翻转。"""
        if previous is None:
            return rotation

        aligned = rotation.copy()
        if np.dot(aligned[1], previous[1]) < 0:
            aligned[1] *= -1.0
            aligned[2] *= -1.0
        if np.dot(aligned[0], previous[0]) < 0:
            aligned[0] *= -1.0
            aligned[2] *= -1.0
        return aligned

    def _smooth_table_transform(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """对桌面坐标系做指数平滑，稳定点云可视化坐标。"""
        previous_rotation = self.smoothed_rotation_matrix
        previous_translation = self.smoothed_translation_vector
        rotation = self._align_rotation_to_previous(rotation, previous_rotation)

        reset = previous_rotation is None or previous_translation is None
        if not reset and self.table_model_reset_distance_m > 0:
            previous_origin = self._camera_origin_from_transform(
                previous_rotation,
                previous_translation,
            )
            current_origin = self._camera_origin_from_transform(rotation, translation)
            origin_delta = np.linalg.norm(current_origin - previous_origin)
            reset = bool(origin_delta > self.table_model_reset_distance_m)

        alpha = self.table_model_smoothing_alpha
        if reset or alpha >= 1.0:
            smoothed_rotation = rotation
            smoothed_translation = translation
        elif alpha <= 0.0:
            smoothed_rotation = previous_rotation
            smoothed_translation = previous_translation
        else:
            smoothed_rotation = self._orthonormalize_rotation(
                (1.0 - alpha) * previous_rotation + alpha * rotation
            )
            smoothed_translation = (
                (1.0 - alpha) * previous_translation + alpha * translation
            )

        self.smoothed_rotation_matrix = smoothed_rotation
        self.smoothed_translation_vector = smoothed_translation
        return smoothed_rotation, smoothed_translation, reset

    def _is_table_transform_outlier(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> bool:
        """若新桌面模型相对上一帧稳定模型跳变过大，则拒绝该帧。"""
        previous_rotation = self.smoothed_rotation_matrix
        previous_translation = self.smoothed_translation_vector
        if previous_rotation is None or previous_translation is None:
            return False

        if self.table_model_reject_distance_m > 0:
            previous_origin = self._camera_origin_from_transform(
                previous_rotation,
                previous_translation,
            )
            current_origin = self._camera_origin_from_transform(rotation, translation)
            origin_delta = np.linalg.norm(current_origin - previous_origin)
            if origin_delta > self.table_model_reject_distance_m:
                return True

        if self.table_model_reject_angle_deg > 0:
            normal_dot = float(np.clip(np.dot(rotation[1], previous_rotation[1]), -1.0, 1.0))
            normal_angle = np.degrees(np.arccos(normal_dot))
            if normal_angle > self.table_model_reject_angle_deg:
                return True

        return False

    def _is_table_range_outlier(self, table_range: Dict) -> bool:
        """若新估计桌面长宽相对上一帧突变，则拒绝该帧。"""
        previous = self.smoothed_table_range
        if previous is None or self.table_model_reject_size_ratio <= 1.0:
            return False

        for key in ("estimated_width_m", "estimated_length_m"):
            if key not in table_range or key not in previous:
                continue
            current_value = float(table_range[key])
            previous_value = float(previous[key])
            if current_value <= 1e-6 or previous_value <= 1e-6:
                continue
            ratio = max(current_value / previous_value, previous_value / current_value)
            if ratio > self.table_model_reject_size_ratio:
                return True

        return False

    def _smooth_table_range(self, table_range: Dict, reset: bool) -> Dict:
        """对绿框边界做指数平滑，减少范围估计的边缘抖动。"""
        previous = self.smoothed_table_range
        alpha = self.table_model_smoothing_alpha
        footprint_changed = (
            previous is not None
            and table_range.get("footprint") != previous.get("footprint")
        )
        if reset or footprint_changed or previous is None or alpha >= 1.0:
            smoothed = dict(table_range)
        elif alpha <= 0.0:
            smoothed = dict(previous)
            smoothed["source"] = table_range.get("source", smoothed.get("source"))
            smoothed["inlier_ratio"] = table_range.get(
                "inlier_ratio",
                smoothed.get("inlier_ratio"),
            )
        else:
            smoothed = dict(table_range)
            for key in (
                "x_min",
                "x_max",
                "z_min",
                "z_max",
                "y_min",
                "y_max",
                "estimated_width_m",
                "estimated_length_m",
                "center_x",
                "center_z",
                "radius_x",
                "radius_z",
                "footprint_fill_ratio",
                "footprint_corner_ratio",
                "inlier_ratio",
            ):
                if key in table_range and key in previous:
                    smoothed[key] = (
                        (1.0 - alpha) * float(previous[key])
                        + alpha * float(table_range[key])
                    )
            if "center" in table_range and "center" in previous:
                smoothed["center"] = (
                    (1.0 - alpha) * np.asarray(previous["center"], dtype=np.float64)
                    + alpha * np.asarray(table_range["center"], dtype=np.float64)
                )

        self.smoothed_table_range = smoothed
        return smoothed

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
        """Andrew 单调链凸包，用于判断桌面 footprint 更像矩形还是圆形。"""
        if points.shape[0] <= 1:
            return points
        unique = np.unique(points, axis=0)
        if unique.shape[0] <= 1:
            return unique
        order = np.lexsort((unique[:, 1], unique[:, 0]))
        sorted_points = unique[order]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for point in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper = []
        for point in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _resolve_table_footprint(
        self,
        table_points: np.ndarray,
        width: float,
        length: float,
    ) -> Tuple[str, float, float]:
        """根据平面内点云填充率自动判断方桌/圆桌，或使用配置强制类型。"""
        if self.table_footprint_mode == "by_table":
            footprint = self.table_footprint_by_table.get(int(self.current_table), "rectangle")
            return footprint, 0.0, 0.0
        if self.table_footprint_mode != "auto":
            return self.table_footprint_mode, 0.0, 0.0

        bbox_area = max(width * length, 1e-9)
        hull = self._convex_hull_2d(table_points[:, [0, 2]])
        fill_ratio = self._polygon_area(hull) / bbox_area
        x_min = np.min(table_points[:, 0])
        x_max = np.max(table_points[:, 0])
        z_min = np.min(table_points[:, 2])
        z_max = np.max(table_points[:, 2])
        center_x = (x_min + x_max) / 2.0
        center_z = (z_min + z_max) / 2.0
        half_x = max((x_max - x_min) / 2.0, 1e-9)
        half_z = max((z_max - z_min) / 2.0, 1e-9)
        norm_x = np.abs((table_points[:, 0] - center_x) / half_x)
        norm_z = np.abs((table_points[:, 2] - center_z) / half_z)
        corner_ratio = float(np.mean((norm_x >= 0.7) & (norm_z >= 0.7)))
        footprint = (
            "ellipse"
            if fill_ratio <= self.round_table_fill_ratio_max
            or corner_ratio <= self.round_table_corner_ratio_max
            else "rectangle"
        )
        return footprint, fill_ratio, corner_ratio

    def _stabilize_footprint(self, footprint: str) -> str:
        """对 auto 判断出的圆/方桌类型做滞回，避免边界帧来回切换。"""
        if self.table_footprint_mode != "auto":
            self.locked_footprint = footprint
            self.pending_footprint = None
            self.pending_footprint_count = 0
            return footprint

        if self.locked_footprint is None:
            self.locked_footprint = footprint
            return footprint

        if footprint == self.locked_footprint:
            self.pending_footprint = None
            self.pending_footprint_count = 0
            return self.locked_footprint

        if footprint != self.pending_footprint:
            self.pending_footprint = footprint
            self.pending_footprint_count = 1
        else:
            self.pending_footprint_count += 1

        if self.pending_footprint_count >= self.footprint_switch_min_frames:
            self.locked_footprint = footprint
            self.pending_footprint = None
            self.pending_footprint_count = 0
        return self.locked_footprint

    def _make_table_range_from_points(
        self,
        table_points: np.ndarray,
        inlier_ratio: float,
    ) -> Optional[Dict]:
        """用桌面平面内点在 table 坐标系下的分布估计绿色过滤框范围。"""
        if table_points.shape[0] < self.plane_min_points:
            return None

        percentile = max(0.0, min(20.0, self.table_range_percentile))
        x_min, x_max = np.percentile(table_points[:, 0], [percentile, 100.0 - percentile])
        z_min, z_max = np.percentile(table_points[:, 2], [percentile, 100.0 - percentile])
        if not np.all(np.isfinite([x_min, x_max, z_min, z_max])):
            return None
        if x_max <= x_min or z_max <= z_min:
            return None

        center_x = (x_min + x_max) / 2.0
        center_z = (z_min + z_max) / 2.0
        estimated_width = x_max - x_min
        estimated_length = z_max - z_min
        in_range = (
            (table_points[:, 0] >= x_min)
            & (table_points[:, 0] <= x_max)
            & (table_points[:, 2] >= z_min)
            & (table_points[:, 2] <= z_max)
        )
        footprint_points = table_points[in_range]
        if footprint_points.shape[0] < self.plane_min_points:
            footprint_points = table_points
        footprint, fill_ratio, corner_ratio = self._resolve_table_footprint(
            footprint_points,
            estimated_width,
            estimated_length,
        )
        stabilized_footprint = self._stabilize_footprint(footprint)
        half_x = estimated_width * self.table_size_scale / 2.0
        half_z = estimated_length * self.table_size_scale / 2.0
        margin = max(0.0, self.table_range_margin_m)
        y_min = -self.plane_distance_thresh_m

        return {
            "x_min": center_x - half_x - margin,
            "x_max": center_x + half_x + margin,
            "z_min": center_z - half_z - margin,
            "z_max": center_z + half_z + margin,
            "y_min": y_min,
            "y_max": y_min + self.table_box_height_m,
            "center": np.array([center_x, 0.0, center_z], dtype=np.float64),
            "center_x": center_x,
            "center_z": center_z,
            "radius_x": half_x + margin,
            "radius_z": half_z + margin,
            "footprint": stabilized_footprint,
            "raw_footprint": footprint,
            "footprint_fill_ratio": fill_ratio,
            "footprint_corner_ratio": corner_ratio,
            "source": "plane_fit",
            "inlier_ratio": inlier_ratio,
            "estimated_width_m": estimated_width,
            "estimated_length_m": estimated_length,
        }

    def _fit_table_model_from_detection(
        self,
        depth_image: np.ndarray,
        detection: Detection,
        exclude_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> bool:
        """基于一个 Table 检测框拟合桌面坐标系和绿色过滤框。"""
        points, unmasked_points = self._depth_points_from_bbox_with_unmasked(
            depth_image,
            detection.bbox,
            exclude_bboxes=exclude_bboxes,
        )
        if points is None:
            return False

        fit_result = self._fit_plane_ransac(points)
        if fit_result is None:
            return False

        normal, center, inliers, inlier_ratio = fit_result
        orientation_points = inliers
        if self.plane_use_unmasked_points_for_orientation and exclude_bboxes:
            if unmasked_points is not None:
                unmasked_inliers = self._points_near_plane(unmasked_points, normal, center)
                if unmasked_inliers.shape[0] >= self.plane_min_points:
                    orientation_points = unmasked_inliers

        raw_rotation = self._table_rotation_from_plane_points(
            normal,
            center,
            orientation_points,
        )
        raw_translation = -raw_rotation @ center
        raw_rotation = self._align_rotation_to_previous(
            raw_rotation,
            self.smoothed_rotation_matrix,
        )
        if self._is_table_transform_outlier(raw_rotation, raw_translation):
            return False

        rotation, translation, reset_smoothing = self._smooth_table_transform(
            raw_rotation,
            raw_translation,
        )
        table_points = orientation_points @ rotation.T + translation
        table_range = self._make_table_range_from_points(table_points, inlier_ratio)
        if table_range is None:
            return False
        if self._is_table_range_outlier(table_range):
            return False
        table_range = self._smooth_table_range(table_range, reset_smoothing)

        self.rotation_matrix = rotation
        self.translation_vector = translation
        self.transform_matrix[:3, :3] = self.rotation_matrix
        self.transform_matrix[:3, 3] = self.translation_vector
        self.current_table_range = table_range
        self.table_model_source = "plane_fit"
        self.table_model_missing_frames = 0
        self.last_plane_fit_frame = self.frame_index
        return True

    def _use_smoothed_table_model(self, source: str) -> bool:
        """复用上一帧稳定桌面模型，避免 Table 漏检或降频更新时丢结果。"""
        if (
            self.smoothed_rotation_matrix is None
            or self.smoothed_translation_vector is None
            or self.smoothed_table_range is None
        ):
            return False

        self.rotation_matrix = self.smoothed_rotation_matrix
        self.translation_vector = self.smoothed_translation_vector
        self.transform_matrix[:3, :3] = self.rotation_matrix
        self.transform_matrix[:3, 3] = self.translation_vector
        self.current_table_range = dict(self.smoothed_table_range)
        self.current_table_range["source"] = source
        self.table_model_source = source
        return True

    def update_table_model(
        self,
        detections: List[Detection],
        depth_image: np.ndarray,
    ) -> None:
        """更新 table 坐标系：优先桌面平面拟合，失败则回退旧的固定角度估计。"""
        self.frame_index += 1
        self.current_table_range = None
        self.table_model_source = "fallback"

        if (
            self.plane_fit_enabled
            and self.last_plane_fit_frame >= 0
            and self.frame_index - self.last_plane_fit_frame < self.plane_update_interval_frames
            and self._use_smoothed_table_model("plane_fit_cached")
        ):
            return

        if self.plane_fit_enabled:
            exclude_bboxes = [
                detection.bbox
                for detection in detections
                if not self._is_table(detection)
            ]
            for detection in detections:
                if self._is_table(detection) and self._fit_table_model_from_detection(
                    depth_image,
                    detection,
                    exclude_bboxes=exclude_bboxes,
                ):
                    return

        if (
            self.table_model_hold_on_failure
            and self.table_model_missing_frames < self.table_model_max_hold_frames
            and self._use_smoothed_table_model("plane_fit_hold")
        ):
            self.table_model_missing_frames += 1
            return

        self.rotation_matrix = self.fallback_rotation_matrix.copy()
        self.smoothed_rotation_matrix = None
        self.smoothed_translation_vector = None
        self.smoothed_table_range = None
        self.table_model_missing_frames = 0
        table_center_rot = self.find_table_center(detections, depth_image)
        self.update_translation_matrix(table_center_rot)

    def get_object_3d_coordinates(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        *,
        allow_table_plane_fallback: bool = True,
    ) -> Optional[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """把检测框中心 + 估计深度反投影成相机坐标系下的 3D 点。

        深度取值优先用框内有效像素的中位数；深度空洞时依次尝试随机采样、
        中心邻域、外扩邻域和桌面平面交点。返回 (3D 点, 深度(米), 裁剪后的框)。
        """
        x1, y1, x2, y2 = bbox
        height, width = depth_image.shape[:2]
        x1 = max(0, min(width - 1, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height - 1, int(y1)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        depth_m = self._valid_depth_m_from_bbox(depth_image, (x1, y1, x2, y2))
        point_cam = None
        if depth_m is None:
            sample_count = max(1, self.coord_sample_num)
            sample_x = np.random.randint(x1, x2, size=sample_count, dtype=np.int32)
            sample_y = np.random.randint(y1, y2, size=sample_count, dtype=np.int32)
            sampled = depth_image[sample_y, sample_x]
            sampled = sampled[sampled > 0]
            if self.depth_trunc_m > 0:
                sampled = sampled[sampled <= self.depth_trunc_m * 1000.0]

            if sampled.size == 0:
                half = 5
                min_x = max(0, center_x - half)
                max_x = min(width, center_x + half + 1)
                min_y = max(0, center_y - half)
                max_y = min(height, center_y + half + 1)
                sampled = depth_image[min_y:max_y, min_x:max_x]
                sampled = sampled[sampled > 0]
                if self.depth_trunc_m > 0:
                    sampled = sampled[sampled <= self.depth_trunc_m * 1000.0]

            if sampled.size > 0:
                depth_m = float(np.median(sampled)) / 1000.0

        if depth_m is None:
            depth_m = self._expanded_depth_m(depth_image, (x1, y1, x2, y2))

        if depth_m is None and allow_table_plane_fallback:
            intersection = self._table_plane_intersection_cam(center_x, center_y)
            if intersection is not None:
                point_cam, depth_m = intersection

        if depth_m is None:
            return None

        if point_cam is None:
            x = (center_x - self.cx) * depth_m / self.fx
            y = (center_y - self.cy) * depth_m / self.fy
            point_cam = np.array([x, y, depth_m], dtype=np.float64)
        return (
            point_cam,
            depth_m,
            (x1, y1, x2, y2),
        )

    def get_detection_filter_coordinates(
        self,
        depth_image: np.ndarray,
        detection: Detection,
    ) -> Optional[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """返回用于桌面 footprint 判断的 3D 点；物品优先使用桌面支撑点。"""
        height, width = depth_image.shape[:2]
        clipped = self._clip_bbox(detection.bbox, width, height)
        if clipped is None:
            return None

        if not self._is_table(detection) and self.use_support_point:
            x1, _y1, x2, y2 = clipped
            support_x = (x1 + x2) // 2
            support_y = max(0, y2 - 1)
            intersection = self._table_plane_intersection_cam(support_x, support_y)
            if intersection is not None:
                point_cam, depth_m = intersection
                grounded = self._support_point_on_ground(
                    depth_image, clipped, depth_m
                )
                if grounded is not None:
                    # 脚点真实深度明显落在桌面平面之后：物体站在地面而非桌面，
                    # 用真实脚点参与过滤（其桌面 y 会落在平面之下而被剔除）。
                    return grounded, float(grounded[2]), clipped
                return point_cam, depth_m, clipped

        return self.get_object_3d_coordinates(
            depth_image,
            clipped,
            allow_table_plane_fallback=True,
        )

    def _support_point_on_ground(
        self,
        depth_image: np.ndarray,
        clipped: Tuple[int, int, int, int],
        plane_depth_m: float,
    ) -> Optional[np.ndarray]:
        """脚点真实深度明显落在桌面平面之后时，返回真实脚点的相机 3D 坐标。

        用于区分"站在桌面上的高叠物品"（脚点贴合平面，深度差≈0）与"紧邻桌子
        的地面物品"（脚点在延伸平面之后，深度明显更大）。阈值 <=0 时禁用。

        为兼容斜放/不规则物体（bbox 底边中心可能落在空角、透出背景），这里在
        bbox 底部横带内取深度，用较近分位数代表脚点——物体本体一定比其后方背景
        更近，取近端可避开落空读到的背景假远值。
        """
        if self.support_point_ground_reject_gap_m <= 0.0:
            return None

        x1, y1, x2, y2 = clipped
        height, width = depth_image.shape[:2]
        box_height = max(1, y2 - y1)
        # 底部横带：取 bbox 下缘一段高度（至少几像素）。
        band_h = max(3, int(round(box_height * self.support_point_band_ratio)))
        band_y1 = max(0, min(height - 1, y2 - band_h))
        band_y2 = min(height, y2)
        band_x1 = max(0, x1)
        band_x2 = min(width, x2)
        if band_y2 <= band_y1 or band_x2 <= band_x1:
            return None

        region = depth_image[band_y1:band_y2, band_x1:band_x2]
        valid = region[region > 0]
        if self.depth_trunc_m > 0:
            valid = valid[valid <= self.depth_trunc_m * 1000.0]
        if valid.size < self.object_depth_min_valid_pixels:
            return None

        # 用较近分位数代表脚点深度，抑制透出背景的远端噪声。
        real_depth_m = float(np.percentile(valid, self.support_point_near_percentile)) / 1000.0
        gap = real_depth_m - plane_depth_m
        if gap <= self.support_point_ground_reject_gap_m:
            return None
        # 合理性上限：过大的 gap 多半是脚点深度读取失败（落空到远处背景），
        # 不足以据此判为地面物品，避免误杀桌面物品。
        if (
            self.support_point_ground_reject_max_gap_m > 0.0
            and gap > self.support_point_ground_reject_max_gap_m
        ):
            return None

        support_x = (band_x1 + band_x2) // 2
        support_y = max(0, band_y2 - 1)
        ray = np.array(
            [
                (float(support_x) - self.cx) / self.fx,
                (float(support_y) - self.cy) / self.fy,
                1.0,
            ],
            dtype=np.float64,
        )
        return (ray * real_depth_m).astype(np.float64)

    def find_table_center(
        self,
        detections: List[Detection],
        depth_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """在检测中找到桌子，返回旋转后（尚未平移）的桌面中心 3D 坐标。"""
        for detection in detections:
            if not self._is_table(detection):
                continue
            result = self.get_object_3d_coordinates(
                depth_image,
                detection.bbox,
                allow_table_plane_fallback=False,
            )
            if result is not None:
                center_raw, _, _ = result
                return self.rotation_matrix @ center_raw
        return None

    def update_translation_matrix(self, table_center_rot: Optional[np.ndarray]) -> None:
        """以桌面中心为原点更新平移向量，使后续 3D 点都落在桌面坐标系里。"""
        if table_center_rot is not None:
            self.translation_vector = -np.asarray(table_center_rot, dtype=np.float64)
        self.transform_matrix[:3, :3] = self.rotation_matrix
        self.transform_matrix[:3, 3] = self.translation_vector

    @staticmethod
    def _expand_bbox(
        bbox: Tuple[int, int, int, int],
        ratio: float,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        pad_x = int(round(box_width * ratio))
        pad_y = int(round(box_height * ratio))
        return DepthFilter._clip_bbox(
            (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
            width,
            height,
        )

    def _valid_depth_m_from_bbox(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        *,
        min_count: Optional[int] = None,
    ) -> Optional[float]:
        height, width = depth_image.shape[:2]
        clipped = self._clip_bbox(bbox, width, height)
        if clipped is None:
            return None

        x1, y1, x2, y2 = clipped
        region = depth_image[y1:y2, x1:x2]
        valid = region[region > 0]
        if self.depth_trunc_m > 0:
            valid = valid[valid <= self.depth_trunc_m * 1000.0]
        required = self.object_depth_min_valid_pixels if min_count is None else min_count
        if valid.size < required:
            return None
        return float(np.median(valid)) / 1000.0

    def _expanded_depth_m(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        height, width = depth_image.shape[:2]
        for ratio in self.object_depth_expand_ratios:
            expanded = self._expand_bbox(bbox, ratio, width, height)
            if expanded is None or expanded == bbox:
                continue
            depth_m = self._valid_depth_m_from_bbox(depth_image, expanded)
            if depth_m is not None:
                return depth_m
        return None

    def _table_plane_intersection_cam(
        self,
        center_x: int,
        center_y: int,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """用 bbox 中心像素射线与当前桌面平面求交，兜底深度空洞物体。"""
        if not self.object_depth_table_plane_fallback:
            return None
        if self.current_table_range is None:
            return None
        if not str(self.current_table_range.get("source", "")).startswith("plane_fit"):
            return None

        ray = np.array(
            [
                (float(center_x) - self.cx) / self.fx,
                (float(center_y) - self.cy) / self.fy,
                1.0,
            ],
            dtype=np.float64,
        )
        table_y_axis = self.rotation_matrix[1]
        denominator = float(table_y_axis @ ray)
        if abs(denominator) < 1e-8:
            return None

        target_y = self.object_depth_plane_offset_m
        depth_m = (target_y - float(self.translation_vector[1])) / denominator
        if not np.isfinite(depth_m) or depth_m <= 0:
            return None
        if self.depth_trunc_m > 0 and depth_m > self.depth_trunc_m:
            return None

        point_cam = ray * depth_m
        return point_cam.astype(np.float64), float(depth_m)

    def get_table_3d_range(
        self,
        depth_image: np.ndarray,
        table_bbox: Tuple[int, int, int, int],
    ) -> Optional[Dict]:
        """根据桌子框估算桌面在桌面坐标系下的 3D 范围（含少量边缘余量）。"""
        if self.current_table_range is not None:
            return self.current_table_range

        table_result = self.get_object_3d_coordinates(
            depth_image,
            table_bbox,
            allow_table_plane_fallback=False,
        )
        if table_result is None:
            return None

        table_center_raw, table_depth, table_2d = table_result
        x1, y1, x2, y2 = table_2d
        table_center = (
            self.rotation_matrix @ table_center_raw + self.translation_vector
        )
        table_width_3d = (
            (x2 - x1) * table_depth / self.fx
        ) * self.dynamic_scale
        table_length_3d = (
            (y2 - y1) * table_depth / self.fy
        ) * self.dynamic_scale

        return {
            "x_min": table_center[0] - table_width_3d / 2.0 - 0.005,
            "x_max": table_center[0] + table_width_3d / 2.0 + 0.005,
            "z_min": table_center[2] - table_length_3d / 2.0 - 0.01,
            "z_max": table_center[2] + table_length_3d / 2.0 + 0.015,
            "y_min": table_center[1] - 0.007,
            "y_max": table_center[1] - 0.007 + 0.05,
            "center": table_center,
            "center_x": table_center[0],
            "center_z": table_center[2],
            "radius_x": table_width_3d / 2.0 + 0.005,
            "radius_z": table_length_3d / 2.0 + 0.015,
            "footprint": "rectangle",
            "source": "fallback",
        }

    @staticmethod
    def create_table_bounding_box(table_range: Dict):
        import open3d as o3d

        if table_range.get("footprint") == "ellipse":
            segments = 64
            center_x = float(table_range["center_x"])
            center_z = float(table_range["center_z"])
            radius_x = max(1e-6, float(table_range["radius_x"]))
            radius_z = max(1e-6, float(table_range["radius_z"]))
            y_min = float(table_range["y_min"])
            y_max = float(table_range.get("y_max", y_min + 0.05))
            angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
            lower = np.column_stack(
                (
                    center_x + radius_x * np.cos(angles),
                    np.full(segments, y_min),
                    center_z + radius_z * np.sin(angles),
                )
            )
            upper = lower.copy()
            upper[:, 1] = y_max
            points = np.vstack((lower, upper))
            lines = []
            for index in range(segments):
                next_index = (index + 1) % segments
                lines.append([index, next_index])
                lines.append([segments + index, segments + next_index])
            for index in range(0, segments, max(1, segments // 8)):
                lines.append([index, segments + index])
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
            )
            line_set.paint_uniform_color([0.0, 1.0, 0.0])
            return line_set

        bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[
                table_range["x_min"],
                table_range["y_min"],
                table_range["z_min"],
            ],
            max_bound=[
                table_range["x_max"],
                table_range.get("y_max", table_range["y_min"] + 0.05),
                table_range["z_max"],
            ],
        )
        box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bounds)
        box.paint_uniform_color([0.0, 1.0, 0.0])
        return box

    def filter_objects_on_table_cached(
        self,
        detections: List[Detection],
        centers_world: np.ndarray,
        depth_image: np.ndarray,
    ) -> List[Detection]:
        """只保留 3D 中心落在桌面范围内的检测；可选保留桌子本身。"""
        table_detections = [
            detection for detection in detections if self._is_table(detection)
        ]
        if not table_detections and self.current_table_range is None:
            return []

        if table_detections:
            table_range = self.get_table_3d_range(
                depth_image,
                table_detections[0].bbox,
            )
        else:
            table_range = self.current_table_range
        if table_range is None:
            return []

        if self.visualizer is not None:
            self.table_bbox = self.create_table_bounding_box(table_range)
            self._refresh_geometry(
                "table_range", self.table_bbox, self._line_material
            )

        x_min = table_range["x_min"] - 0.02
        x_max = table_range["x_max"] + 0.02
        z_min = table_range["z_min"] - 0.02
        z_max = table_range["z_max"] + 0.02
        y_min = table_range["y_min"]
        footprint = table_range.get("footprint", "rectangle")
        center_x = float(table_range.get("center_x", (x_min + x_max) / 2.0))
        center_z = float(table_range.get("center_z", (z_min + z_max) / 2.0))
        radius_x = max(1e-6, float(table_range.get("radius_x", (x_max - x_min) / 2.0)))
        radius_z = max(1e-6, float(table_range.get("radius_z", (z_max - z_min) / 2.0)))

        filtered: List[Detection] = []
        fp_params = {
            "x_min": x_min, "x_max": x_max, "z_min": z_min, "z_max": z_max,
            "y_min": y_min, "footprint": footprint,
            "center_x": center_x, "center_z": center_z,
            "radius_x": radius_x, "radius_z": radius_z,
        }
        for detection, center in zip(detections, centers_world):
            if self._is_table(detection):
                if self.keep_table:
                    filtered.append(detection)
                continue

            keep = self._center_in_footprint(center, fp_params)
            # 方案 A：support_point 交点对高/边缘物体会横向漂移出桌面。
            # 若主判定落到桌外，但物体框内有真实深度，则用真实深度反投影的
            # (x,z) 复核——竖直物体真实中心的水平位置≈落脚点，不受射线角度影响。
            # 仅对高物体复核（矮物体漂移小），避免误救桌外物品。
            if not keep and self.support_point_reproject_recover:
                fallback_center = self._real_depth_center_world(depth_image, detection.bbox)
                if fallback_center is not None:
                    height_above = float(fallback_center[1]) - float(y_min)
                    if height_above >= self.support_point_reproject_min_height_m:
                        keep = self._center_in_footprint(fallback_center, fp_params)
            if keep:
                filtered.append(detection)

        return filtered

    def _center_in_footprint(self, center: np.ndarray, params: dict) -> bool:
        """判断一个 3D 中心是否落在桌面 footprint 内（含垂直扩张与高度约束）。"""
        extra_margin = self._vertical_footprint_extra_margin(center, params["y_min"])
        local_x_min = params["x_min"] - extra_margin
        local_x_max = params["x_max"] + extra_margin
        local_z_min = params["z_min"] - extra_margin
        local_z_max = params["z_max"] + extra_margin
        if params["footprint"] == "ellipse":
            norm_x = (center[0] - params["center_x"]) / (params["radius_x"] + 0.02 + extra_margin)
            norm_z = (center[2] - params["center_z"]) / (params["radius_z"] + 0.02 + extra_margin)
            in_footprint = norm_x * norm_x + norm_z * norm_z <= 1.0
        else:
            in_x = local_x_min <= center[0] <= local_x_max
            in_z = local_z_min <= center[2] <= local_z_max
            in_footprint = in_x and in_z
        above_y = center[1] >= params["y_min"]
        return bool(in_footprint and above_y)

    def _real_depth_center_world(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """用物体框内真实深度反投影出 3D 中心（桌面坐标系），用于复核落脚位置。"""
        result = self.get_object_3d_coordinates(
            depth_image, bbox, allow_table_plane_fallback=False
        )
        if result is None:
            return None
        center_cam, _, _ = result
        return self.rotation_matrix @ center_cam + self.translation_vector

    def _vertical_footprint_extra_margin(self, center: np.ndarray, y_min: float) -> float:
        """按 table-y 高度把桌面 footprint 向外扩成倒置梯形/锥台。"""
        if not self.footprint_vertical_expansion_enabled:
            return 0.0
        if self.footprint_vertical_expansion_tan <= 0.0:
            return 0.0
        height = float(center[1]) - float(y_min) - self.footprint_vertical_expansion_min_height_m
        if height <= 0.0:
            return 0.0
        extra_margin = height * self.footprint_vertical_expansion_tan
        if self.footprint_vertical_expansion_max_extra_margin_m > 0.0:
            extra_margin = min(extra_margin, self.footprint_vertical_expansion_max_extra_margin_m)
        return max(0.0, extra_margin)

    def _bbox_depth_m(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        """为 2D 检测框估计一个代表深度，用于把框角点投影到 3D。"""
        height, width = depth_image.shape[:2]
        clipped = self._clip_bbox(bbox, width, height)
        if clipped is None:
            return None
        x1, y1, x2, y2 = clipped

        if self.detection_box_depth_mode == "center":
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            depth_value = float(depth_image[center_y, center_x])
            if depth_value > 0:
                return depth_value / 1000.0

        depth_m = self._valid_depth_m_from_bbox(
            depth_image,
            (x1, y1, x2, y2),
            min_count=1,
        )
        if depth_m is not None:
            return depth_m

        depth_m = self._expanded_depth_m(depth_image, (x1, y1, x2, y2))
        if depth_m is not None:
            return depth_m

        intersection = self._table_plane_intersection_cam(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )
        if intersection is None:
            return None
        _point_cam, depth_m = intersection
        return depth_m

    def _bbox_corners_world(
        self,
        depth_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """把 2D bbox 四个角按代表深度反投影，并转换到 table 坐标系。"""
        height, width = depth_image.shape[:2]
        clipped = self._clip_bbox(bbox, width, height)
        if clipped is None:
            return None

        depth_m = self._bbox_depth_m(depth_image, clipped)
        if depth_m is None:
            return None

        x1, y1, x2, y2 = clipped
        # x2/y2 是半开区间边界，显示角点时收回到图像内最后一个像素。
        x2f = max(x1, x2 - 1)
        y2f = max(y1, y2 - 1)
        pixels = np.array(
            [
                [x1, y1],
                [x2f, y1],
                [x2f, y2f],
                [x1, y2f],
            ],
            dtype=np.float64,
        )
        x = (pixels[:, 0] - self.cx) * depth_m / self.fx
        y = (pixels[:, 1] - self.cy) * depth_m / self.fy
        z = np.full(4, depth_m, dtype=np.float64)
        corners_cam = np.column_stack((x, y, z))
        return corners_cam @ self.rotation_matrix.T + self.translation_vector

    def _init_o3d_visualizer(self) -> None:
        """初始化 O3DVisualizer（支持真实线宽、3D 文字标签与工具栏暂停按钮）。"""
        import open3d as o3d
        from open3d.visualization import gui, rendering

        self._o3d = o3d
        self._gui = gui
        self._rendering = rendering

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.intrinsic_config["width"],
            height=self.intrinsic_config["height"],
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )

        gui.Application.instance.initialize()
        # 自建窗口：一个铺满的 3D 视图 + 顶部两个小按钮，只保留暂停/步进。
        self.window = gui.Application.instance.create_window(
            "Depth Filter Point Cloud", self.window_width, self.window_height
        )
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.visualizer = self.scene_widget  # 几何操作走 SceneWidget.scene

        # 材质：点云、标记点、粗线框。
        self._point_material = rendering.MaterialRecord()
        self._point_material.shader = "defaultUnlit"
        self._point_material.point_size = self.viz_point_size

        self._marker_material = rendering.MaterialRecord()
        self._marker_material.shader = "defaultUnlit"
        self._marker_material.point_size = self.viz_point_size * 3.0

        self._line_material = rendering.MaterialRecord()
        self._line_material.shader = "unlitLine"
        self._line_material.line_width = self.viz_line_width

        # 几何体容器。
        self.point_cloud = o3d.geometry.PointCloud()
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[-0.1, 0.0, -0.1]
        )
        self.markers_pcd = o3d.geometry.PointCloud()
        self.detection_boxes = o3d.geometry.LineSet()
        self.link_lines = o3d.geometry.LineSet()

        # 暂停/步进逻辑。
        def _toggle_pause():
            self._viz_paused = not self._viz_paused
            self._pause_button.text = "Resume" if self._viz_paused else "Pause"
            print(
                "[DepthFilter viz] "
                + ("PAUSED" if self._viz_paused else "RESUMED"),
                flush=True,
            )

        def _step():
            if self._viz_paused:
                self._viz_step = True

        # 顶部按钮条。
        em = self.window.theme.font_size
        self._pause_button = gui.Button("Pause")
        self._pause_button.set_on_clicked(_toggle_pause)
        step_button = gui.Button("Step")
        step_button.set_on_clicked(_step)
        button_bar = gui.Horiz(0.25 * em, gui.Margins(0.5 * em, 0.25 * em, 0.5 * em, 0.25 * em))
        button_bar.add_child(self._pause_button)
        button_bar.add_child(step_button)
        button_bar.add_stretch()

        self.window.add_child(self.scene_widget)
        self.window.add_child(button_bar)

        def _on_layout(ctx):
            rect = self.window.content_rect
            self.scene_widget.frame = rect
            pref = button_bar.calc_preferred_size(ctx, gui.Widget.Constraints())
            button_bar.frame = gui.Rect(
                rect.x, rect.y, rect.width, pref.height
            )

        self.window.set_on_layout(_on_layout)

        # 快捷键：空格暂停/继续，N/右方向键单步。
        def _on_key(event):
            if event.type == gui.KeyEvent.DOWN:
                if event.key == gui.KeyName.SPACE:
                    _toggle_pause()
                    return gui.Widget.EventCallbackResult.HANDLED
                if event.key in (gui.KeyName.N, gui.KeyName.RIGHT):
                    _step()
                    return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED

        self.scene_widget.set_on_key(_on_key)

    def _pump_gui(self) -> None:
        """驱动一次 GUI 事件循环。"""
        if self._gui is not None:
            self._gui.Application.instance.run_one_tick()

    def _refresh_geometry(self, name: str, geometry, material) -> None:
        """按名添加/更新几何体（先 remove 再 add 才能刷新）。"""
        scene = self.scene_widget.scene
        if name in self._geom_added:
            scene.remove_geometry(name)
        scene.add_geometry(name, geometry, material)
        self._geom_added.add(name)

    def _update_detection_box_visualization(
        self,
        depth_image: np.ndarray,
        detections: List[Detection],
    ) -> list:
        """构建每个检测的 3D 框线框，并返回标签列表 [(位置, 文本, 颜色)]。"""
        if self.detection_boxes is None:
            return []
        import open3d as o3d

        points = []
        lines = []
        colors = []
        labels = []
        for detection in detections:
            corners = self._bbox_corners_world(depth_image, detection.bbox)
            if corners is None:
                continue

            base_index = len(points)
            points.extend(corners.tolist())
            lines.extend(
                [
                    [base_index + 0, base_index + 1],
                    [base_index + 1, base_index + 2],
                    [base_index + 2, base_index + 3],
                    [base_index + 3, base_index + 0],
                    [base_index + 0, base_index + 2],
                ]
            )
            color = [0.0, 1.0, 1.0] if self._is_table(detection) else [1.0, 1.0, 0.0]
            colors.extend([color] * 5)
            # 标签放在框的上边中点（前两个角的中点略微上抬）。
            corners_arr = np.asarray(corners, dtype=np.float64)
            label_pos = corners_arr[:2].mean(axis=0)
            labels.append((label_pos, detection.class_name, color))

        self.detection_boxes.points = o3d.utility.Vector3dVector(
            np.asarray(points, dtype=np.float64).reshape((-1, 3))
            if points
            else np.zeros((0, 3), dtype=np.float64)
        )
        self.detection_boxes.lines = o3d.utility.Vector2iVector(
            np.asarray(lines, dtype=np.int32).reshape((-1, 2))
            if lines
            else np.zeros((0, 2), dtype=np.int32)
        )
        self.detection_boxes.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=np.float64).reshape((-1, 3))
            if colors
            else np.zeros((0, 3), dtype=np.float64)
        )
        return labels

    def update_open3d_visualization(
        self,
        depth_image: np.ndarray,
        centers_world: np.ndarray,
        detections: List[Detection],
        kept: Optional[List[Detection]] = None,
        rgb_image: Optional[np.ndarray] = None,
    ) -> None:
        """（可选）刷新 open3d 点云与目标标记；未开启可视化时直接返回。"""
        if self.visualizer is None:
            return
        import open3d as o3d

        stride = max(1, int(self.point_cloud_stride))
        depth_sub = np.ascontiguousarray(depth_image[::stride, ::stride].astype(np.uint16))
        sub_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=depth_sub.shape[1],
            height=depth_sub.shape[0],
            fx=self.fx / stride,
            fy=self.fy / stride,
            cx=self.cx / stride,
            cy=self.cy / stride,
        )
        o3d_depth = o3d.geometry.Image(depth_sub)

        if rgb_image is not None:
            # frame.rgb 已是 RGB 顺序，直接用于点云上色。
            rgb_sub = np.ascontiguousarray(rgb_image[::stride, ::stride, :3].astype(np.uint8))
            if rgb_sub.shape[:2] == depth_sub.shape[:2]:
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb_sub),
                    o3d_depth,
                    depth_scale=1000.0,
                    depth_trunc=self.depth_trunc_m,
                    convert_rgb_to_intensity=False,
                )
                point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd, sub_intrinsic
                )
            else:
                point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                    o3d_depth, sub_intrinsic, depth_scale=1000.0,
                    depth_trunc=self.depth_trunc_m,
                )
        else:
            point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                o3d_depth, sub_intrinsic, depth_scale=1000.0,
                depth_trunc=self.depth_trunc_m,
            )
        point_cloud.transform(self.transform_matrix)
        self.point_cloud.points = point_cloud.points
        self.point_cloud.colors = point_cloud.colors

        if centers_world.shape[0] > 0:
            self.markers_pcd.points = o3d.utility.Vector3dVector(centers_world)
            if self.viz_color_by_keep and kept is not None:
                # 绿=保留（在桌面上），红=被过滤（判为桌外/地面）。
                kept_ids = {id(det) for det in kept}
                marker_colors = np.array(
                    [
                        [0.0, 1.0, 0.0] if id(det) in kept_ids else [1.0, 0.0, 0.0]
                        for det in detections
                    ],
                    dtype=np.float64,
                ).reshape((-1, 3))
            else:
                marker_colors = np.tile(
                    np.array([[1.0, 0.0, 0.0]]),
                    (centers_world.shape[0], 1),
                )
            self.markers_pcd.colors = o3d.utility.Vector3dVector(marker_colors)
        else:
            empty = np.zeros((0, 3))
            self.markers_pcd.points = o3d.utility.Vector3dVector(empty)
            self.markers_pcd.colors = o3d.utility.Vector3dVector(empty)

        # bbox 中心 ↔ 对应 footprint point 的连线。
        link_points = []
        link_lines = []
        if self.link_lines is not None and centers_world.shape[0] > 0:
            for idx, detection in enumerate(detections):
                if self._is_table(detection):
                    continue
                corners = self._bbox_corners_world(depth_image, detection.bbox)
                if corners is None:
                    continue
                box_center = np.asarray(corners, dtype=np.float64).mean(axis=0)
                base = len(link_points)
                link_points.append(box_center.tolist())
                link_points.append(centers_world[idx].tolist())
                link_lines.append([base, base + 1])
        if self.link_lines is not None:
            self.link_lines.points = o3d.utility.Vector3dVector(
                np.asarray(link_points, dtype=np.float64).reshape((-1, 3))
                if link_points
                else np.zeros((0, 3), dtype=np.float64)
            )
            self.link_lines.lines = o3d.utility.Vector2iVector(
                np.asarray(link_lines, dtype=np.int32).reshape((-1, 2))
                if link_lines
                else np.zeros((0, 2), dtype=np.int32)
            )
            if link_lines:
                # 连线用醒目的洋红色。
                self.link_lines.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[1.0, 0.0, 1.0]]), (len(link_lines), 1))
                )

        labels = []
        if self.show_detection_boxes:
            labels = self._update_detection_box_visualization(depth_image, detections)
        elif self.detection_boxes is not None:
            empty_points = np.zeros((0, 3), dtype=np.float64)
            empty_lines = np.zeros((0, 2), dtype=np.int32)
            self.detection_boxes.points = o3d.utility.Vector3dVector(empty_points)
            self.detection_boxes.lines = o3d.utility.Vector2iVector(empty_lines)
            self.detection_boxes.colors = o3d.utility.Vector3dVector(empty_points)

        # 按名刷新几何体。
        self._refresh_geometry("point_cloud", self.point_cloud, self._point_material)
        self._refresh_geometry("coord_frame", self.coordinate_frame, self._point_material)
        self._refresh_geometry("markers", self.markers_pcd, self._marker_material)
        if self.detection_boxes is not None and len(self.detection_boxes.lines) > 0:
            self._refresh_geometry("det_boxes", self.detection_boxes, self._line_material)
        elif "det_boxes" in self._geom_added:
            self.scene_widget.scene.remove_geometry("det_boxes")
            self._geom_added.discard("det_boxes")
        if self.link_lines is not None and len(self.link_lines.lines) > 0:
            self._refresh_geometry("link_lines", self.link_lines, self._line_material)
        elif "link_lines" in self._geom_added:
            self.scene_widget.scene.remove_geometry("link_lines")
            self._geom_added.discard("link_lines")

        # 3D 文字标签：每帧清空重建。
        for handle in self._labels:
            self.scene_widget.remove_3d_label(handle)
        self._labels = []
        if self.viz_show_labels:
            for position, text, _color in labels:
                self._labels.append(
                    self.scene_widget.add_3d_label(position, str(text))
                )

        if self.first_frame:
            bounds = self.scene_widget.scene.bounding_box
            self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
            self.first_frame = False
        self.window.post_redraw()

        self._pump_gui()
        # 暂停态：持续泵 GUI 事件（保持可旋转/缩放/按按钮），直到继续或单步。
        # 暂停期间通过 pause_clock 冻结轮次计时，避免观察过久触发计时退出。
        if self._viz_paused and not self._viz_step:
            pause_clock.begin_pause()
            try:
                while self._viz_paused and not self._viz_step:
                    self._pump_gui()
                    time.sleep(0.01)
            finally:
                pause_clock.end_pause()
        self._viz_step = False

    def process(
        self,
        detections: List[Detection],
        frame: Frame,
        _table: int,
    ) -> List[Detection]:
        """过滤主流程：定位桌面 → 计算各检测 3D 中心 → 保留桌面上的目标。"""
        depth_image = frame.depth
        if depth_image is None:
            return detections

        self.current_table = int(_table)
        self.update_table_model(detections, depth_image)

        valid_detections: List[Detection] = []
        centers_world = []
        for detection in detections:
            result = self.get_detection_filter_coordinates(
                depth_image,
                detection,
            )
            if result is None:
                continue
            center_cam, _, _ = result
            center_world = (
                self.rotation_matrix @ center_cam + self.translation_vector
            )
            valid_detections.append(detection)
            centers_world.append(center_world)

        if centers_world:
            centers_array = np.asarray(centers_world, dtype=np.float64)
        else:
            centers_array = np.zeros((0, 3), dtype=np.float64)

        kept = self.filter_objects_on_table_cached(
            valid_detections,
            centers_array,
            depth_image,
        )
        self.update_open3d_visualization(
            depth_image, centers_array, valid_detections, kept, frame.rgb
        )
        return kept

    def close(self) -> None:
        """关闭可视化窗口，释放 open3d 资源。"""
        if self.window is not None:
            try:
                self.window.close()
            except Exception:
                pass
            self.window = None
            self.scene_widget = None
            self.visualizer = None
