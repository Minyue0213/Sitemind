"""Short-horizon temporal memory for aligned geometry and semantic BEV layers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np

from sitemind.fusion.projection import SemanticBev, fuse_semantic_bev_with_geometry
from sitemind.fusion.alignment import BevGridSpec, warp_bev_nearest


@dataclass(frozen=True)
class TemporalBev:
    """Conservative aggregate of a trailing window of aligned BEV frames."""

    geometry_risk: np.ndarray
    geometry_valid: np.ndarray
    semantic_risk: np.ndarray
    semantic_uncertainty: np.ndarray
    semantic_observed: np.ndarray
    fused_risk: np.ndarray
    fused_valid: np.ndarray
    history_size: int


class TemporalBevMemory:
    """Aggregate recent observations in each aligned BEV cell.

    When ``world_from_frame`` is supplied, every stored local map is transformed
    into the current vehicle frame before aggregation. Geometry uses the median
    of recent valid measurements to suppress one-frame slope/step noise;
    semantic risk uses the worst recent observation so hazards do not disappear
    immediately.
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._frames: deque[tuple[tuple[np.ndarray, ...], BevGridSpec, np.ndarray]] = deque(
            maxlen=window_size
        )
        self._shape: tuple[int, int] | None = None
        self._pose_aware: bool | None = None

    def update(
        self,
        geometry_risk: np.ndarray,
        geometry_valid: np.ndarray,
        semantic_risk: np.ndarray,
        semantic_uncertainty: np.ndarray,
        semantic_observed: np.ndarray,
        *,
        world_from_frame: np.ndarray | None = None,
        x_min_m: float = 0.0,
        y_min_m: float = 0.0,
        resolution_m: float = 1.0,
    ) -> TemporalBev:
        geometry = np.asarray(geometry_risk, dtype=np.float32)
        geometry_mask = np.asarray(geometry_valid, dtype=bool)
        semantic = np.asarray(semantic_risk, dtype=np.float32)
        uncertainty = np.asarray(semantic_uncertainty, dtype=np.float32)
        semantic_mask = np.asarray(semantic_observed, dtype=bool)
        arrays = (geometry, geometry_mask, semantic, uncertainty, semantic_mask)
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("all BEV layers must be 2D")
        if any(array.shape != geometry.shape for array in arrays[1:]):
            raise ValueError("all BEV layers must have the same shape")
        if self._shape is None:
            self._shape = geometry.shape
        elif geometry.shape != self._shape:
            raise ValueError("BEV shape changed; align frames before temporal fusion")
        pose_aware = world_from_frame is not None
        if self._pose_aware is None:
            self._pose_aware = pose_aware
        elif self._pose_aware != pose_aware:
            raise ValueError("cannot mix pose-aware and pre-aligned frames in one memory")
        pose = np.eye(4, dtype=np.float64) if world_from_frame is None else world_from_frame
        grid = BevGridSpec(geometry.shape, x_min_m, y_min_m, resolution_m)
        self._frames.append((tuple(array.copy() for array in arrays), grid, np.asarray(pose)))

        aligned_frames: list[tuple[np.ndarray, ...]] = []
        for frame_arrays, frame_grid, frame_pose in self._frames:
            if not pose_aware:
                aligned_frames.append(frame_arrays)
                continue
            aligned_geometry, aligned_geometry_valid = warp_bev_nearest(
                frame_arrays[0], frame_arrays[1], frame_grid, grid, frame_pose, pose
            )
            aligned_semantic, aligned_semantic_observed = warp_bev_nearest(
                frame_arrays[2], frame_arrays[4], frame_grid, grid, frame_pose, pose
            )
            aligned_uncertainty, _ = warp_bev_nearest(
                frame_arrays[3], frame_arrays[4], frame_grid, grid, frame_pose, pose
            )
            aligned_frames.append(
                (
                    aligned_geometry,
                    aligned_geometry_valid,
                    aligned_semantic,
                    aligned_uncertainty,
                    aligned_semantic_observed,
                )
            )

        geometry_valid_stack = np.stack([frame[1] for frame in aligned_frames])
        semantic_observed_stack = np.stack([frame[4] for frame in aligned_frames])
        temporal_geometry_valid = np.any(geometry_valid_stack, axis=0)
        temporal_semantic_observed = np.any(semantic_observed_stack, axis=0)
        geometry_stack = np.stack(
            [np.where(frame[1], frame[0], np.nan) for frame in aligned_frames]
        )
        semantic_stack = np.stack(
            [np.where(frame[4], frame[2], -np.inf) for frame in aligned_frames]
        )
        uncertainty_stack = np.stack(
            [np.where(frame[4], frame[3], -np.inf) for frame in aligned_frames]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temporal_geometry = np.nanmedian(geometry_stack, axis=0)
        temporal_semantic = np.max(semantic_stack, axis=0)
        temporal_uncertainty = np.max(uncertainty_stack, axis=0)
        temporal_geometry[~temporal_geometry_valid] = np.nan
        temporal_semantic[~temporal_semantic_observed] = np.nan
        temporal_uncertainty[~temporal_semantic_observed] = np.nan

        semantic_bev = SemanticBev(
            temporal_semantic,
            temporal_uncertainty,
            temporal_semantic_observed,
            projected_point_count=0,
        )
        fused, fused_valid = fuse_semantic_bev_with_geometry(
            semantic_bev, temporal_geometry, temporal_geometry_valid
        )
        return TemporalBev(
            temporal_geometry,
            temporal_geometry_valid,
            temporal_semantic,
            temporal_uncertainty,
            temporal_semantic_observed,
            fused,
            fused_valid,
            len(self._frames),
        )
