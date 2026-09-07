"""Pose-aware alignment for local bird's-eye-view grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BevGridSpec:
    """Metric definition of a local BEV raster."""

    shape: tuple[int, int]
    x_min_m: float
    y_min_m: float
    resolution_m: float

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape must contain two positive dimensions")
        if not np.isfinite([self.x_min_m, self.y_min_m, self.resolution_m]).all():
            raise ValueError("grid origin and resolution must be finite")
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")


def planar_pose(world_from_frame: np.ndarray) -> np.ndarray:
    """Extract an SE(2) pose from a homogeneous SE(3) or SE(2) transform."""

    transform = np.asarray(world_from_frame, dtype=np.float64)
    if transform.shape == (4, 4):
        translation = transform[:2, 3]
        x_axis = transform[:2, 0]
    elif transform.shape == (3, 3):
        translation = transform[:2, 2]
        x_axis = transform[:2, 0]
    else:
        raise ValueError("world_from_frame must have shape (4, 4) or (3, 3)")
    if not np.all(np.isfinite(transform)):
        raise ValueError("world_from_frame must be finite")
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-9:
        raise ValueError("world_from_frame has a degenerate planar rotation")
    cosine, sine = x_axis / norm
    return np.array(
        [
            [cosine, -sine, translation[0]],
            [sine, cosine, translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def warp_bev_nearest(
    values: np.ndarray,
    valid: np.ndarray,
    source_grid: BevGridSpec,
    target_grid: BevGridSpec,
    world_from_source: np.ndarray,
    world_from_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a source-frame BEV into a target vehicle frame.

    BEV cells are sampled at their metric centres with nearest-neighbour
    lookup. This preserves discrete risk observations and never invents values
    outside the source field of view.
    """

    source_values = np.asarray(values, dtype=np.float32)
    source_valid = np.asarray(valid, dtype=bool)
    if source_values.shape != source_grid.shape or source_valid.shape != source_grid.shape:
        raise ValueError("source arrays do not match source_grid.shape")

    source_pose = planar_pose(world_from_source)
    target_pose = planar_pose(world_from_target)
    source_from_target = np.linalg.inv(source_pose) @ target_pose

    rows, cols = np.indices(target_grid.shape, dtype=np.float64)
    target_x = target_grid.x_min_m + (cols.ravel() + 0.5) * target_grid.resolution_m
    target_y = target_grid.y_min_m + (rows.ravel() + 0.5) * target_grid.resolution_m
    target_xy1 = np.vstack((target_x, target_y, np.ones(target_x.size)))
    source_xy = source_from_target @ target_xy1
    source_cols = np.floor(
        (source_xy[0] - source_grid.x_min_m) / source_grid.resolution_m
    ).astype(np.int64)
    source_rows = np.floor(
        (source_xy[1] - source_grid.y_min_m) / source_grid.resolution_m
    ).astype(np.int64)
    inside = (
        (source_rows >= 0)
        & (source_rows < source_grid.shape[0])
        & (source_cols >= 0)
        & (source_cols < source_grid.shape[1])
    )

    output = np.full(target_grid.shape, np.nan, dtype=np.float32)
    output_valid = np.zeros(target_grid.shape, dtype=bool)
    flat_output = output.ravel()
    flat_valid = output_valid.ravel()
    target_indices = np.flatnonzero(inside)
    sampled_valid = source_valid[source_rows[inside], source_cols[inside]]
    accepted = target_indices[sampled_valid]
    flat_output[accepted] = source_values[
        source_rows[inside][sampled_valid], source_cols[inside][sampled_valid]
    ]
    flat_valid[accepted] = True
    return output, output_valid
