"""Interpretable elevation-grid terrain features for SiteMind.

Risk uses SiteMind's convention: 0 is low risk and 1 is forbidden/unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    resolution_m: float = 0.2
    min_points_per_cell: int = 3
    slope_radius_cells: int = 1
    step_radius_cells: int = 3
    safe_slope_deg: float = 10.0
    critical_slope_deg: float = 35.0
    safe_step_m: float = 0.10
    critical_step_m: float = 0.35
    slope_weight: float = 0.5
    step_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if self.min_points_per_cell < 1:
            raise ValueError("min_points_per_cell must be at least one")
        if not 0 <= self.safe_slope_deg < self.critical_slope_deg:
            raise ValueError("slope thresholds must satisfy 0 <= safe < critical")
        if not 0 <= self.safe_step_m < self.critical_step_m:
            raise ValueError("step thresholds must satisfy 0 <= safe < critical")
        if not np.isclose(self.slope_weight + self.step_weight, 1.0):
            raise ValueError("slope_weight and step_weight must sum to one")


@dataclass(frozen=True)
class ElevationGrid:
    height_m: np.ndarray
    point_count: np.ndarray
    x_min_m: float
    y_min_m: float
    resolution_m: float

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.height_m)

    def cell_center_xy(self, row: int, col: int) -> Tuple[float, float]:
        return (
            self.x_min_m + (col + 0.5) * self.resolution_m,
            self.y_min_m + (row + 0.5) * self.resolution_m,
        )


@dataclass(frozen=True)
class GeometryLayers:
    elevation_m: np.ndarray
    slope_deg: np.ndarray
    step_height_m: np.ndarray
    risk: np.ndarray
    valid: np.ndarray


def rasterize_max_to_grid(
    points_xy: np.ndarray,
    values: np.ndarray,
    grid: ElevationGrid,
) -> np.ndarray:
    """Rasterize per-point values using the most conservative value per cell."""

    points = np.asarray(points_xy, dtype=np.float64)
    scalar = np.asarray(values, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points_xy must have shape (N, >=2)")
    if scalar.ndim != 1 or len(scalar) != len(points):
        raise ValueError("values must be a 1D array matching points_xy")

    cols = np.floor((points[:, 0] - grid.x_min_m) / grid.resolution_m).astype(int)
    rows = np.floor((points[:, 1] - grid.y_min_m) / grid.resolution_m).astype(int)
    height, width = grid.height_m.shape
    inside = (
        np.isfinite(scalar)
        & (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )
    flattened = np.full(height * width, -np.inf, dtype=np.float32)
    indices = rows[inside] * width + cols[inside]
    np.maximum.at(flattened, indices, scalar[inside])
    result = flattened.reshape(height, width)
    result[~np.isfinite(result)] = np.nan
    return result


def build_elevation_grid(
    points_xyz: np.ndarray,
    config: GeometryConfig = GeometryConfig(),
    bounds_xy: Optional[Tuple[float, float, float, float]] = None,
) -> ElevationGrid:
    """Aggregate an ``N x 3`` point cloud into a median-height BEV grid."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyz must have shape (N, >=3)")
    points = points[:, :3]
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        raise ValueError("points_xyz contains no finite points")

    resolution = config.resolution_m
    if bounds_xy is None:
        x_min = np.floor(points[:, 0].min() / resolution) * resolution
        y_min = np.floor(points[:, 1].min() / resolution) * resolution
        x_max = np.ceil(points[:, 0].max() / resolution) * resolution + resolution
        y_max = np.ceil(points[:, 1].max() / resolution) * resolution + resolution
    else:
        x_min, x_max, y_min, y_max = map(float, bounds_xy)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("bounds_xy must be (x_min, x_max, y_min, y_max)")

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution)))
    cols = np.floor((points[:, 0] - x_min) / resolution).astype(int)
    rows = np.floor((points[:, 1] - y_min) / resolution).astype(int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows, cols, z_values = rows[inside], cols[inside], points[inside, 2]

    buckets: dict[Tuple[int, int], list[float]] = {}
    for row, col, z_value in zip(rows, cols, z_values):
        buckets.setdefault((int(row), int(col)), []).append(float(z_value))

    elevation = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)
    for (row, col), values in buckets.items():
        counts[row, col] = len(values)
        if len(values) >= config.min_points_per_cell:
            elevation[row, col] = np.median(values)

    return ElevationGrid(elevation, counts, x_min, y_min, resolution)


def _window(array: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    r0, r1 = max(0, row - radius), min(array.shape[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(array.shape[1], col + radius + 1)
    return array[r0:r1, c0:c1]


def compute_slope_degrees(grid: ElevationGrid, radius_cells: int = 1) -> np.ndarray:
    """Estimate local surface slope using the smallest PCA eigenvector."""

    heights = grid.height_m
    result = np.full_like(heights, np.nan, dtype=np.float32)
    for row, col in np.argwhere(grid.valid):
        r0, r1 = max(0, row - radius_cells), min(heights.shape[0], row + radius_cells + 1)
        c0, c1 = max(0, col - radius_cells), min(heights.shape[1], col + radius_cells + 1)
        patch = heights[r0:r1, c0:c1]
        valid_rows, valid_cols = np.nonzero(np.isfinite(patch))
        if len(valid_rows) < 3:
            continue
        xyz = np.column_stack(
            (
                (valid_cols + c0 + 0.5) * grid.resolution_m,
                (valid_rows + r0 + 0.5) * grid.resolution_m,
                patch[valid_rows, valid_cols],
            )
        )
        centered = xyz - xyz.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / len(xyz)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        result[row, col] = np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
    return result


def compute_step_height(grid: ElevationGrid, radius_cells: int = 3) -> np.ndarray:
    """Return the maximum local height discontinuity around each valid cell."""

    heights = grid.height_m
    result = np.full_like(heights, np.nan, dtype=np.float32)
    for row, col in np.argwhere(grid.valid):
        patch = _window(heights, int(row), int(col), radius_cells)
        values = patch[np.isfinite(patch)]
        if len(values) < 2:
            continue
        result[row, col] = np.max(np.abs(values - heights[row, col]))
    return result


def compute_geometry_risk(
    grid: ElevationGrid,
    config: GeometryConfig = GeometryConfig(),
) -> GeometryLayers:
    """Compute slope, step height and an interpretable geometric risk layer."""

    slope = compute_slope_degrees(grid, config.slope_radius_cells)
    step = compute_step_height(grid, config.step_radius_cells)
    valid = grid.valid & np.isfinite(slope) & np.isfinite(step)
    risk = np.ones_like(grid.height_m, dtype=np.float32)

    safe = valid & (slope < config.safe_slope_deg) & (step < config.safe_step_m)
    critical = valid & (
        (slope > config.critical_slope_deg) | (step > config.critical_step_m)
    )
    middle = valid & ~safe & ~critical
    risk[safe] = 0.0
    risk[critical] = 1.0
    risk[middle] = np.clip(
        config.slope_weight * slope[middle] / config.critical_slope_deg
        + config.step_weight * step[middle] / config.critical_step_m,
        0.0,
        1.0,
    )
    return GeometryLayers(grid.height_m, slope, step, risk, valid)
