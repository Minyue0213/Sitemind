"""Measured camera/BEV correspondence; no flat-ground or invisible-path guesses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sitemind.fusion.projection import CameraCalibration, project_lidar_to_image


@dataclass(frozen=True)
class GroundCorrespondence:
    xyz: np.ndarray
    pixels: np.ndarray
    cells: np.ndarray
    depth: np.ndarray


def transform_goal_point(transform: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Transform one measured 3D goal between local and world frames."""
    matrix = np.asarray(transform, dtype=np.float64)
    point = np.asarray(xyz, dtype=np.float64)
    if matrix.shape != (4, 4) or point.shape != (3,):
        raise ValueError('goal transform and point must have shapes (4, 4) and (3,)')
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(point)):
        raise ValueError('goal transform and point must be finite')
    homogeneous = np.append(point, 1.0)
    result = matrix @ homogeneous
    if abs(float(result[3])) < 1e-12:
        raise ValueError('goal transform produced an invalid homogeneous point')
    return result[:3] / result[3]


def visible_ground(points: np.ndarray, calibration: CameraCalibration, layers: dict) -> GroundCorrespondence:
    """Keep current measured near-ground returns, with neighbourhood occlusion checks.

    All raw returns (including the excavator) participate in occlusion testing.
    This is a sampled visibility check, not a complete surface reconstruction.
    """
    projection = project_lidar_to_image(points, calibration, occlusion_tolerance_m=0.15)
    xyz = points[projection.point_indices, :3]
    pixels = projection.pixels_xy
    depth = projection.depth_m
    # Nearby foreground returns conservatively suppress background through sparse holes.
    zbuffer = np.full((calibration.image_height, calibration.image_width), np.inf)
    np.minimum.at(zbuffer, (pixels[:, 1], pixels[:, 0]), depth)
    nearest_depth = depth.copy()
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            x = np.clip(pixels[:, 0] + dx, 0, calibration.image_width - 1)
            y = np.clip(pixels[:, 1] + dy, 0, calibration.image_height - 1)
            nearest_depth = np.minimum(nearest_depth, zbuffer[y, x])
    resolution = float(layers['resolution_m'])
    cols = np.floor((xyz[:, 0] - float(layers['x_min_m'])) / resolution).astype(int)
    rows = np.floor((xyz[:, 1] - float(layers['y_min_m'])) / resolution).astype(int)
    shape = layers['geometry_valid'].shape
    inside = (rows >= 0) & (cols >= 0) & (rows < shape[0]) & (cols < shape[1])
    indices = np.flatnonzero(inside & (depth <= nearest_depth + 0.15))
    r, c = rows[indices], cols[indices]
    ground = (layers['geometry_valid'][r, c].astype(bool)
              & np.isfinite(layers['elevation_m'][r, c])
              & (np.abs(xyz[indices, 2] - layers['elevation_m'][r, c]) <= 0.20))
    indices = indices[ground]
    return GroundCorrespondence(xyz[indices], pixels[indices],
                                np.column_stack((rows[indices], cols[indices])), depth[indices])


def select_camera_goal(samples: GroundCorrespondence, pixel: tuple[float, float],
                       risk: np.ndarray, radius_px: float = 12.0, *,
                       origin_xy_m: tuple[float, float] = (0.0, 0.0),
                       forward_xy: tuple[float, float] | None = None,
                       min_distance_m: float = 2.0,
                       max_distance_m: float = 12.0,
                       max_forward_angle_deg: float = 70.0) -> int:
    """Resolve a click locally, reject sparse/ambiguous/blocked ground, never search elsewhere."""
    if radius_px <= 0 or not np.all(np.isfinite(pixel)):
        raise ValueError('invalid target pixel or radius')
    distances = np.linalg.norm(samples.pixels - np.asarray(pixel), axis=1)
    nearby = np.flatnonzero(distances <= radius_px)
    if len(nearby) < 3:
        raise ValueError('目标附近没有足够的可见地面深度，请换一个位置')
    depth = samples.depth[nearby]
    if np.ptp(depth) > 0.5:
        raise ValueError('目标位于深度不连续边缘，不能可靠确定地面位置')
    index = int(nearby[np.argmin(distances[nearby])])
    cell = tuple(samples.cells[index])
    if not np.isfinite(risk[cell]) or risk[cell] >= 0.99:
        raise ValueError('所选目标在未知或不可通行区域，不自动移动目标')
    displacement = samples.xyz[index, :2] - np.asarray(origin_xy_m, dtype=np.float64)
    distance_m = float(np.linalg.norm(displacement))
    if not 0 <= min_distance_m < max_distance_m or not min_distance_m <= distance_m <= max_distance_m:
        raise ValueError('目标距离不适合作为本次局部移位终点')
    if forward_xy is not None:
        forward = np.asarray(forward_xy, dtype=np.float64)
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(norm) or norm < 1e-9 or not 0 < max_forward_angle_deg < 90:
            raise ValueError('底盘前进方向或前向扇区参数无效')
        cosine = float(np.dot(displacement / distance_m, forward / norm))
        if cosine < np.cos(np.deg2rad(max_forward_angle_deg)):
            raise ValueError('目标不在履带底盘前向扇区内；倒车任务必须显式选择')
    return index


def bounded_start(risk: np.ndarray, xy: tuple[float, float], layers: dict,
                  max_distance_m: float = 0.75) -> tuple[int, int]:
    """Find a nearby start proxy with an explicit metric displacement limit."""
    rows, cols = np.indices(risk.shape)
    resolution = float(layers['resolution_m'])
    distances = np.hypot(float(layers['x_min_m']) + (cols + 0.5) * resolution - xy[0],
                         float(layers['y_min_m']) + (rows + 0.5) * resolution - xy[1])
    valid = np.isfinite(risk) & (risk < 0.99) & (distances <= max_distance_m)
    if not np.any(valid):
        raise ValueError('近场起点附近没有可通行地面，不自动跳到远处')
    return tuple(int(v) for v in np.unravel_index(np.argmin(np.where(valid, distances, np.inf)), risk.shape))


def route_image_samples(samples: GroundCorrespondence, path: list) -> list:
    """One measured pixel per route cell; None means unsupported or invisible.

    Pixels represent the route's grid cells, not an exact continuous centreline.
    Render as dots, without bridging unsupported cells.
    """
    by_cell = {}
    for index, cell in enumerate(samples.cells):
        by_cell.setdefault(tuple(cell), []).append(index)
    result = []
    for cell in path:
        indices = by_cell.get(tuple(cell), [])
        if not indices:
            result.append(None)
        else:
            pixels = samples.pixels[indices]
            index = np.argmin(np.linalg.norm(pixels - np.median(pixels, axis=0), axis=1))
            result.append(tuple(int(v) for v in pixels[index]))
    return result
