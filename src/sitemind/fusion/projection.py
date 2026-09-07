"""Calibration-based projection from LiDAR points to camera semantics.

The module deliberately has no ROS dependency.  A bag reader only needs to
provide the camera projection matrix and the LiDAR-to-camera transform from
``camera_info`` and ``/tf_static``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable

import numpy as np

from sitemind.geometry.elevation_grid import ElevationGrid, rasterize_max_to_grid


@dataclass(frozen=True)
class CameraCalibration:
    """Minimal calibrated pinhole camera model.

    ``projection`` is the ROS/OpenCV 3x4 projection matrix.  The transform maps
    homogeneous LiDAR-frame coordinates into the optical camera frame, where
    +x is right, +y is down and +z points forward.
    """

    projection: np.ndarray
    lidar_to_camera: np.ndarray
    image_width: int
    image_height: int
    intrinsic: np.ndarray | None = None
    distortion: np.ndarray | None = None
    distortion_model: str = "plumb_bob"

    def __post_init__(self) -> None:
        projection = np.asarray(self.projection, dtype=np.float64)
        transform = np.asarray(self.lidar_to_camera, dtype=np.float64)
        if projection.shape != (3, 4):
            raise ValueError("projection must have shape (3, 4)")
        if transform.shape != (4, 4):
            raise ValueError("lidar_to_camera must have shape (4, 4)")
        if not np.all(np.isfinite(projection)) or not np.all(np.isfinite(transform)):
            raise ValueError("calibration matrices must be finite")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        intrinsic = None if self.intrinsic is None else np.asarray(self.intrinsic, dtype=np.float64)
        distortion = (
            None if self.distortion is None else np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        )
        if intrinsic is not None and intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape (3, 3)")
        if intrinsic is not None and not np.all(np.isfinite(intrinsic)):
            raise ValueError("intrinsic must be finite")
        if distortion is not None and not np.all(np.isfinite(distortion)):
            raise ValueError("distortion must be finite")
        if intrinsic is not None and self.distortion_model not in {"", "plumb_bob"}:
            raise ValueError(f"unsupported distortion model: {self.distortion_model}")
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "lidar_to_camera", transform)
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(self, "distortion", distortion)

    @classmethod
    def from_intrinsic_matrix(
        cls,
        intrinsic: np.ndarray,
        lidar_to_camera: np.ndarray,
        image_size: tuple[int, int],
    ) -> "CameraCalibration":
        """Create a calibration from a 3x3 intrinsic matrix and image size."""

        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape (3, 3)")
        projection = np.column_stack((intrinsic, np.zeros(3, dtype=np.float64)))
        width, height = image_size
        return cls(
            projection,
            lidar_to_camera,
            int(width),
            int(height),
            intrinsic=intrinsic,
            distortion=np.zeros(5, dtype=np.float64),
        )


@dataclass(frozen=True)
class ImageProjection:
    """Visible LiDAR samples and their nearest image pixels."""

    point_indices: np.ndarray
    pixels_xy: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True)
class SemanticBev:
    """Semantic risk rasterized in the coordinate system of an elevation grid."""

    risk: np.ndarray
    uncertainty: np.ndarray
    observed: np.ndarray
    projected_point_count: int


def rigid_transform(
    translation_xyz: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Return a homogeneous transform from ROS translation and quaternion."""

    translation = np.asarray(translation_xyz, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("translation and quaternion must have shapes (3,) and (4,)")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def resolve_frame_transform(
    transforms: Iterable[tuple[str, str, np.ndarray]],
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    """Resolve ``target <- source`` through a ROS TF graph.

    Each input tuple is ``(parent, child, T_parent_child)`` following ROS TF
    semantics.  Leading slashes are ignored when matching frame names.
    """

    normalize = lambda value: value.lstrip("/")
    source = normalize(source_frame)
    target = normalize(target_frame)
    if not source or not target:
        raise ValueError("source_frame and target_frame must be non-empty")
    if source == target:
        return np.eye(4, dtype=np.float64)
    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    for parent_name, child_name, matrix in transforms:
        parent, child = normalize(parent_name), normalize(child_name)
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("every TF transform must be a finite 4x4 matrix")
        graph.setdefault(child, []).append((parent, transform))
        graph.setdefault(parent, []).append((child, np.linalg.inv(transform)))

    queue = deque([(source, np.eye(4, dtype=np.float64))])
    visited = {source}
    while queue:
        frame, frame_from_source = queue.popleft()
        for neighbour, neighbour_from_frame in graph.get(frame, []):
            if neighbour in visited:
                continue
            neighbour_from_source = neighbour_from_frame @ frame_from_source
            if neighbour == target:
                return neighbour_from_source
            visited.add(neighbour)
            queue.append((neighbour, neighbour_from_source))
    raise ValueError(f"no TF path from {source_frame!r} to {target_frame!r}")


def project_lidar_to_image(
    points_xyz: np.ndarray,
    calibration: CameraCalibration,
    *,
    min_depth_m: float = 0.1,
    occlusion_tolerance_m: float = 0.5,
) -> ImageProjection:
    """Project LiDAR points and retain the front surface at each image pixel."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyz must have shape (N, >=3)")
    if min_depth_m <= 0:
        raise ValueError("min_depth_m must be positive")
    if occlusion_tolerance_m < 0:
        raise ValueError("occlusion_tolerance_m must be non-negative")

    xyz = points[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    homogeneous = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float64)))
    camera_xyz = (calibration.lidar_to_camera @ homogeneous.T).T
    depth = camera_xyz[:, 2]
    if calibration.intrinsic is None:
        projected = (calibration.projection @ camera_xyz.T).T
        denominator = projected[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = projected[:, :2] / denominator[:, None]
    else:
        denominator = depth
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = camera_xyz[:, :2] / depth[:, None]
        x, y = normalized[:, 0], normalized[:, 1]
        coefficients = np.zeros(5, dtype=np.float64)
        if calibration.distortion is not None:
            count = min(5, len(calibration.distortion))
            coefficients[:count] = calibration.distortion[:count]
        k1, k2, p1, p2, k3 = coefficients
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        distorted_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        homogeneous_pixels = calibration.intrinsic @ np.vstack(
            (distorted_x, distorted_y, np.ones(len(distorted_x), dtype=np.float64))
        )
        pixels = (homogeneous_pixels[:2] / homogeneous_pixels[2]).T
    finite_pixels = np.all(np.isfinite(pixels), axis=1)
    roundable = (
        finite
        & finite_pixels
        & (depth > min_depth_m)
        & (denominator > 0)
        & np.all(np.abs(pixels) <= 2 * max(calibration.image_width, calibration.image_height), axis=1)
    )
    pixel_int = np.zeros((len(pixels), 2), dtype=np.int64)
    pixel_int[roundable] = np.floor(pixels[roundable] + 0.5).astype(np.int64)
    inside = (
        roundable
        & (pixel_int[:, 0] >= 0)
        & (pixel_int[:, 0] < calibration.image_width)
        & (pixel_int[:, 1] >= 0)
        & (pixel_int[:, 1] < calibration.image_height)
    )
    indices = np.flatnonzero(inside)
    if not len(indices):
        return ImageProjection(indices, np.empty((0, 2), dtype=np.int64), np.empty(0))

    visible_pixels = pixel_int[indices]
    visible_depth = depth[indices]
    flat_pixels = visible_pixels[:, 1] * calibration.image_width + visible_pixels[:, 0]
    z_buffer = np.full(calibration.image_width * calibration.image_height, np.inf)
    np.minimum.at(z_buffer, flat_pixels, visible_depth)
    front = visible_depth <= z_buffer[flat_pixels] + occlusion_tolerance_m
    return ImageProjection(indices[front], visible_pixels[front], visible_depth[front])


def sample_image_at_projection(image: np.ndarray, projection: ImageProjection) -> np.ndarray:
    """Nearest-neighbour sampling for labels, risk, or uncertainty images."""

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    if len(projection.pixels_xy) == 0:
        return np.empty((0, *array.shape[2:]), dtype=array.dtype)
    x = projection.pixels_xy[:, 0]
    y = projection.pixels_xy[:, 1]
    if np.any(x < 0) or np.any(x >= array.shape[1]) or np.any(y < 0) or np.any(y >= array.shape[0]):
        raise ValueError("projection contains pixels outside the supplied image")
    return array[y, x]


def project_semantic_risk_to_bev(
    points_xyz: np.ndarray,
    pixel_risk: np.ndarray,
    pixel_uncertainty: np.ndarray,
    calibration: CameraCalibration,
    grid: ElevationGrid,
    *,
    uncertainty_weight: float = 1.0,
    occlusion_tolerance_m: float = 0.5,
) -> SemanticBev:
    """Project camera evidence onto LiDAR points and rasterize it into BEV."""

    risk = np.asarray(pixel_risk, dtype=np.float32)
    uncertainty = np.asarray(pixel_uncertainty, dtype=np.float32)
    expected_shape = (calibration.image_height, calibration.image_width)
    if risk.shape != expected_shape or uncertainty.shape != expected_shape:
        raise ValueError(f"risk and uncertainty images must have shape {expected_shape}")
    if np.nanmin(risk) < 0 or np.nanmax(risk) > 1:
        raise ValueError("pixel_risk values must lie in [0, 1]")
    if np.nanmin(uncertainty) < 0 or np.nanmax(uncertainty) > 1:
        raise ValueError("pixel_uncertainty values must lie in [0, 1]")
    if not 0 <= uncertainty_weight <= 1:
        raise ValueError("uncertainty_weight must lie in [0, 1]")

    projection = project_lidar_to_image(
        points_xyz,
        calibration,
        occlusion_tolerance_m=occlusion_tolerance_m,
    )
    point_risk = sample_image_at_projection(risk, projection)
    point_uncertainty = sample_image_at_projection(uncertainty, projection)
    guarded_point_risk = np.maximum(point_risk, uncertainty_weight * point_uncertainty)
    points = np.asarray(points_xyz)
    semantic_risk = rasterize_max_to_grid(
        points[projection.point_indices, :2], guarded_point_risk, grid
    )
    semantic_uncertainty = rasterize_max_to_grid(
        points[projection.point_indices, :2], point_uncertainty, grid
    )
    observed = np.isfinite(semantic_risk)
    return SemanticBev(semantic_risk, semantic_uncertainty, observed, len(projection.point_indices))


def fuse_semantic_bev_with_geometry(
    semantic: SemanticBev,
    geometry_risk: np.ndarray,
    geometry_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively fuse aligned BEV layers without treating missing data as safe."""

    geometry = np.asarray(geometry_risk, dtype=np.float32)
    valid = np.asarray(geometry_valid, dtype=bool)
    if geometry.shape != semantic.risk.shape or valid.shape != geometry.shape:
        raise ValueError("semantic and geometry BEV layers must have the same shape")
    finite_geometry = valid & np.isfinite(geometry)
    combined_valid = finite_geometry | semantic.observed
    fused = np.ones(geometry.shape, dtype=np.float32)
    fused[finite_geometry] = geometry[finite_geometry]
    only_semantic = semantic.observed & ~finite_geometry
    fused[only_semantic] = semantic.risk[only_semantic]
    both = semantic.observed & finite_geometry
    fused[both] = np.maximum(geometry[both], semantic.risk[both])
    return np.clip(fused, 0.0, 1.0), combined_valid
