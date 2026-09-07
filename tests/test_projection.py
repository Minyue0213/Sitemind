import numpy as np

from sitemind.fusion import (
    CameraCalibration,
    fuse_semantic_bev_with_geometry,
    project_lidar_to_image,
    project_semantic_risk_to_bev,
    resolve_frame_transform,
    rigid_transform,
)
from sitemind.geometry import GeometryConfig, build_elevation_grid


def calibration() -> CameraCalibration:
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    return CameraCalibration.from_intrinsic_matrix(intrinsic, np.eye(4), (11, 11))


def test_projection_rejects_behind_camera_and_outside_image():
    points = np.array(
        [
            [0.0, 0.0, 2.0],   # centre pixel (5, 5)
            [0.2, 0.0, 2.0],   # pixel (6, 5)
            [0.0, 0.0, -1.0],  # behind camera
            [5.0, 0.0, 1.0],   # outside image
        ]
    )
    result = project_lidar_to_image(points, calibration())
    np.testing.assert_array_equal(result.point_indices, [0, 1])
    np.testing.assert_array_equal(result.pixels_xy, [[5, 5], [6, 5]])


def test_projection_uses_z_buffer_for_same_pixel():
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    result = project_lidar_to_image(points, calibration(), occlusion_tolerance_m=0.1)
    np.testing.assert_array_equal(result.point_indices, [0])


def test_projection_applies_plumb_bob_distortion():
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    model = CameraCalibration(
        np.column_stack((intrinsic, np.zeros(3))),
        np.eye(4),
        20,
        20,
        intrinsic=intrinsic,
        distortion=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    result = project_lidar_to_image(np.array([[1.0, 0.0, 2.0]]), model)
    np.testing.assert_array_equal(result.pixels_xy, [[11, 5]])


def test_projected_semantics_change_aligned_bev_risk():
    points = np.array([[0.0, 0.0, 2.0], [0.3, 0.0, 2.0]])
    grid = build_elevation_grid(
        points,
        GeometryConfig(resolution_m=0.25, min_points_per_cell=1),
        bounds_xy=(-0.25, 0.5, -0.25, 0.25),
    )
    pixel_risk = np.zeros((11, 11), dtype=np.float32)
    pixel_risk[5, 5] = 0.9
    pixel_risk[5, 7] = 0.2
    uncertainty = np.zeros_like(pixel_risk)
    uncertainty[5, 7] = 0.8

    semantic = project_semantic_risk_to_bev(
        points,
        pixel_risk,
        uncertainty,
        calibration(),
        grid,
        uncertainty_weight=1.0,
    )
    assert semantic.projected_point_count == 2
    assert np.isclose(np.nanmax(semantic.risk), 0.9)

    geometry = np.full(grid.height_m.shape, 0.1, dtype=np.float32)
    geometry_valid = grid.valid
    fused, fused_valid = fuse_semantic_bev_with_geometry(semantic, geometry, geometry_valid)
    assert np.isclose(fused[1, 1], 0.9)
    assert np.isclose(fused[1, 2], 0.8)
    assert np.all(fused[~fused_valid] == 1.0)


def test_tf_graph_resolves_child_to_camera_chain():
    base_from_lidar = rigid_transform(np.array([1.0, 0.0, 0.0]), np.array([0, 0, 0, 1]))
    base_from_camera = rigid_transform(np.array([0.0, 2.0, 0.0]), np.array([0, 0, 0, 1]))
    camera_from_lidar = resolve_frame_transform(
        [("base", "lidar", base_from_lidar), ("base", "camera", base_from_camera)],
        "/lidar",
        "camera",
    )
    transformed = camera_from_lidar @ np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(transformed, [1.0, -2.0, 0.0, 1.0])
