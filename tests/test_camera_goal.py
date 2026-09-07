import numpy as np
import pytest

from sitemind.fusion.projection import CameraCalibration
from sitemind.planning.camera_goal import (
    GroundCorrespondence, bounded_start, route_image_samples,
    select_camera_goal, transform_goal_point, visible_ground,
)


def samples():
    return GroundCorrespondence(
        np.array([[3., 1., 2.], [3.1, 1., 2.], [3., 1.1, 2.]]),
        np.array([[10, 10], [11, 10], [10, 11]]),
        np.array([[1, 1], [1, 1], [1, 1]]), np.array([2., 2., 2.]),
    )


def test_camera_goal_preserves_measured_cell():
    assert select_camera_goal(samples(), (10, 10), np.zeros((3, 3))) == 0


def test_measured_goal_can_be_locked_in_world_and_recovered_locally():
    world_from_local = np.array([
        [0., -1., 0., 10.],
        [1., 0., 0., 20.],
        [0., 0., 1., 2.],
        [0., 0., 0., 1.],
    ])
    local = np.array([3., 1., -1.])
    world = transform_goal_point(world_from_local, local)
    np.testing.assert_allclose(world, [9., 23., 1.])
    np.testing.assert_allclose(transform_goal_point(np.linalg.inv(world_from_local), world), local)


def test_camera_goal_checks_local_distance_and_chassis_forward_sector():
    source = samples()
    assert select_camera_goal(source, (10, 10), np.zeros((3, 3)), forward_xy=(1., 0.)) == 0
    with pytest.raises(ValueError, match='前向扇区'):
        select_camera_goal(source, (10, 10), np.zeros((3, 3)), forward_xy=(-1., 0.))
    with pytest.raises(ValueError, match='距离'):
        select_camera_goal(source, (10, 10), np.zeros((3, 3)), min_distance_m=5.)


def test_camera_goal_rejects_unknown_blocked_or_missing_without_snapping():
    risk = np.zeros((3, 3))
    for blocked in [1., np.nan]:
        risk[1, 1] = blocked
        with pytest.raises(ValueError, match='不自动移动'):
            select_camera_goal(samples(), (10, 10), risk)
    with pytest.raises(ValueError, match='没有足够'):
        select_camera_goal(samples(), (100, 100), np.zeros((3, 3)))


def test_camera_goal_rejects_depth_edge():
    source = samples()
    source.depth[2] = 4.
    with pytest.raises(ValueError, match='深度不连续'):
        select_camera_goal(source, (10, 10), np.zeros((3, 3)))


def test_start_has_metric_snap_limit():
    layers = dict(resolution_m=1., x_min_m=0., y_min_m=0.)
    risk = np.ones((4, 4))
    risk[3, 3] = 0.
    with pytest.raises(ValueError, match='不自动跳'):
        bounded_start(risk, (0., 0.), layers)
    assert bounded_start(risk, (3.4, 3.4), layers) == (3, 3)


def test_route_preserves_invisible_gaps():
    assert route_image_samples(samples(), [(1, 1), (2, 2), (1, 1)]) == [(10, 10), None, (10, 10)]


def test_raw_foreground_occludes_ground_and_outside_grid_rejected():
    calibration = CameraCalibration.from_intrinsic_matrix(
        np.array([[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]]), np.eye(4), (30, 30))
    layers = dict(resolution_m=1., x_min_m=0., y_min_m=0.,
                  geometry_valid=np.ones((3, 3), dtype=bool), elevation_m=np.full((3, 3), 2.))
    # Rear ground at the same pixel must not survive a nearer machine return.
    points = np.array([[0., 0., 2.], [0., 0., 1.], [1., 1., 2.], [-0.2, 0., 2.]])
    visible = visible_ground(points, calibration, layers)
    np.testing.assert_allclose(visible.xyz, [[1., 1., 2.]])
