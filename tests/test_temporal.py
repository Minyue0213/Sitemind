import numpy as np
import pytest

from sitemind.fusion import BevGridSpec, TemporalBevMemory, warp_bev_nearest


def frame(observed_cell: tuple[int, int], geometry_risk: float, semantic_risk: float):
    shape = (3, 3)
    geometry = np.full(shape, np.nan, dtype=np.float32)
    geometry_valid = np.zeros(shape, dtype=bool)
    semantic = np.full(shape, np.nan, dtype=np.float32)
    uncertainty = np.full(shape, np.nan, dtype=np.float32)
    semantic_observed = np.zeros(shape, dtype=bool)
    geometry[observed_cell] = geometry_risk
    geometry_valid[observed_cell] = True
    semantic[observed_cell] = semantic_risk
    uncertainty[observed_cell] = 0.1
    semantic_observed[observed_cell] = True
    return geometry, geometry_valid, semantic, uncertainty, semantic_observed


def test_temporal_memory_fills_one_frame_observation_gaps():
    memory = TemporalBevMemory(window_size=2)
    memory.update(*frame((1, 1), 0.2, 0.8))
    result = memory.update(*frame((1, 2), 0.1, 0.3))
    assert result.history_size == 2
    assert result.geometry_valid[1, 1]
    assert result.semantic_observed[1, 1]
    assert np.isclose(result.fused_risk[1, 1], 0.8)
    assert np.isclose(result.fused_risk[1, 2], 0.3)


def test_temporal_memory_forgets_frames_outside_window():
    memory = TemporalBevMemory(window_size=2)
    memory.update(*frame((0, 0), 0.2, 0.8))
    memory.update(*frame((1, 1), 0.2, 0.4))
    result = memory.update(*frame((2, 2), 0.2, 0.3))
    assert not result.geometry_valid[0, 0]
    assert not result.semantic_observed[0, 0]


def test_temporal_memory_suppresses_one_frame_geometry_spike():
    memory = TemporalBevMemory(window_size=3)
    memory.update(*frame((1, 1), 0.2, 0.3))
    memory.update(*frame((1, 1), 1.0, 0.3))
    result = memory.update(*frame((1, 1), 0.2, 0.3))
    assert np.isclose(result.geometry_risk[1, 1], 0.2)


def test_temporal_memory_rejects_shape_changes():
    memory = TemporalBevMemory(window_size=2)
    memory.update(*frame((1, 1), 0.2, 0.8))
    arrays = [np.zeros((2, 2), dtype=np.float32) for _ in range(5)]
    arrays[1] = arrays[1].astype(bool)
    arrays[4] = arrays[4].astype(bool)
    with pytest.raises(ValueError, match="shape changed"):
        memory.update(*arrays)


def test_warp_bev_accounts_for_vehicle_translation():
    values = np.full((3, 3), np.nan, dtype=np.float32)
    valid = np.zeros((3, 3), dtype=bool)
    values[1, 2] = 0.9
    valid[1, 2] = True
    grid = BevGridSpec((3, 3), 0.0, 0.0, 1.0)
    world_from_source = np.eye(4)
    world_from_target = np.eye(4)
    world_from_target[0, 3] = 1.0

    warped, warped_valid = warp_bev_nearest(
        values, valid, grid, grid, world_from_source, world_from_target
    )

    assert warped_valid[1, 1]
    assert np.isclose(warped[1, 1], 0.9)
    assert not warped_valid[1, 2]


def test_warp_bev_accounts_for_vehicle_yaw():
    values = np.full((3, 3), np.nan, dtype=np.float32)
    valid = np.zeros((3, 3), dtype=bool)
    values[1, 2] = 0.8  # world point (x=1, y=0)
    valid[1, 2] = True
    grid = BevGridSpec((3, 3), -1.5, -1.5, 1.0)
    world_from_source = np.eye(4)
    world_from_target = np.eye(4)
    world_from_target[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])

    warped, warped_valid = warp_bev_nearest(
        values, valid, grid, grid, world_from_source, world_from_target
    )

    # The same world point lies at (x=0, y=-1) in the rotated target frame.
    assert warped_valid[0, 1]
    assert np.isclose(warped[0, 1], 0.8)


def test_temporal_memory_aligns_history_into_current_vehicle_frame():
    memory = TemporalBevMemory(window_size=2)
    first_pose = np.eye(4)
    first = frame((1, 2), 0.2, 0.9)
    memory.update(
        *first,
        world_from_frame=first_pose,
        x_min_m=0.0,
        y_min_m=0.0,
        resolution_m=1.0,
    )

    current_pose = np.eye(4)
    current_pose[0, 3] = 1.0
    current = frame((0, 0), 0.1, 0.2)
    result = memory.update(
        *current,
        world_from_frame=current_pose,
        x_min_m=0.0,
        y_min_m=0.0,
        resolution_m=1.0,
    )

    assert result.semantic_observed[1, 1]
    assert np.isclose(result.semantic_risk[1, 1], 0.9)
    assert not result.semantic_observed[1, 2]


def test_temporal_memory_rejects_mixed_pose_modes():
    memory = TemporalBevMemory(window_size=2)
    memory.update(*frame((1, 1), 0.2, 0.8))
    with pytest.raises(ValueError, match="cannot mix"):
        memory.update(*frame((1, 1), 0.2, 0.8), world_from_frame=np.eye(4))
