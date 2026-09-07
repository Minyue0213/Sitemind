import numpy as np
import pytest

from sitemind.planning.vehicle_origin import metric_cell, plan_from_vehicle_origin


KWARGS = dict(x_min_m=-3., y_min_m=-3., resolution_m=1.,
              self_exclusion_bounds_m=(-1., 1., -1., 1.))


def test_plan_starts_at_vehicle_origin_and_separates_unmeasured_connector():
    risk = np.zeros((6, 6), dtype=np.float32)
    risk[2:4, 2:4] = np.nan
    result = plan_from_vehicle_origin(risk, (5, 5), **KWARGS)
    assert result is not None
    assert result.route.path[0] == result.origin_cell == (3, 3)
    assert result.connector_path[-1] == result.verified_path[0]
    assert np.isfinite(risk[result.verified_path[0]])


def test_unknown_outside_self_region_is_not_opened():
    risk = np.full((6, 6), np.nan, dtype=np.float32)
    risk[5, 5] = 0.
    assert plan_from_vehicle_origin(risk, (5, 5), **KWARGS) is None


def test_forward_mode_rejects_goal_or_departure_behind_tracks():
    risk = np.zeros((6, 6), dtype=np.float32)
    assert plan_from_vehicle_origin(risk, (3, 5), forward_xy=(1., 0.), **KWARGS) is not None
    assert plan_from_vehicle_origin(risk, (3, 0), forward_xy=(1., 0.), **KWARGS) is None
    # A forward goal cannot be reached by silently backing out through the only opening.
    risk[:, :] = np.nan
    risk[3, 0] = 0.
    risk[3, 5] = 0.
    assert plan_from_vehicle_origin(risk, (3, 5), forward_xy=(1., 0.), **KWARGS) is None


def test_invalid_vehicle_reference_is_rejected():
    with pytest.raises(ValueError, match='inside the self-exclusion'):
        plan_from_vehicle_origin(np.zeros((6, 6)), (5, 5), origin_xy_m=(-2.5, -2.5), **KWARGS)


def test_metric_cell_checks_grid():
    assert metric_cell((0., 0.), (6, 6), x_min_m=-3., y_min_m=-3., resolution_m=1.) == (3, 3)
    with pytest.raises(ValueError, match='outside'):
        metric_cell((9., 9.), (6, 6), x_min_m=-3., y_min_m=-3., resolution_m=1.)
