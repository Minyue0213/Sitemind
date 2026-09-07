import numpy as np

from sitemind.geometry import (
    GeometryConfig,
    build_elevation_grid,
    compute_geometry_risk,
    rasterize_max_to_grid,
)


def make_points(height: np.ndarray, resolution: float = 0.2) -> np.ndarray:
    points = []
    for row in range(height.shape[0]):
        for col in range(height.shape[1]):
            for offset in (-0.04, 0.0, 0.04):
                points.append(
                    ((col + 0.5) * resolution + offset, (row + 0.5) * resolution, height[row, col])
                )
    return np.asarray(points)


def test_flat_terrain_has_low_interior_risk():
    config = GeometryConfig()
    grid = build_elevation_grid(
        make_points(np.zeros((12, 12))),
        config,
        bounds_xy=(0.0, 2.4, 0.0, 2.4),
    )
    layers = compute_geometry_risk(grid, config)
    assert np.nanmax(layers.slope_deg[2:-2, 2:-2]) < 0.1
    assert np.max(layers.risk[3:-3, 3:-3]) == 0.0


def test_large_step_becomes_critical_risk():
    height = np.zeros((15, 15))
    height[:, 8:] = 0.6
    config = GeometryConfig()
    grid = build_elevation_grid(
        make_points(height),
        config,
        bounds_xy=(0.0, 3.0, 0.0, 3.0),
    )
    layers = compute_geometry_risk(grid, config)
    assert np.all(layers.risk[4:-4, 7:9] == 1.0)


def test_rasterize_max_keeps_most_conservative_point_value():
    config = GeometryConfig(resolution_m=1.0, min_points_per_cell=1)
    points = np.array([[0.1, 0.1, 0.0], [0.8, 0.8, 0.0], [1.2, 0.2, 0.0]])
    grid = build_elevation_grid(points, config, bounds_xy=(0.0, 2.0, 0.0, 1.0))
    result = rasterize_max_to_grid(points[:, :2], np.array([0.25, 1.0, 0.5]), grid)
    np.testing.assert_allclose(result, [[1.0, 0.5]])
