"""LiDAR-derived terrain geometry."""

from .elevation_grid import (
    GeometryConfig,
    build_elevation_grid,
    compute_geometry_risk,
    rasterize_max_to_grid,
)

__all__ = [
    "GeometryConfig",
    "build_elevation_grid",
    "compute_geometry_risk",
    "rasterize_max_to_grid",
]
