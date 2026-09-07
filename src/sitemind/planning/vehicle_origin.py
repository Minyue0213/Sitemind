"""Connect a vehicle-centred start to observed terrain without claiming self-space is measured."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .astar import AStarResult, astar


@dataclass(frozen=True)
class VehicleOriginPlan:
    """A non-evaluated path inside self-space followed by an evaluated terrain path."""

    route: AStarResult
    origin_cell: tuple[int, int]
    connector_path: list[tuple[int, int]]
    verified_path: list[tuple[int, int]]


def metric_cell(xy: tuple[float, float], shape: tuple[int, int], *,
                x_min_m: float, y_min_m: float, resolution_m: float) -> tuple[int, int]:
    cell = (int(np.floor((xy[1] - y_min_m) / resolution_m)),
            int(np.floor((xy[0] - x_min_m) / resolution_m)))
    if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
        raise ValueError('vehicle reference lies outside the BEV grid')
    return cell


def plan_from_vehicle_origin(
    risk: np.ndarray,
    goal: tuple[int, int],
    *,
    x_min_m: float,
    y_min_m: float,
    resolution_m: float,
    origin_xy_m: tuple[float, float] = (0.0, 0.0),
    self_exclusion_bounds_m: tuple[float, float, float, float],
    risk_weight: float = 5.0,
    forward_xy: tuple[float, float] | None = None,
    max_goal_angle_deg: float = 80.0,
) -> VehicleOriginPlan | None:
    """Plan from the real reference point while isolating the unobserved self region.

    Only cells inside the supplied *sensor self-exclusion* region are opened for
    the initial connector. Unknown cells outside it remain blocked. The two path
    segments are returned separately so a renderer cannot imply that terrain
    underneath the machine was measured safe.
    """
    risk = np.asarray(risk, dtype=np.float32)
    if risk.ndim != 2 or resolution_m <= 0:
        raise ValueError('risk must be 2D and resolution_m must be positive')
    x_low, x_high, y_low, y_high = self_exclusion_bounds_m
    if not x_low < x_high or not y_low < y_high:
        raise ValueError('invalid self-exclusion bounds')
    rows, cols = np.indices(risk.shape)
    x = x_min_m + (cols + 0.5) * resolution_m
    y = y_min_m + (rows + 0.5) * resolution_m
    self_mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    origin = metric_cell(origin_xy_m, risk.shape, x_min_m=x_min_m,
                         y_min_m=y_min_m, resolution_m=resolution_m)
    if not self_mask[origin]:
        raise ValueError('vehicle reference must lie inside the self-exclusion region')

    forward = None
    if forward_xy is not None:
        forward = np.asarray(forward_xy, dtype=np.float64)
        norm = np.linalg.norm(forward)
        if not np.isfinite(norm) or norm < 1e-9 or not 0 < max_goal_angle_deg < 90:
            raise ValueError('forward direction and goal angle must be valid')
        forward /= norm
        goal_xy = np.array([
            x_min_m + (goal[1] + 0.5) * resolution_m,
            y_min_m + (goal[0] + 0.5) * resolution_m,
        ])
        displacement = goal_xy - np.asarray(origin_xy_m)
        distance = np.linalg.norm(displacement)
        if distance < resolution_m or np.dot(displacement / distance, forward) < np.cos(np.deg2rad(max_goal_angle_deg)):
            return None

    planning = risk.copy()
    planning[self_mask] = 0.0
    route = astar(planning, origin, goal, risk_weight=risk_weight)
    if route is None:
        return None
    outside = [index for index, cell in enumerate(route.path) if not self_mask[cell]]
    if not outside:
        return None
    first_outside = outside[0]
    if any(self_mask[cell] for cell in route.path[first_outside:]):
        return None
    # Include the first measured cell in both lists to show a continuous handoff.
    connector = route.path[: first_outside + 1]
    verified = route.path[first_outside:]
    if forward is not None:
        exit_xy = np.array([
            x_min_m + (verified[0][1] + 0.5) * resolution_m,
            y_min_m + (verified[0][0] + 0.5) * resolution_m,
        ])
        if np.dot(exit_xy - np.asarray(origin_xy_m), forward) <= 0:
            return None
    return VehicleOriginPlan(route, origin, connector, verified)
