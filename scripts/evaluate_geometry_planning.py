"""Stress-test fixed-waypoint planning across real GOOSE-Ex LiDAR frames."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from math import hypot
from pathlib import Path

import numpy as np
import yaml

from sitemind.data import discover_point_cloud_pairs, load_point_cloud
from sitemind.geometry import GeometryConfig, build_elevation_grid, compute_geometry_risk
from sitemind.planning import AStarResult, astar


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FrameResult:
    index: int
    sequence: str
    status: str
    points_in_roi: int
    geometry_coverage: float
    start_snap_m: float = float("nan")
    goal_snap_m: float = float("nan")
    shortest_length_m: float = float("nan")
    safer_length_m: float = float("nan")
    shortest_exposure: float = float("nan")
    safer_exposure: float = float("nan")


def nearest_traversable(
    risk: np.ndarray, target: tuple[int, int], resolution: float
) -> tuple[tuple[int, int], float]:
    rows, cols = np.indices(risk.shape)
    distance_squared = (rows - target[0]) ** 2 + (cols - target[1]) ** 2
    distance_squared[risk >= 0.99] = np.iinfo(np.int32).max
    cell = tuple(int(v) for v in np.unravel_index(np.argmin(distance_squared), risk.shape))
    return cell, float(np.sqrt(distance_squared[cell]) * resolution)


def path_length(result: AStarResult, resolution: float) -> float:
    return resolution * sum(
        hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(result.path, result.path[1:])
    )


def path_exposure(result: AStarResult, risk: np.ndarray, resolution: float) -> float:
    return resolution * sum(float(risk[cell]) for cell in result.path)


def connected_patch_endpoints(
    risk: np.ndarray, resolution: float, minimum_span_m: float
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Find distant endpoints in the widest four-connected observed patch."""

    free = risk < 0.99
    seen = np.zeros_like(free, dtype=bool)
    best: tuple[int, list[tuple[int, int]], int] | None = None
    for seed_row, seed_col in np.argwhere(free):
        seed = (int(seed_row), int(seed_col))
        if seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        cells: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + d_row, col + d_col)
                if (
                    0 <= neighbor[0] < free.shape[0]
                    and 0 <= neighbor[1] < free.shape[1]
                    and free[neighbor]
                    and not seen[neighbor]
                ):
                    seen[neighbor] = True
                    stack.append(neighbor)
        coordinates = np.asarray(cells)
        row_span = int(coordinates[:, 0].max() - coordinates[:, 0].min())
        col_span = int(coordinates[:, 1].max() - coordinates[:, 1].min())
        span = max(row_span, col_span)
        if best is None or span > best[0]:
            best = (span, cells, 0 if row_span >= col_span else 1)
    if best is None or best[0] * resolution < minimum_span_m:
        return None
    _, cells, axis = best
    return min(cells, key=lambda cell: cell[axis]), max(cells, key=lambda cell: cell[axis])


def evaluate_one(task: tuple) -> FrameResult:
    (
        index,
        sequence,
        scan_path,
        config,
        bounds,
        z_range,
        start_xy,
        goal_xy,
        max_snap_m,
        risk_weight,
        endpoint_mode,
        minimum_route_m,
    ) = task
    points = load_point_cloud(scan_path)
    x_min, x_max, y_min, y_max = bounds
    z_min, z_max = z_range
    ego_footprint = (
        (points[:, 0] >= -3.0)
        & (points[:, 0] <= 4.0)
        & (points[:, 1] >= -1.6)
        & (points[:, 1] <= 1.6)
    )
    keep = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
        & ~ego_footprint
    )
    points = points[keep]
    if len(points) < 3:
        return FrameResult(index, sequence, "too_few_points", len(points), 0.0)
    grid = build_elevation_grid(points[:, :3], config, bounds)
    geometry = compute_geometry_risk(grid, config)
    risk = geometry.risk.copy()
    risk[~geometry.valid] = 1.0

    def cell(xy: tuple[float, float]) -> tuple[int, int]:
        x, y = xy
        return (int((y - y_min) / config.resolution_m), int((x - x_min) / config.resolution_m))

    if endpoint_mode == "component":
        endpoints = connected_patch_endpoints(risk, config.resolution_m, minimum_route_m)
        if endpoints is None:
            return FrameResult(
                index,
                sequence,
                "no_long_connected_patch",
                len(points),
                float(geometry.valid.mean()),
            )
        start, goal = endpoints
        start_snap = goal_snap = 0.0
    else:
        start, start_snap = nearest_traversable(risk, cell(start_xy), config.resolution_m)
        goal, goal_snap = nearest_traversable(risk, cell(goal_xy), config.resolution_m)
    base = FrameResult(
        index,
        sequence,
        "",
        len(points),
        float(geometry.valid.mean()),
        start_snap,
        goal_snap,
    )
    if endpoint_mode == "fixed" and max(start_snap, goal_snap) > max_snap_m:
        return replace(base, status="insufficient_endpoint_evidence")
    shortest = astar(risk, start, goal, risk_weight=0.0)
    safer = astar(risk, start, goal, risk_weight=risk_weight)
    if shortest is None or safer is None:
        return replace(base, status="not_connected")
    return replace(
        base,
        status="success",
        shortest_length_m=path_length(shortest, config.resolution_m),
        safer_length_m=path_length(safer, config.resolution_m),
        shortest_exposure=path_exposure(shortest, risk, config.resolution_m),
        safer_exposure=path_exposure(safer, risk, config.resolution_m),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--geometry-config", type=Path, default=ROOT / "configs" / "geometry.yaml")
    parser.add_argument("--bounds", nargs=4, type=float, default=(-20.0, 20.0, -20.0, 20.0))
    parser.add_argument("--z-range", nargs=2, type=float, default=(-2.0, 5.0))
    parser.add_argument("--start", nargs=2, type=float, default=(3.0, 0.0))
    parser.add_argument("--goal", nargs=2, type=float, default=(15.0, 0.0))
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--max-snap-m", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=8.0)
    parser.add_argument("--endpoint-mode", choices=("fixed", "component"), default="fixed")
    parser.add_argument("--minimum-route-m", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "geometry_planning_eval")
    args = parser.parse_args()

    with args.geometry_config.open(encoding="utf-8") as handle:
        config = GeometryConfig(**yaml.safe_load(handle))
    config = replace(config, resolution_m=args.resolution, min_points_per_cell=1, step_radius_cells=2)
    pairs = discover_point_cloud_pairs(args.dataset_root)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    tasks = [
        (
            index,
            pair.sequence,
            pair.scan_path,
            config,
            tuple(args.bounds),
            tuple(args.z_range),
            tuple(args.start),
            tuple(args.goal),
            args.max_snap_m,
            args.risk_weight,
            args.endpoint_mode,
            args.minimum_route_m,
        )
        for index, pair in enumerate(pairs)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(evaluate_one, tasks))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    successes = [result for result in results if result.status == "success"]
    overhead = np.array(
        [result.safer_length_m / result.shortest_length_m - 1.0 for result in successes]
    )
    valid_exposure = [result for result in successes if result.shortest_exposure > 1e-9]
    reduction = np.array(
        [1.0 - result.safer_exposure / result.shortest_exposure for result in valid_exposure]
    )
    status_counts = {status: sum(result.status == status for result in results) for status in sorted({r.status for r in results})}
    summary = {
        "frames": len(results),
        "status_counts": status_counts,
        "success_rate": len(successes) / max(1, len(results)),
        "median_geometry_coverage": float(np.median([r.geometry_coverage for r in results])),
        "median_distance_overhead": float(np.median(overhead)) if len(overhead) else None,
        "median_risk_reduction": float(np.median(reduction)) if len(reduction) else None,
        "risk_reduced_frames": int(np.count_nonzero(reduction > 1e-9)),
        "risk_comparable_frames": len(valid_exposure),
        "configuration": {
            "bounds": list(args.bounds),
            "start_xy": list(args.start),
            "goal_xy": list(args.goal),
            "resolution_m": args.resolution,
            "max_snap_m": args.max_snap_m,
            "risk_weight": args.risk_weight,
            "endpoint_mode": args.endpoint_mode,
            "minimum_route_m": args.minimum_route_m,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
