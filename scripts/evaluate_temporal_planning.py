"""Compare single-frame and temporal-BEV route availability and stability."""

from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path

import numpy as np

from sitemind.planning import AStarResult, astar


def nearest_common_cell(
    geometry: np.ndarray, fused: np.ndarray, target: tuple[int, int]
) -> tuple[int, int]:
    common = (geometry < 0.99) & (fused < 0.99)
    rows, cols = np.indices(common.shape)
    distance = (rows - target[0]) ** 2 + (cols - target[1]) ** 2
    distance[~common] = np.iinfo(np.int32).max
    if not np.any(common):
        raise RuntimeError("there is no cell traversable in both maps")
    return tuple(int(v) for v in np.unravel_index(np.argmin(distance), common.shape))


def route_length(route: AStarResult, resolution: float) -> float:
    return resolution * sum(
        hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route.path, route.path[1:])
    )


def route_exposure(route: AStarResult, risk: np.ndarray, resolution: float) -> float:
    return resolution * sum(float(risk[cell]) for cell in route.path)


def route_jitter(first: AStarResult, second: AStarResult, resolution: float) -> float:
    """Symmetric mean nearest-path distance between consecutive routes."""

    a = np.asarray(first.path, dtype=np.float32)
    b = np.asarray(second.path, dtype=np.float32)
    distances = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()) * resolution)


def evaluate(
    layer_root: Path,
    reference_root: Path,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    risk_weight: float,
) -> tuple[dict[str, object], list[AStarResult | None]]:
    paths = sorted(layer_root.glob("frame_*/fusion_layers.npz"))
    routes: list[AStarResult | None] = []
    lengths: list[float] = []
    exposures: list[float] = []
    frame_results: list[dict[str, object]] = []
    for path in paths:
        with np.load(path) as layers:
            geometry = layers["geometry_risk"].astype(np.float32)
            valid = layers["geometry_valid"].astype(bool)
            fused = layers["fused_risk"].astype(np.float32)
            resolution = float(layers["resolution_m"])
            x_min = float(layers["x_min_m"])
            y_min = float(layers["y_min_m"])
        geometry[~valid] = 1.0
        fused[~valid] = 1.0
        target_start = (
            int((start_xy[1] - y_min) / resolution),
            int((start_xy[0] - x_min) / resolution),
        )
        target_goal = (
            int((goal_xy[1] - y_min) / resolution),
            int((goal_xy[0] - x_min) / resolution),
        )
        start = nearest_common_cell(geometry, fused, target_start)
        goal = nearest_common_cell(geometry, fused, target_goal)
        route = astar(fused, start, goal, risk_weight=risk_weight)
        routes.append(route)
        item: dict[str, object] = {"frame_name": path.parent.name, "status": "stop"}
        if route is not None:
            with np.load(reference_root / path.parent.name / "fusion_layers.npz") as reference:
                current_semantic = np.where(
                    reference["semantic_observed"], reference["semantic_risk"], 0.0
                )
            length = route_length(route, resolution)
            exposure = route_exposure(route, current_semantic, resolution)
            lengths.append(length)
            exposures.append(exposure)
            item.update(status="route", length_m=length, current_semantic_exposure=exposure)
        frame_results.append(item)

    jitters = [
        route_jitter(first, second, resolution)
        for first, second in zip(routes, routes[1:])
        if first is not None and second is not None
    ]
    summary: dict[str, object] = {
        "frame_count": len(paths),
        "route_frames": sum(route is not None for route in routes),
        "stop_frames": sum(route is None for route in routes),
        "median_route_length_m": float(np.median(lengths)) if lengths else None,
        "total_current_semantic_exposure": float(np.sum(exposures)),
        "mean_consecutive_route_jitter_m": float(np.mean(jitters)) if jitters else None,
        "p95_consecutive_route_jitter_m": float(np.percentile(jitters, 95)) if jitters else None,
        "frames": frame_results,
    }
    return summary, routes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("single_frame_root", type=Path)
    parser.add_argument("temporal_root", type=Path)
    parser.add_argument("--start", nargs=2, type=float, default=(4.0, 1.0))
    parser.add_argument("--goal", nargs=2, type=float, default=(12.0, 6.0))
    parser.add_argument("--risk-weight", type=float, default=8.0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/alice_seq02_temporal_evaluation.json")
    )
    args = parser.parse_args()
    start, goal = tuple(args.start), tuple(args.goal)
    single, single_routes = evaluate(
        args.single_frame_root, args.single_frame_root, start, goal, args.risk_weight
    )
    temporal, temporal_routes = evaluate(
        args.temporal_root, args.single_frame_root, start, goal, args.risk_weight
    )
    common_indices = [
        index
        for index, (single_route, temporal_route) in enumerate(zip(single_routes, temporal_routes))
        if single_route is not None and temporal_route is not None
    ]
    single_common_exposure = sum(
        float(single["frames"][index]["current_semantic_exposure"]) for index in common_indices
    )
    temporal_common_exposure = sum(
        float(temporal["frames"][index]["current_semantic_exposure"]) for index in common_indices
    )
    jitter_before = float(single["mean_consecutive_route_jitter_m"])
    jitter_after = float(temporal["mean_consecutive_route_jitter_m"])
    comparison = {
        "single_frame": single,
        "temporal": temporal,
        "common_route_frames": len(common_indices),
        "common_frames_semantic_exposure_before": single_common_exposure,
        "common_frames_semantic_exposure_after": temporal_common_exposure,
        "common_frames_semantic_exposure_reduction": (
            0.0
            if single_common_exposure <= 1e-9
            else 1.0 - temporal_common_exposure / single_common_exposure
        ),
        "mean_route_jitter_reduction": (
            0.0 if jitter_before <= 1e-9 else 1.0 - jitter_after / jitter_before
        ),
        "start_xy_m": list(start),
        "goal_xy_m": list(goal),
        "risk_weight": args.risk_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"routes={single['route_frames']}->{temporal['route_frames']} "
        f"stops={single['stop_frames']}->{temporal['stop_frames']} "
        f"jitter_m={jitter_before:.3f}->{jitter_after:.3f} "
        f"common_exposure={single_common_exposure:.3f}->{temporal_common_exposure:.3f}"
    )


if __name__ == "__main__":
    main()
