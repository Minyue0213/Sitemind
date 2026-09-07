"""Build trailing-window temporal BEV layers for an already aligned sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sitemind.fusion import TemporalBevMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_root", type=Path)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument(
        "--ignore-poses",
        action="store_true",
        help="ablation only: aggregate local grids without compensating vehicle motion",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/alice_seq02_sequence_temporal")
    )
    args = parser.parse_args()
    paths = sorted(args.frame_root.glob("frame_*/fusion_layers.npz"))
    if not paths:
        raise SystemExit(f"no fusion_layers.npz files found below {args.frame_root}")

    memory = TemporalBevMemory(args.window_size)
    reference_grid: tuple[tuple[int, ...], float, float, float] | None = None
    pose_mode: bool | None = None
    summaries: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        with np.load(path) as source:
            arrays = {name: source[name] for name in source.files}
        grid = (
            arrays["geometry_risk"].shape,
            float(arrays["x_min_m"]),
            float(arrays["y_min_m"]),
            float(arrays["resolution_m"]),
        )
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError("input BEV grids differ; spatially align them before temporal fusion")

        frame_has_pose = "world_from_frame" in arrays and not args.ignore_poses
        if pose_mode is None:
            pose_mode = frame_has_pose
        elif pose_mode != frame_has_pose:
            raise ValueError("some input frames have poses and others do not")

        result = memory.update(
            arrays["geometry_risk"],
            arrays["geometry_valid"],
            arrays["semantic_risk"],
            arrays["semantic_uncertainty"],
            arrays["semantic_observed"],
            world_from_frame=(arrays.get("world_from_frame") if frame_has_pose else None),
            x_min_m=float(arrays["x_min_m"]),
            y_min_m=float(arrays["y_min_m"]),
            resolution_m=float(arrays["resolution_m"]),
        )
        arrays.update(
            geometry_risk=result.geometry_risk,
            geometry_valid=result.geometry_valid,
            semantic_risk=result.semantic_risk,
            semantic_uncertainty=result.semantic_uncertainty,
            semantic_observed=result.semantic_observed,
            fused_risk=result.fused_risk,
            fused_valid=result.fused_valid,
            temporal_history_size=np.array(result.history_size),
            temporal_window_size=np.array(args.window_size),
        )
        frame_name = path.parent.name
        output_dir = args.output / frame_name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "fusion_layers.npz", **arrays)
        summary = {
            "index": index,
            "frame_name": frame_name,
            "history_size": result.history_size,
            "geometry_cells": int(np.count_nonzero(result.geometry_valid)),
            "semantic_cells": int(np.count_nonzero(result.semantic_observed)),
            "fused_cells": int(np.count_nonzero(result.fused_valid)),
            "pose_aligned": bool(pose_mode),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries.append(summary)
        print(
            f"[{index + 1}/{len(paths)}] {frame_name} history={result.history_size} "
            f"geometry_cells={summary['geometry_cells']} semantic_cells={summary['semantic_cells']}"
        )

    sequence_summary = {
        "frame_count": len(paths),
        "window_size": args.window_size,
        "geometry_aggregation": "median of valid observations over trailing window",
        "semantic_aggregation": "worst observed risk over trailing window",
        "alignment": (
            "historical local BEVs transformed into the current vehicle frame"
            if pose_mode
            else "inputs treated as pre-aligned local grids"
        ),
        "pose_compensation_disabled_for_ablation": args.ignore_poses,
        "frames": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sequence.json").write_text(
        json.dumps(sequence_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output={args.output} frames={len(paths)} window={args.window_size}")


if __name__ == "__main__":
    main()
