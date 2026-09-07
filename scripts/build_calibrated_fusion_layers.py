"""Build aligned geometry, camera-semantic, and fused BEV layers."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from sitemind.data import load_label_mapping
from sitemind.fusion import (
    CameraCalibration,
    fuse_semantic_bev_with_geometry,
    load_risk_values,
    project_lidar_to_image,
    project_semantic_risk_to_bev,
    sample_image_at_projection,
    semantic_ids_to_risk,
)
from sitemind.geometry import GeometryConfig, build_elevation_grid, compute_geometry_risk


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibrated_frame", type=Path)
    parser.add_argument("camera_image", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("uncertainty", type=Path)
    parser.add_argument("label_mapping", type=Path)
    parser.add_argument(
        "--prediction-source-image",
        type=Path,
        help="RGB image used for prediction; when supplied it must match the extracted bag image",
    )
    parser.add_argument("--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml")
    parser.add_argument("--geometry-config", type=Path, default=ROOT / "configs" / "geometry.yaml")
    parser.add_argument("--bounds", nargs=4, type=float, default=(-20.0, 20.0, -20.0, 20.0))
    parser.add_argument("--z-range", nargs=2, type=float, default=(-2.0, 5.0))
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "alice_seq02_fusion")
    args = parser.parse_args()

    raw = np.load(args.calibrated_frame)
    points = raw["points_xyz"].astype(np.float32)
    width, height = (int(v) for v in raw["image_size"])
    calibration = CameraCalibration(
        raw["projection"],
        raw["lidar_to_camera"],
        width,
        height,
        intrinsic=raw["intrinsic"],
        distortion=raw["distortion"],
        distortion_model=str(raw["distortion_model"]) if "distortion_model" in raw else "plumb_bob",
    )
    prediction = np.asarray(Image.open(args.prediction))
    uncertainty = np.asarray(Image.open(args.uncertainty), dtype=np.float32) / 255.0
    camera = Image.open(args.camera_image).convert("RGB")
    if camera.size != (width, height):
        raise ValueError(f"camera image size {camera.size} does not match calibration {(width, height)}")
    source_image_mae = None
    if args.prediction_source_image:
        prediction_source = Image.open(args.prediction_source_image).convert("RGB")
        if prediction_source.size != camera.size:
            raise ValueError("prediction source and extracted bag image have different sizes")
        source_image_mae = float(
            np.mean(
                np.abs(
                    np.asarray(prediction_source, dtype=np.float32)
                    - np.asarray(camera, dtype=np.float32)
                )
            )
        )
        if source_image_mae > 1.0:
            raise ValueError(
                f"prediction source does not match bag image (mean absolute RGB error {source_image_mae:.3f})"
            )
    mapping = load_label_mapping(args.label_mapping)
    class_risk = load_risk_values(args.risk_config)
    pixel_risk = semantic_ids_to_risk(prediction, mapping, class_risk)

    with args.geometry_config.open(encoding="utf-8") as handle:
        config = GeometryConfig(**yaml.safe_load(handle))
    config = replace(config, resolution_m=args.resolution, min_points_per_cell=1, step_radius_cells=2)
    x_min, x_max, y_min, y_max = args.bounds
    z_min, z_max = args.z_range
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
    grid = build_elevation_grid(points, config, tuple(args.bounds))
    geometry = compute_geometry_risk(grid, config)
    semantic = project_semantic_risk_to_bev(
        points,
        pixel_risk,
        uncertainty,
        calibration,
        grid,
        uncertainty_weight=args.uncertainty_weight,
    )
    fused, fused_valid = fuse_semantic_bev_with_geometry(
        semantic, geometry.risk, geometry.valid
    )

    projection = project_lidar_to_image(points, calibration)
    point_risk = sample_image_at_projection(pixel_risk, projection)
    overlay = camera.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    palette = np.array(
        [[24, 183, 92], [166, 214, 46], [255, 193, 7], [255, 112, 40], [229, 57, 53]]
    )
    bins = np.clip((point_risk * 4).astype(int), 0, 4)
    stride = max(1, len(projection.pixels_xy) // 50000)
    for (x, y), color in zip(projection.pixels_xy[::stride], palette[bins[::stride]]):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=tuple(int(v) for v in color) + (190,))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output_dir / "projection_overlay.png")
    np.savez_compressed(
        args.output_dir / "fusion_layers.npz",
        elevation_m=grid.height_m,
        slope_deg=geometry.slope_deg,
        step_height_m=geometry.step_height_m,
        geometry_risk=geometry.risk,
        geometry_valid=geometry.valid,
        semantic_risk=semantic.risk,
        semantic_uncertainty=semantic.uncertainty,
        semantic_observed=semantic.observed,
        fused_risk=fused,
        fused_valid=fused_valid,
        x_min_m=grid.x_min_m,
        y_min_m=grid.y_min_m,
        resolution_m=grid.resolution_m,
        **(
            {"world_from_frame": raw["world_from_pointcloud"]}
            if "world_from_pointcloud" in raw
            else {}
        ),
    )
    metadata = {
        "points_in_roi": int(len(points)),
        "points_projected_to_camera": semantic.projected_point_count,
        "geometry_cells": int(np.count_nonzero(geometry.valid)),
        "semantic_cells": int(np.count_nonzero(semantic.observed)),
        "overlap_cells": int(np.count_nonzero(semantic.observed & geometry.valid)),
        "grid_shape": list(grid.height_m.shape),
        "bounds_xy_m": list(args.bounds),
        "resolution_m": args.resolution,
        "uncertainty_weight": args.uncertainty_weight,
        "prediction_source_rgb_mae": source_image_mae,
        "pose_aligned": "world_from_pointcloud" in raw,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
