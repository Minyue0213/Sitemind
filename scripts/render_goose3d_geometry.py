"""Render real GOOSE-Ex LiDAR elevation and geometry-risk diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from sitemind.data import (
    discover_point_cloud_pairs,
    load_label_mapping,
    load_point_cloud,
    load_point_labels,
)
from sitemind.fusion import fuse_risk_maps, load_risk_values, semantic_ids_to_risk
from sitemind.geometry import (
    GeometryConfig,
    build_elevation_grid,
    compute_geometry_risk,
    rasterize_max_to_grid,
)


ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = np.array([30, 40, 49], dtype=np.uint8)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def risk_rgb(risk: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.empty((*risk.shape, 3), dtype=np.uint8)
    result[risk < 0.25] = (24, 183, 92)
    result[(risk >= 0.25) & (risk < 0.5)] = (166, 214, 46)
    result[(risk >= 0.5) & (risk < 0.75)] = (255, 193, 7)
    result[(risk >= 0.75) & (risk < 0.99)] = (255, 112, 40)
    result[risk >= 0.99] = (229, 57, 53)
    result[~valid] = UNKNOWN
    return result


def height_rgb(height: np.ndarray) -> np.ndarray:
    valid = np.isfinite(height)
    result = np.empty((*height.shape, 3), dtype=np.uint8)
    result[:] = UNKNOWN
    if not np.any(valid):
        return result
    lo, hi = np.percentile(height[valid], [3, 97])
    scale = np.clip((height - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    result[..., 0] = np.nan_to_num(32 + 213 * scale, nan=UNKNOWN[0]).astype(np.uint8)
    result[..., 1] = np.nan_to_num(77 + 148 * scale, nan=UNKNOWN[1]).astype(np.uint8)
    result[..., 2] = np.nan_to_num(118 + 105 * (1 - scale), nan=UNKNOWN[2]).astype(np.uint8)
    result[~valid] = UNKNOWN
    return result


def scalar_rgb(values: np.ndarray, valid: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(np.nan_to_num(values, nan=0.0) / maximum, 0.0, 1.0)
    result = np.zeros((*values.shape, 3), dtype=np.uint8)
    result[..., 0] = (35 + 220 * normalized).astype(np.uint8)
    result[..., 1] = (170 - 120 * normalized).astype(np.uint8)
    result[..., 2] = (220 - 170 * normalized).astype(np.uint8)
    result[~valid] = UNKNOWN
    return result


def panel(title: str, subtitle: str, array: np.ndarray, width: int = 720) -> Image.Image:
    header = 72
    source = Image.fromarray(np.flipud(array))
    height = width
    source = source.resize((width, height), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, height + header), "#121c26")
    canvas.paste(source, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 9), title, fill="white", font=font(24, bold=True))
    draw.text((18, 42), subtitle, fill="#aac0cf", font=font(16))
    draw.line((width - 70, height + 42, width - 70, height + 10), fill="white", width=3)
    draw.polygon(
        [(width - 70, height + 3), (width - 77, height + 15), (width - 63, height + 15)],
        fill="white",
    )
    draw.text((width - 57, height + 12), "+Y", fill="white", font=font(14))
    draw.line((width - 70, height + 42, width - 34, height + 42), fill="white", width=3)
    draw.polygon(
        [(width - 27, height + 42), (width - 39, height + 35), (width - 39, height + 49)],
        fill="white",
    )
    draw.text((width - 57, height + 48), "+X", fill="white", font=font(14))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--geometry-config", type=Path, default=ROOT / "configs" / "geometry.yaml")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml")
    parser.add_argument("--bounds", nargs=4, type=float, default=(-20.0, 20.0, -20.0, 20.0))
    parser.add_argument("--z-range", nargs=2, type=float, default=(-2.0, 5.0))
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "goose3d_geometry.png")
    parser.add_argument("--layers-output", type=Path)
    args = parser.parse_args()

    with args.geometry_config.open(encoding="utf-8") as handle:
        config = GeometryConfig(**yaml.safe_load(handle))
    config = replace(config, resolution_m=args.resolution, min_points_per_cell=1, step_radius_cells=2)
    pair = discover_point_cloud_pairs(args.dataset_root)[args.index]
    points = load_point_cloud(pair.scan_path)
    semantic_ids, _ = load_point_labels(pair.label_path, expected_points=len(points))
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
        & ~ego_footprint  # fixed self-filter from the known platform footprint
    )
    points = points[keep]
    semantic_ids = semantic_ids[keep]
    grid = build_elevation_grid(points[:, :3], config, tuple(args.bounds))
    geometry = compute_geometry_risk(grid, config)

    mapping = load_label_mapping(args.dataset_root / "goose_label_mapping.csv")
    class_risk = load_risk_values(args.risk_config)
    point_risk = semantic_ids_to_risk(semantic_ids, mapping, class_risk)
    semantic_risk = rasterize_max_to_grid(points[:, :2], point_risk, grid)
    semantic_valid = np.isfinite(semantic_risk)
    joint_valid = geometry.valid & semantic_valid
    fused = fuse_risk_maps(semantic_risk, geometry.risk)

    panels = [
        panel(
            "1. Real LiDAR elevation",
            f"{len(points):,} points in a 40 m x 40 m area; ego platform removed",
            height_rgb(grid.height_m),
        ),
        panel(
            "2. Estimated slope",
            "Blue = flat; red = steep surface or vertical obstacle",
            scalar_rgb(geometry.slope_deg, geometry.valid, config.critical_slope_deg),
        ),
        panel(
            "3. Geometry risk",
            "Interpretable risk from slope + local step height",
            risk_rgb(geometry.risk, geometry.valid),
        ),
        panel(
            "4. Fusion validation (oracle semantics)",
            "3D labels are used only as a reference here, not claimed as model output",
            risk_rgb(fused, joint_valid),
        ),
    ]
    gap, title_height, footer_height = 16, 100, 78
    tile_width, tile_height = panels[0].size
    canvas = Image.new(
        "RGB",
        (tile_width * 2 + gap * 3, title_height + tile_height * 2 + gap + footer_height),
        "#09121a",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "SiteMind real-LiDAR terrain understanding", fill="white", font=font(34, bold=True))
    draw.text(
        (24, 60),
        f"GOOSE-Ex validation · {pair.sequence} · frame {args.index}",
        fill="#a9bdcc",
        font=font(20),
    )
    for index, tile in enumerate(panels):
        row, column = divmod(index, 2)
        canvas.paste(tile, (gap + column * (tile_width + gap), title_height + row * (tile_height + gap)))

    footer_y = title_height + 2 * tile_height + gap + 18
    legend = [
        ("Low risk", (24, 183, 92)),
        ("Medium", (255, 193, 7)),
        ("High / blocked", (229, 57, 53)),
        ("No LiDAR evidence", tuple(int(v) for v in UNKNOWN)),
    ]
    x = 28
    for label, color in legend:
        draw.rounded_rectangle((x, footer_y, x + 34, footer_y + 34), radius=4, fill=color)
        draw.text((x + 46, footer_y + 5), label, fill="white", font=font(17))
        x += 330

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    layers_output = args.layers_output or args.output.with_suffix(".npz")
    np.savez_compressed(
        layers_output,
        elevation_m=grid.height_m,
        slope_deg=geometry.slope_deg,
        step_height_m=geometry.step_height_m,
        geometry_risk=geometry.risk,
        semantic_risk=semantic_risk,
        fused_risk=fused,
        geometry_valid=geometry.valid,
        fusion_valid=joint_valid,
    )
    metadata = {
        "frame_index": args.index,
        "scan": str(pair.scan_path),
        "points_in_roi": int(len(points)),
        "grid_shape": list(grid.height_m.shape),
        "geometry_coverage": float(geometry.valid.mean()),
        "joint_coverage": float(joint_valid.mean()),
        "oracle_semantics": True,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **metadata}))


if __name__ == "__main__":
    main()
