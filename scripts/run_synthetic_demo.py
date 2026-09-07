"""Run the entire SiteMind core on a deterministic synthetic worksite."""

from __future__ import annotations

from math import hypot
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sitemind.fusion import fuse_risk_maps
from sitemind.geometry import GeometryConfig, build_elevation_grid, compute_geometry_risk
from sitemind.planning import AStarResult, astar


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "synthetic_demo.png"


def terrain_points(height: np.ndarray, resolution: float) -> np.ndarray:
    points = []
    offsets = ((-0.045, -0.035), (0.035, -0.025), (-0.015, 0.045), (0.04, 0.04))
    for row in range(height.shape[0]):
        for col in range(height.shape[1]):
            for dx, dy in offsets:
                points.append(
                    (
                        (col + 0.5) * resolution + dx,
                        (row + 0.5) * resolution + dy,
                        height[row, col],
                    )
                )
    return np.asarray(points, dtype=np.float32)


def risk_rgb(risk: np.ndarray) -> np.ndarray:
    risk = np.clip(risk, 0.0, 1.0)
    rgb = np.zeros((*risk.shape, 3), dtype=np.uint8)
    low = risk <= 0.5
    rgb[..., 0] = np.where(low, 2 * risk * 255, 255).astype(np.uint8)
    rgb[..., 1] = np.where(low, 180 + 2 * risk * 75, 2 * (1 - risk) * 255).astype(np.uint8)
    rgb[..., 2] = np.where(risk >= 0.99, 15, 35).astype(np.uint8)
    return rgb


def scalar_rgb(values: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    finite = values[np.isfinite(values)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    scale = np.nan_to_num((values - lo) / max(hi - lo, 1e-6), nan=0.0)
    base = np.array(color, dtype=np.float32)
    return (35 + scale[..., None] * (base - 35)).astype(np.uint8)


def path_length(path: list[tuple[int, int]], resolution: float) -> float:
    return resolution * sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def path_exposure(path: list[tuple[int, int]], risk: np.ndarray, resolution: float) -> float:
    return resolution * sum(float(risk[cell]) for cell in path)


def panel(
    title: str,
    array_rgb: np.ndarray,
    *,
    paths: tuple[AStarResult, AStarResult] | None = None,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
) -> Image.Image:
    scale = 8
    label_height = 34
    image = Image.fromarray(array_rgb).resize(
        (array_rgb.shape[1] * scale, array_rgb.shape[0] * scale), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (image.width, image.height + label_height), "#101820")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, fill="white")
    if paths is not None:
        shortest, safer = paths

        def xy(cell: tuple[int, int]) -> tuple[float, float]:
            return ((cell[1] + 0.5) * scale, label_height + (cell[0] + 0.5) * scale)

        draw.line([xy(cell) for cell in shortest.path], fill="#00e5ff", width=3)
        draw.line([xy(cell) for cell in safer.path], fill="#ffffff", width=3)
        assert start is not None and goal is not None
        for cell, fill in ((start, "#00ff66"), (goal, "#ff3df2")):
            x, y = xy(cell)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill, outline="black")
    return canvas


def main() -> None:
    rows, cols = 50, 80
    resolution = 0.2
    rr, cc = np.mgrid[:rows, :cols]

    height = 0.015 * rr
    height += 0.45 * np.exp(-(((rr - 25) / 7) ** 2 + ((cc - 34) / 8) ** 2))
    height[7:17, 56:61] += 0.65

    config = GeometryConfig(resolution_m=resolution)
    grid = build_elevation_grid(
        terrain_points(height, resolution),
        config,
        bounds_xy=(0.0, cols * resolution, 0.0, rows * resolution),
    )
    geometry = compute_geometry_risk(grid, config)

    semantic = np.zeros_like(geometry.risk)
    semantic[19:32, 23:46] = 0.62  # rough work zone across the direct route
    semantic[8:18, 44:51] = 1.0  # stationary machinery / hard obstacle
    fused = fuse_risk_maps(semantic, geometry.risk)

    start, goal = (25, 4), (25, 75)
    shortest = astar(fused, start, goal, risk_weight=0.0)
    safer = astar(fused, start, goal, risk_weight=8.0)
    if shortest is None or safer is None:
        raise RuntimeError("Synthetic scene unexpectedly has no valid route")

    images = [
        panel("1  LiDAR elevation grid", scalar_rgb(geometry.elevation_m, (225, 238, 255))),
        panel("2  Geometry risk: slope + step", risk_rgb(geometry.risk)),
        panel("3  Semantic risk", risk_rgb(semantic)),
        panel(
            "4  Fused risk | cyan=shortest, white=SiteMind",
            risk_rgb(fused),
            paths=(shortest, safer),
            start=start,
            goal=goal,
        ),
    ]
    gap = 12
    canvas = Image.new(
        "RGB",
        (images[0].width * 2 + gap, images[0].height * 2 + gap),
        "#263238",
    )
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (images[0].width + gap, 0))
    canvas.paste(images[2], (0, images[0].height + gap))
    canvas.paste(images[3], (images[0].width + gap, images[0].height + gap))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)

    print(f"output={OUTPUT}")
    print(
        "shortest: "
        f"length={path_length(shortest.path, resolution):.2f}m, "
        f"risk_exposure={path_exposure(shortest.path, fused, resolution):.2f}"
    )
    print(
        "sitemind: "
        f"length={path_length(safer.path, resolution):.2f}m, "
        f"risk_exposure={path_exposure(safer.path, fused, resolution):.2f}"
    )


if __name__ == "__main__":
    main()
