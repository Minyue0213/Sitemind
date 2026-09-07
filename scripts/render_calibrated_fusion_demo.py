"""Render an honest before/after route comparison using calibrated fusion."""

from __future__ import annotations

import argparse
from math import hypot
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sitemind.planning import AStarResult, astar


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.BILINEAR)
    result = Image.new("RGB", size, "#1c2730")
    result.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return result


def risk_rgb(risk: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.empty((*risk.shape, 3), dtype=np.uint8)
    result[risk < 0.25] = (24, 183, 92)
    result[(risk >= 0.25) & (risk < 0.5)] = (166, 214, 46)
    result[(risk >= 0.5) & (risk < 0.75)] = (255, 193, 7)
    result[(risk >= 0.75) & (risk < 0.99)] = (255, 112, 40)
    result[risk >= 0.99] = (229, 57, 53)
    result[~valid] = (28, 39, 48)
    return result


def nearest_common_cell(
    geometry: np.ndarray,
    fused: np.ndarray,
    target: tuple[int, int],
) -> tuple[int, int]:
    common = (geometry < 0.99) & (fused < 0.99)
    rows, cols = np.indices(common.shape)
    distance = (rows - target[0]) ** 2 + (cols - target[1]) ** 2
    distance[~common] = np.iinfo(np.int32).max
    if not np.any(common):
        raise RuntimeError("there is no cell traversable in both maps")
    return tuple(int(v) for v in np.unravel_index(np.argmin(distance), common.shape))


def length(result: AStarResult, resolution: float) -> float:
    return resolution * sum(
        hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(result.path, result.path[1:])
    )


def exposure(result: AStarResult, risk: np.ndarray, resolution: float) -> float:
    return resolution * sum(float(risk[cell]) for cell in result.path)


def format_optional(value: float | None, suffix: str = "") -> str:
    return "不可达" if value is None else f"{value:.2f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", type=Path)
    parser.add_argument("camera", type=Path)
    parser.add_argument("projection_overlay", type=Path)
    parser.add_argument("--start", nargs=2, type=float, default=(4.0, 1.0))
    parser.add_argument("--goal", nargs=2, type=float, default=(12.0, 6.0))
    parser.add_argument("--risk-weight", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibrated_fusion_demo.png"))
    args = parser.parse_args()

    layers = np.load(args.layers)
    geometry = layers["geometry_risk"].astype(np.float32)
    geometry_valid = layers["geometry_valid"].astype(bool)
    semantic = layers["semantic_risk"].astype(np.float32)
    semantic_observed = layers["semantic_observed"].astype(bool)
    fused = layers["fused_risk"].astype(np.float32)
    resolution = float(layers["resolution_m"])
    x_min, y_min = float(layers["x_min_m"]), float(layers["y_min_m"])
    geometry[~geometry_valid] = 1.0
    fused[~geometry_valid] = 1.0  # planning remains inside LiDAR-observed terrain

    def cell(xy: tuple[float, float]) -> tuple[int, int]:
        return (int((xy[1] - y_min) / resolution), int((xy[0] - x_min) / resolution))

    start = nearest_common_cell(geometry, fused, cell(tuple(args.start)))
    goal = nearest_common_cell(geometry, fused, cell(tuple(args.goal)))
    geometry_route = astar(geometry, start, goal, risk_weight=args.risk_weight)
    fused_route = astar(fused, start, goal, risk_weight=args.risk_weight)

    map_size = 540
    geometry_map = Image.fromarray(np.flipud(risk_rgb(geometry, geometry_valid))).resize(
        (map_size, map_size), Image.Resampling.NEAREST
    )
    semantic_map = Image.fromarray(
        np.flipud(risk_rgb(np.nan_to_num(semantic, nan=1.0), semantic_observed))
    ).resize((map_size, map_size), Image.Resampling.NEAREST)
    fused_map = Image.fromarray(np.flipud(risk_rgb(fused, geometry_valid))).resize(
        (map_size, map_size), Image.Resampling.NEAREST
    )
    route_map = fused_map.copy()
    rows, cols = geometry.shape

    def map_path(result: AStarResult | None) -> list[tuple[float, float]]:
        if result is None:
            return []
        return [
            ((col + 0.5) * map_size / cols, (rows - row - 0.5) * map_size / rows)
            for row, col in result.path
        ]

    geometry_path = map_path(geometry_route)
    fused_path = map_path(fused_route)
    if geometry_path:
        ImageDraw.Draw(geometry_map).line(geometry_path, fill="#00d9ff", width=8, joint="curve")
    fused_draw = ImageDraw.Draw(route_map)
    if geometry_path:
        fused_draw.line(geometry_path, fill="#00d9ff", width=5, joint="curve")
    if fused_path:
        fused_draw.line(fused_path, fill="white", width=8, joint="curve")

    def map_point(grid_cell: tuple[int, int]) -> tuple[float, float]:
        row, col = grid_cell
        return ((col + 0.5) * map_size / cols, (rows - row - 0.5) * map_size / rows)

    def mark_endpoints(image: Image.Image) -> None:
        marker_draw = ImageDraw.Draw(image)
        for point, label, color in (
            (map_point(start), "起", "#00e5ff"),
            (map_point(goal), "终", "#ff55d5"),
        ):
            x, y = point
            marker_draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=color, outline="white", width=3)
            marker_draw.text((x + 13, y - 13), label, fill="white", font=font(18), stroke_width=2, stroke_fill="#081019")

    mark_endpoints(geometry_map)
    mark_endpoints(route_map)
    camera_panel = fit(Image.open(args.camera), (map_size, map_size))
    overlay_panel = fit(Image.open(args.projection_overlay), (map_size, map_size))

    semantic_for_exposure = np.where(semantic_observed, semantic, 0.0)
    before_semantic = (
        exposure(geometry_route, semantic_for_exposure, resolution)
        if geometry_route is not None
        else None
    )
    after_semantic = (
        exposure(fused_route, semantic_for_exposure, resolution) if fused_route is not None else None
    )
    before_length = length(geometry_route, resolution) if geometry_route is not None else None
    after_length = length(fused_route, resolution) if fused_route is not None else None
    route_subtitle = (
        "青=原路线，白=融合路线"
        if fused_route is not None
        else "未发现连续安全通道：停车/等待"
    )

    canvas = Image.new("RGB", (1880, 1515), "#081019")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 27), "相机语义 BEV + LiDAR 几何 BEV，怎样变成一张可规划地图？", fill="white", font=font(38))
    draw.text((42, 78), "ALICE sequence02 · 同次录制的 CameraInfo + TF 外参 · 灰色代表没有观测，不把未知当安全", fill="#9fb3c2", font=font(21))

    panels = [
        (40, 120, "① 相机看到现场", "RGB：模型识别泥地、机器等", camera_panel),
        (650, 120, "② 相机—LiDAR 对齐", "彩色点：LiDAR 点取得的像素风险", overlay_panel),
        (1260, 120, "③ 相机语义 BEV", "有深度的像素落入俯视格；其余为灰", semantic_map),
        (40, 760, "④ LiDAR 几何 BEV", "坡度、台阶形成几何风险；青为原路线", geometry_map),
        (650, 760, "⑤ 统一风险 BEV", "同一格综合语义和几何，取更保守风险", fused_map),
        (1260, 760, "⑥ 同一起终点重新规划", route_subtitle, route_map),
    ]
    for x, y, title, subtitle, panel_image in panels:
        h = panel_image.height + 82
        draw.rounded_rectangle((x, y, x + 580, y + h), radius=20, fill="#121d27")
        draw.text((x + 20, y + 12), title, fill="white", font=font(25))
        draw.text((x + 20, y + 47), subtitle, fill="#adc0cd", font=font(17))
        canvas.paste(panel_image, (x + 20, y + 74))

    draw.rounded_rectangle((40, 1395, 1840, 1478), radius=20, fill="#17273a")
    if fused_route is None:
        summary = (
            f"同一任务：几何路线 {format_optional(before_length, ' m')} → 融合路线不可达；"
            "当前帧执行保守停车/等待，下一帧继续更新"
        )
    elif geometry_route is None:
        summary = f"同一任务：几何路线不可达 → 融合路线 {format_optional(after_length, ' m')}"
    else:
        assert before_semantic is not None and after_semantic is not None
        reduction = 0.0 if before_semantic <= 1e-9 else 1.0 - after_semantic / before_semantic
        summary = (
            f"同一任务：几何路线 {before_length:.2f} m → 融合路线 {after_length:.2f} m；"
            f"相机所见风险暴露 {before_semantic:.2f} → {after_semantic:.2f}（降低 {100 * reduction:.1f}%）"
        )
    draw.text((68, 1421), summary, fill="white", font=font(22))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(
        f"output={args.output} geometry_length={before_length} fused_length={after_length} "
        f"semantic_exposure_before={before_semantic} semantic_exposure_after={after_semantic} "
        f"status={'route' if fused_route is not None else 'stop'}"
    )


if __name__ == "__main__":
    main()
