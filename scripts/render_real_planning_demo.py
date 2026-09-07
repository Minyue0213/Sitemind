"""Render a Chinese, non-technical route comparison from real LiDAR geometry."""

from __future__ import annotations

import argparse
import json
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


def risk_rgb(risk: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.empty((*risk.shape, 3), dtype=np.uint8)
    result[risk < 0.25] = (24, 183, 92)
    result[(risk >= 0.25) & (risk < 0.5)] = (166, 214, 46)
    result[(risk >= 0.5) & (risk < 0.75)] = (255, 193, 7)
    result[(risk >= 0.75) & (risk < 0.99)] = (255, 112, 40)
    result[risk >= 0.99] = (229, 57, 53)
    result[~valid] = (28, 39, 48)
    return result


def nearest_traversable(risk: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    rows, cols = np.indices(risk.shape)
    distances = (rows - target[0]) ** 2 + (cols - target[1]) ** 2
    distances[risk >= 0.99] = np.iinfo(np.int32).max
    if np.all(distances == np.iinfo(np.int32).max):
        raise RuntimeError("risk map has no traversable cell")
    return tuple(int(v) for v in np.unravel_index(np.argmin(distances), distances.shape))


def path_length(result: AStarResult, resolution: float) -> float:
    return resolution * sum(
        hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(result.path, result.path[1:])
    )


def path_exposure(result: AStarResult, risk: np.ndarray, resolution: float) -> float:
    return resolution * sum(float(risk[cell]) for cell in result.path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", type=Path)
    parser.add_argument("--bounds", nargs=4, type=float, default=(-20.0, 20.0, -20.0, 20.0))
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--start", nargs=2, type=float, default=(3.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--goal", nargs=2, type=float, default=(15.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--risk-weight", type=float, default=8.0)
    parser.add_argument("--evaluation-summary", type=Path)
    parser.add_argument("--fixed-summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/real_planning_demo.png"))
    args = parser.parse_args()

    layers = np.load(args.layers)
    risk = layers["geometry_risk"].astype(np.float32)
    valid = layers["geometry_valid"].astype(bool)
    risk[~valid] = 1.0
    x_min, _, y_min, _ = args.bounds

    def world_to_cell(xy: tuple[float, float]) -> tuple[int, int]:
        x, y = xy
        return (int((y - y_min) / args.resolution), int((x - x_min) / args.resolution))

    start = nearest_traversable(risk, world_to_cell(tuple(args.start)))
    goal = nearest_traversable(risk, world_to_cell(tuple(args.goal)))
    shortest = astar(risk, start, goal, risk_weight=0.0)
    safer = astar(risk, start, goal, risk_weight=args.risk_weight)
    if shortest is None or safer is None:
        raise RuntimeError("selected start and goal are not connected")

    shortest_length = path_length(shortest, args.resolution)
    safer_length = path_length(safer, args.resolution)
    shortest_exposure = path_exposure(shortest, risk, args.resolution)
    safer_exposure = path_exposure(safer, risk, args.resolution)
    distance_increase = safer_length / shortest_length - 1.0
    risk_reduction = 1.0 - safer_exposure / shortest_exposure
    evaluation = None
    fixed_evaluation = None
    if args.evaluation_summary:
        evaluation = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    if args.fixed_summary:
        fixed_evaluation = json.loads(args.fixed_summary.read_text(encoding="utf-8"))

    map_size = 900
    map_image = Image.fromarray(np.flipud(risk_rgb(risk, valid))).resize(
        (map_size, map_size), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (1700, 1080), "#081019")
    canvas.paste(map_image, (60, 130))
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 34), "SiteMind：真实工地上的风险感知路径规划", fill="#f4f8fb", font=font(46))
    draw.text(
        (62, 91),
        "输入：GOOSE-Ex LiDAR · 规划只使用坡度/台阶几何风险，不使用人工语义标签",
        fill="#9eb2c2",
        font=font(24),
    )

    rows, cols = risk.shape

    def screen_path(result: AStarResult) -> list[tuple[float, float]]:
        return [
            (60 + (col + 0.5) * map_size / cols, 130 + (rows - row - 0.5) * map_size / rows)
            for row, col in result.path
        ]

    draw.line(screen_path(shortest), fill="#00d9ff", width=10, joint="curve")
    draw.line(screen_path(safer), fill="white", width=10, joint="curve")
    for cell, color, label in ((start, "#00ff87", "起点"), (goal, "#ff43db", "目标")):
        x = 60 + (cell[1] + 0.5) * map_size / cols
        y = 130 + (rows - cell[0] - 0.5) * map_size / rows
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=color, outline="#081019", width=4)
        draw.text((x + 18, y - 17), label, fill="white", font=font(22))

    side_x = 1020
    draw.rounded_rectangle((side_x, 144, 1640, 338), radius=24, fill="#102333")
    draw.text((side_x + 30, 168), "青色：只求距离最短", fill="#00d9ff", font=font(28))
    draw.text((side_x + 30, 220), f"路线长度  {shortest_length:.2f} m", fill="white", font=font(28))
    draw.text((side_x + 30, 269), f"风险暴露  {shortest_exposure:.2f}", fill="#ff7067", font=font(28))

    draw.rounded_rectangle((side_x, 364, 1640, 558), radius=24, fill="#162b27")
    draw.text((side_x + 30, 388), "白色：SiteMind 安全路线", fill="white", font=font(28))
    draw.text((side_x + 30, 440), f"路线长度  {safer_length:.2f} m", fill="white", font=font(28))
    draw.text((side_x + 30, 489), f"风险暴露  {safer_exposure:.2f}", fill="#31d28c", font=font(28))

    draw.rounded_rectangle((side_x, 584, 1640, 752), radius=24, fill="#211d38")
    draw.text((side_x + 30, 609), "系统做出的取舍", fill="#b66cff", font=font(28))
    draw.text((side_x + 30, 659), f"多走 {safer_length - shortest_length:.2f} m（{100 * distance_increase:.1f}%）", fill="white", font=font(26))
    draw.text((side_x + 30, 704), f"风险暴露降低 {100 * risk_reduction:.1f}%", fill="#31d28c", font=font(30))

    draw.rounded_rectangle((side_x, 778, 1640, 994), radius=24, fill="#111e2a")
    draw.text((side_x + 30, 803), "407 帧全量压力测试", fill="white", font=font(27))
    if evaluation:
        successes = evaluation["status_counts"]["success"]
        comparable = evaluation["risk_comparable_frames"]
        reduced = evaluation["risk_reduced_frames"]
        fixed_line = (
            f"固定跨平台坐标任务：{fixed_evaluation['status_counts']['success']}/407 成功"
            if fixed_evaluation
            else "固定坐标任务受单帧覆盖与传感器朝向限制"
        )
        lines = [
            f"{successes}/407 帧有 ≥8 m 已观测连通区域",
            f"{comparable} 个可比帧中，{reduced} 个降低风险",
            f"中位：多走 {100 * evaluation['median_distance_overhead']:.1f}%，风险下降 {100 * evaluation['median_risk_reduction']:.1f}%",
            fixed_line,
        ]
    else:
        lines = [
            "绿色：较平坦、可通行",
            "黄色/红色：坡陡、台阶或障碍",
            "深灰：LiDAR 没有足够证据，禁止冒险",
            "安全路线主动绕开高风险区域",
        ]
    for index, line in enumerate(lines):
        draw.text((side_x + 32, 852 + index * 34), line, fill="#b5c5d0", font=font(21))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(
        f"output={args.output} shortest_length={shortest_length:.3f} safer_length={safer_length:.3f} "
        f"shortest_exposure={shortest_exposure:.3f} safer_exposure={safer_exposure:.3f}"
    )


if __name__ == "__main__":
    main()
