"""Explain the synchronized camera and LiDAR evidence as two parallel branches."""

from __future__ import annotations

import argparse
from math import hypot
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sitemind.data import load_label_mapping
from sitemind.fusion import load_risk_values, semantic_ids_to_risk
from sitemind.planning import AStarResult, astar


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)
PURPLE = np.array([142, 68, 173], dtype=np.uint8)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def only_png(directory: Path) -> Path:
    paths = list(directory.glob("*.png"))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one PNG in {directory}, found {len(paths)}")
    return paths[0]


def risk_rgb(risk: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.empty((*risk.shape, 3), dtype=np.uint8)
    result[risk < 0.25] = (24, 183, 92)
    result[(risk >= 0.25) & (risk < 0.5)] = (166, 214, 46)
    result[(risk >= 0.5) & (risk < 0.75)] = (255, 193, 7)
    result[(risk >= 0.75) & (risk < 0.99)] = (255, 112, 40)
    result[risk >= 0.99] = (229, 57, 53)
    if valid is not None:
        result[~valid] = (28, 39, 48)
    return result


def nearest_traversable(risk: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    rows, cols = np.indices(risk.shape)
    distances = (rows - target[0]) ** 2 + (cols - target[1]) ** 2
    distances[risk >= 0.99] = np.iinfo(np.int32).max
    return tuple(int(v) for v in np.unravel_index(np.argmin(distances), distances.shape))


def path_length(result: AStarResult, resolution: float) -> float:
    return resolution * sum(
        hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(result.path, result.path[1:])
    )


def path_exposure(result: AStarResult, risk: np.ndarray, resolution: float) -> float:
    return resolution * sum(float(risk[cell]) for cell in result.path)


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.BILINEAR)
    result = Image.new("RGB", size, "#1c2730")
    result.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("geometry_layers", type=Path)
    parser.add_argument("--risk-config", type=Path, default=Path("configs/risk_mapping.yaml"))
    parser.add_argument("--entropy-threshold", type=float, default=0.23529411764705882)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/multimodal_storyboard.png"))
    args = parser.parse_args()

    camera = Image.open(only_png(args.inputs / "camera")).convert("RGB")
    prediction = np.asarray(Image.open(only_png(args.inputs / "label_ids")))
    uncertainty = np.asarray(Image.open(only_png(args.inputs / "uncertainty")), dtype=np.float32) / 255.0
    target = np.asarray(Image.open(only_png(args.inputs / "ground_truth")))
    mapping = load_label_mapping(args.inputs / "goose_label_mapping.csv")
    class_risk = load_risk_values(args.risk_config)
    predicted_risk = semantic_ids_to_risk(prediction, mapping, class_risk)
    target_risk = semantic_ids_to_risk(target, mapping, class_risk)
    abstained = uncertainty >= args.entropy_threshold
    guarded_risk = predicted_risk.copy()
    guarded_risk[abstained] = 1.0
    camera_risk_rgb = risk_rgb(guarded_risk)
    camera_risk_rgb[abstained] = PURPLE
    hazardous = target_risk >= 0.75
    fine_fsr = float(np.mean(predicted_risk[hazardous] < target_risk[hazardous]))
    guarded_fsr = float(np.mean(guarded_risk[hazardous] < target_risk[hazardous]))

    layers = np.load(args.geometry_layers)
    geometry_risk = layers["geometry_risk"].astype(np.float32)
    geometry_valid = layers["geometry_valid"].astype(bool)
    geometry_risk[~geometry_valid] = 1.0
    start = nearest_traversable(geometry_risk, (80, 92))
    goal = nearest_traversable(geometry_risk, (80, 140))
    shortest = astar(geometry_risk, start, goal, risk_weight=0.0)
    safer = astar(geometry_risk, start, goal, risk_weight=8.0)
    if shortest is None or safer is None:
        raise RuntimeError("the synchronized LiDAR frame has no route between selected waypoints")
    shortest_length = path_length(shortest, args.resolution)
    safer_length = path_length(safer, args.resolution)
    shortest_exposure = path_exposure(shortest, geometry_risk, args.resolution)
    safer_exposure = path_exposure(safer, geometry_risk, args.resolution)
    reduction = 1.0 - safer_exposure / shortest_exposure

    small_panel_size = (370, 370)
    route_panel_size = (520, 520)
    camera_panel = fit_panel(camera, small_panel_size)
    risk_panel = fit_panel(Image.fromarray(camera_risk_rgb), small_panel_size)
    route_panel = Image.fromarray(np.flipud(risk_rgb(geometry_risk, geometry_valid))).resize(
        route_panel_size, Image.Resampling.NEAREST
    )
    route_draw = ImageDraw.Draw(route_panel)
    rows, cols = geometry_risk.shape

    def route_points(result: AStarResult) -> list[tuple[float, float]]:
        return [
            (
                (col + 0.5) * route_panel_size[0] / cols,
                (rows - row - 0.5) * route_panel_size[1] / rows,
            )
            for row, col in result.path
        ]

    route_draw.line(route_points(shortest), fill="#00d9ff", width=7, joint="curve")
    route_draw.line(route_points(safer), fill="white", width=7, joint="curve")

    canvas = Image.new("RGB", (1800, 1160), "#081019")
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 30), "同一时刻，两条并行的安全证据链", fill="#f4f8fb", font=font(44))
    draw.text(
        (66, 86),
        "GOOSE-Ex ALICE 挖掘机 · 同一时间戳 · 相机负责识别“是什么”，LiDAR 负责测量“在哪里、地形怎样”",
        fill="#9fb3c2",
        font=font(22),
    )

    # Camera branch: two views from one camera frame.
    draw.rounded_rectangle((64, 132, 872, 662), radius=24, fill="#121d27", outline="#7446a7", width=3)
    draw.text((92, 153), "A  相机语义证据", fill="#c688ff", font=font(31))
    draw.text((92, 198), "识别水、土、机器、人员等类别；不确定时主动保守", fill="#b6c6d1", font=font(21))
    canvas.paste(camera_panel, (88, 242))
    canvas.paste(risk_panel, (478, 242))
    draw.text((153, 619), "原始相机画面", fill="white", font=font(21))
    draw.text((516, 619), "语义风险（紫色 = 不确定）", fill="white", font=font(21))
    draw.text((447, 391), "→", fill="#c688ff", font=font(36))

    # LiDAR branch: a bird's-eye metric map, not another camera image.
    draw.rounded_rectangle((912, 132, 1736, 662), radius=24, fill="#121d27", outline="#168d70", width=3)
    draw.text((940, 153), "B  LiDAR 几何证据", fill="#43dda6", font=font(31))
    draw.text((940, 198), "点云变成俯视 BEV 地图；计算高度、坡度、台阶和路线", fill="#b6c6d1", font=font(21))
    canvas.paste(route_panel, (940, 232))
    draw.text((1480, 292), "俯视图", fill="#43dda6", font=font(22))
    draw.text((1480, 335), "青：最短路线", fill="#00d9ff", font=font(20))
    draw.text((1480, 375), "白：更安全路线", fill="white", font=font(20))
    draw.text((1480, 415), "红：不可通行", fill="#ff5753", font=font(20))
    draw.text((1480, 455), "它不是相机画面", fill="#ffca6a", font=font(20))
    draw.text((1480, 490), "也不是第 A 栏的", fill="#ffca6a", font=font(20))
    draw.text((1480, 525), "像素级叠加", fill="#ffca6a", font=font(20))

    # Explicitly show the present decision-level merge and the next calibrated merge.
    draw.line((468, 674, 468, 723), fill="#c688ff", width=5)
    draw.line((1324, 674, 1324, 723), fill="#43dda6", width=5)
    draw.line((468, 723, 1324, 723), fill="#4aaeff", width=5)
    draw.line((896, 723, 896, 755), fill="#4aaeff", width=5)
    draw.rounded_rectangle((290, 755, 1502, 858), radius=22, fill="#13233a", outline="#4aaeff", width=3)
    draw.text((329, 775), "当前汇合方式：决策级", fill="#63b8ff", font=font(27))
    draw.text(
        (329, 817),
        "LiDAR 负责规划；相机若发现高风险或不确定，则触发绕行、减速或停车。两者尚未投影到同一网格。",
        fill="white",
        font=font(21),
    )

    cards = [
        (
            "视觉兜底",
            f"本帧危险漏判 {100 * fine_fsr:.1f}% → {100 * guarded_fsr:.1f}%",
            f"{100 * abstained.mean():.1f}% 不确定区域交给 LiDAR / 减速停车",
            "#1b1930",
        ),
        (
            "路线取舍",
            f"最短 {shortest_length:.2f} m → 安全 {safer_length:.2f} m",
            f"多走 {safer_length - shortest_length:.2f} m，风险暴露降低 {100 * reduction:.1f}%",
            "#132b27",
        ),
    ]
    for index, (title, first, second, color) in enumerate(cards):
        x = 64 + index * 858
        draw.rounded_rectangle((x, 886, x + 816, 1065), radius=24, fill=color)
        draw.text((x + 30, 908), title, fill="#b66cff" if index == 0 else "#31d28c", font=font(27))
        draw.text((x + 30, 958), first, fill="white", font=font(26))
        draw.text((x + 30, 1008), second, fill="#b5c5d0", font=font(21))

    draw.text(
        (66, 1100),
        "下一步：从同一原始 ROS bag 取得相机内参和 TF 外参，把语义投影到真实点云，生成统一语义—几何 BEV。",
        fill="#7f96a7",
        font=font(19),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"output={args.output} fine_fsr={fine_fsr:.6f} guarded_fsr={guarded_fsr:.6f} risk_reduction={reduction:.6f}")


if __name__ == "__main__":
    main()
