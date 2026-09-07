"""Render the main Sequence02 demo: camera, calibrated evidence, and safe route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sitemind.planning import astar
from sitemind.planning.camera_goal import (
    visible_ground, select_camera_goal, route_image_samples, transform_goal_point,
)
from sitemind.planning.vehicle_origin import metric_cell, plan_from_vehicle_origin
from sitemind.fusion.projection import CameraCalibration


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
    result = Image.new("RGB", size, "#1c2730")
    source = image.convert("RGB")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
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


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], *,
                fill: str, width: int, dash_px: float = 10.0, gap_px: float = 7.0) -> None:
    """Draw a dashed polyline so an unmeasured connector cannot look terrain-verified."""
    for start, end in zip(points, points[1:]):
        start_xy = np.asarray(start, dtype=np.float64)
        delta = np.asarray(end, dtype=np.float64) - start_xy
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            continue
        direction = delta / length
        offset = 0.0
        while offset < length:
            dash_end = min(offset + dash_px, length)
            a = start_xy + direction * offset
            b = start_xy + direction * dash_end
            draw.line((tuple(a), tuple(b)), fill=fill, width=width)
            offset += dash_px + gap_px


def nearest_traversable(risk: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    traversable = risk < 0.99
    if not np.any(traversable):
        raise RuntimeError("frame has no traversable cells")
    rows, cols = np.indices(risk.shape)
    distance = (rows - target[0]) ** 2 + (cols - target[1]) ** 2
    distance[~traversable] = np.iinfo(np.int32).max
    return tuple(int(value) for value in np.unravel_index(np.argmin(distance), risk.shape))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("temporal_root", type=Path)
    parser.add_argument("camera_root", type=Path)
    parser.add_argument("projection_root", type=Path)
    parser.add_argument("sequence_metadata", type=Path)
    parser.add_argument("--start", nargs=2, type=float, default=(4.0, 1.0))
    parser.add_argument("--goal", nargs=2, type=float, default=(12.0, 3.0))
    parser.add_argument("--goal-pixel", nargs=2, type=float,
                        help="Select a measured ground target in original camera pixels")
    parser.add_argument("--goal-frame", type=int,
                        help="Frame used to select a world-fixed goal for a sequence")
    parser.add_argument("--chassis-forward", nargs=2, type=float,
                        help="Verified chassis-forward vector in the point-cloud frame")
    parser.add_argument("--frame", type=int, help="Render one frame by zero-based sequence index")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame for a sequence preview")
    parser.add_argument("--count", type=int, help="Maximum frames to render from --start-frame")
    parser.add_argument("--risk-weight", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sitemind_sequence02_demo.gif"))
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("fps must be positive")
    if args.start_frame < 0 or (args.count is not None and args.count <= 0):
        raise SystemExit("start frame must be non-negative and count must be positive")

    metadata = json.loads(args.sequence_metadata.read_text(encoding="utf-8"))
    by_name = {item["frame_name"]: item for item in metadata["frames"]}
    all_layer_paths = sorted(args.temporal_root.glob("frame_*/fusion_layers.npz"))
    if not all_layer_paths:
        raise SystemExit(f"no temporal frames found below {args.temporal_root}")
    first_timestamp = int(by_name[all_layer_paths[0].parent.name]["camera_timestamp_ns"])
    if args.goal_frame is not None and args.goal_pixel is None:
        raise SystemExit("--goal-frame requires --goal-pixel")
    if args.goal_pixel is not None and args.frame is None and args.goal_frame is None:
        raise SystemExit("Use --goal-frame to lock a sequence target, or --frame for a single-frame preview")
    if args.goal_frame is not None and not 0 <= args.goal_frame < len(all_layer_paths):
        raise SystemExit("goal frame index outside available sequence")
    if args.frame is not None:
        if not 0 <= args.frame < len(all_layer_paths):
            raise SystemExit("frame index outside available sequence")
        layer_paths = [all_layer_paths[args.frame]]
    else:
        layer_paths = all_layer_paths[args.start_frame:]
        if args.count is not None:
            layer_paths = layer_paths[:args.count]
        if not layer_paths:
            raise SystemExit("requested sequence range contains no frames")

    world_goal = None
    world_goal_selection = None
    if args.goal_frame is not None:
        selection_path = all_layer_paths[args.goal_frame]
        selection_name = selection_path.parent.name
        with np.load(selection_path) as temporal:
            selection_risk = temporal["fused_risk"].astype(np.float32)
            selection_valid = temporal["geometry_valid"].astype(bool)
        selection_risk[~selection_valid] = 1.0
        with np.load(args.camera_root / selection_name / "calibrated_frame.npz") as raw:
            if "world_from_pointcloud" not in raw:
                raise SystemExit("Selected frame lacks the world pose required to lock a fixed target")
            width, height = (int(value) for value in raw["image_size"])
            calibration = CameraCalibration(
                raw["projection"], raw["lidar_to_camera"], width, height,
                intrinsic=raw["intrinsic"], distortion=raw["distortion"],
            )
            with np.load(args.projection_root / selection_name / "fusion_layers.npz") as current:
                correspondence = visible_ground(raw["points_xyz"], calibration, dict(current))
            try:
                target_index = select_camera_goal(
                    correspondence, tuple(args.goal_pixel), selection_risk,
                    forward_xy=tuple(args.chassis_forward) if args.chassis_forward else None,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            selected_local = correspondence.xyz[target_index].astype(np.float64)
            world_goal = transform_goal_point(raw["world_from_pointcloud"], selected_local)
            world_goal_selection = {
                "selection_frame": selection_name,
                "selection_frame_index": args.goal_frame,
                "requested_pixel": list(args.goal_pixel),
                "selected_pixel": correspondence.pixels[target_index].tolist(),
                "selected_local_xyz_m": selected_local.tolist(),
                "world_xyz_m": world_goal.tolist(),
            }

    panel_size = (350, 350)
    panel_x = (25, 405, 785, 1165, 1545)
    frames: list[Image.Image] = []
    route_count = 0
    frame_reports: list[dict[str, object]] = []

    for index, layer_path in enumerate(layer_paths):
        frame_name = layer_path.parent.name
        camera_path = args.camera_root / frame_name / "camera.png"
        projection_path = args.projection_root / frame_name / "projection_overlay.png"
        if not camera_path.exists() or not projection_path.exists():
            raise SystemExit(f"missing camera or projection image for {frame_name}")

        with np.load(layer_path) as layers:
            grid_layers = {name: layers[name] for name in layers.files}
            semantic = layers["semantic_risk"].astype(np.float32)
            semantic_valid = layers["semantic_observed"].astype(bool)
            geometry = layers["geometry_risk"].astype(np.float32)
            geometry_valid = layers["geometry_valid"].astype(bool)
            fused = layers["fused_risk"].astype(np.float32)
            resolution = float(layers["resolution_m"])
            x_min = float(layers["x_min_m"])
            y_min = float(layers["y_min_m"])
        valid = geometry_valid
        planning_risk = fused.copy()
        planning_risk[~valid] = 1.0

        def cell(xy: tuple[float, float]) -> tuple[int, int]:
            return (int((xy[1] - y_min) / resolution), int((xy[0] - x_min) / resolution))

        correspondence = None
        target_pixel = None
        target_local = None
        goal = None
        vehicle_plan = None
        if args.goal_pixel is not None:
            # Current, not temporally accumulated, points establish image visibility.
            with np.load(args.camera_root / frame_name / "calibrated_frame.npz") as raw:
                width, height = (int(v) for v in raw["image_size"])
                calibration = CameraCalibration(
                    raw["projection"], raw["lidar_to_camera"], width, height,
                    intrinsic=raw["intrinsic"], distortion=raw["distortion"],
                )
                with np.load(args.projection_root / frame_name / "fusion_layers.npz") as current:
                    correspondence = visible_ground(raw["points_xyz"], calibration, dict(current))
                if world_goal is not None:
                    if "world_from_pointcloud" not in raw:
                        raise SystemExit(f"{frame_name} lacks the world pose required to track the target")
                    target_local = transform_goal_point(
                        np.linalg.inv(raw["world_from_pointcloud"]), world_goal
                    )
                    if len(correspondence.xyz):
                        distances = np.linalg.norm(correspondence.xyz - target_local, axis=1)
                        visible_index = int(np.argmin(distances))
                        if distances[visible_index] <= 0.5:
                            target_pixel = correspondence.pixels[visible_index]
                else:
                    try:
                        selected_index = select_camera_goal(
                            correspondence, tuple(args.goal_pixel), planning_risk,
                            forward_xy=tuple(args.chassis_forward) if args.chassis_forward else None,
                        )
                    except ValueError as exc:
                        raise SystemExit(str(exc)) from exc
                    target_local = correspondence.xyz[selected_index]
                    target_pixel = correspondence.pixels[selected_index]

            start = metric_cell(
                (0.0, 0.0), planning_risk.shape, x_min_m=x_min,
                y_min_m=y_min, resolution_m=resolution,
            )
            try:
                goal = metric_cell(
                    tuple(target_local[:2]), planning_risk.shape, x_min_m=x_min,
                    y_min_m=y_min, resolution_m=resolution,
                )
            except ValueError:
                goal = None
            if goal is not None and np.isfinite(planning_risk[goal]) and planning_risk[goal] < 0.99:
                vehicle_plan = plan_from_vehicle_origin(
                    planning_risk, goal, x_min_m=x_min, y_min_m=y_min,
                    resolution_m=resolution, self_exclusion_bounds_m=(-3.0, 4.0, -1.6, 1.6),
                    risk_weight=args.risk_weight,
                    forward_xy=tuple(args.chassis_forward) if args.chassis_forward else None,
                )
            route = vehicle_plan.route if vehicle_plan is not None else None
        else:
            start = nearest_traversable(planning_risk, cell(tuple(args.start)))
            goal = nearest_traversable(planning_risk, cell(tuple(args.goal)))
            route = astar(planning_risk, start, goal, risk_weight=args.risk_weight)
        route_count += route is not None

        camera_source = Image.open(camera_path).convert('RGB')
        projection_source = Image.open(projection_path).convert('RGB')
        supported_pixels = []
        if correspondence is not None:
            supported_pixels = route_image_samples(
                correspondence, vehicle_plan.verified_path if vehicle_plan else [])
            for source in (camera_source, projection_source):
                overlay = ImageDraw.Draw(source)
                for pixel in supported_pixels:
                    if pixel is not None:
                        x, y = pixel
                        overlay.ellipse((x-7, y-7, x+7, y+7), fill='white', outline='#081019', width=2)
                if target_pixel is not None:
                    x, y = target_pixel
                    overlay.ellipse((x-22, y-22, x+22, y+22), fill='#ff55d5', outline='white', width=4)
                    overlay.text((x+28, y-28), 'A', font=font(45), fill='white',
                                 stroke_width=3, stroke_fill='#081019')
            report = {
                'frame': frame_name,
                'target_visible_in_camera': target_pixel is not None,
                'target_pixel': target_pixel.tolist() if target_pixel is not None else None,
                'target_local_xyz_m': target_local.tolist(),
                'target_cell': list(goal) if goal is not None else None,
                'start_cell': list(start),
                'vehicle_reference_xy_m': [0.0, 0.0],
                'start_is_near_field_proxy': False,
                'self_exclusion_bounds_m': [-3.0, 4.0, -1.6, 1.6],
                'connector_is_terrain_verified': False,
                'chassis_forward_xy': args.chassis_forward,
                'chassis_forward_verified': args.chassis_forward is not None,
                'connector_cells': len(vehicle_plan.connector_path) if vehicle_plan else 0,
                'terrain_verified_route_cells': len(vehicle_plan.verified_path) if vehicle_plan else 0,
                'route_length_m': float(sum(np.linalg.norm(np.subtract(b, a)) * resolution
                                            for a, b in zip(route.path, route.path[1:]))) if route else None,
                'route_cells': len(route.path) if route else 0,
                'camera_supported_route_cells': sum(p is not None for p in supported_pixels),
                'world_fixed_tracking': world_goal is not None,
                'note': ('Fixed world target; dotted pixels are current measured route-cell samples.'
                         if world_goal is not None else
                         'Single-frame preview; dotted pixels are measured route-cell samples.'),
            }
            frame_reports.append(report)
            if world_goal is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_suffix('.json').write_text(
                    json.dumps(report, indent=2), encoding='utf-8'
                )
                camera_source.save(args.output.with_name(args.output.stem + '_camera.png'))
                print(json.dumps(report))
        camera = fit(camera_source, panel_size)
        projection = fit(projection_source, panel_size)
        semantic_map = Image.fromarray(
            np.flipud(risk_rgb(np.nan_to_num(semantic, nan=1.0), semantic_valid))
        ).resize(panel_size, Image.Resampling.NEAREST)
        geometry_map = Image.fromarray(
            np.flipud(risk_rgb(np.nan_to_num(geometry, nan=1.0), geometry_valid))
        ).resize(panel_size, Image.Resampling.NEAREST)
        fused_map = Image.fromarray(np.flipud(risk_rgb(fused, valid))).resize(
            panel_size, Image.Resampling.NEAREST
        )
        map_draw = ImageDraw.Draw(fused_map)
        rows, cols = fused.shape

        def map_point(grid_cell: tuple[int, int]) -> tuple[float, float]:
            row, col = grid_cell
            return (
                (col + 0.5) * panel_size[0] / cols,
                (rows - row - 0.5) * panel_size[1] / rows,
            )

        def metric_point(x_m: float, y_m: float) -> tuple[float, float]:
            return map_point(
                (int((y_m - y_min) / resolution), int((x_m - x_min) / resolution))
            )

        if route is not None:
            if vehicle_plan is not None:
                dashed_line(map_draw, [map_point(item) for item in vehicle_plan.connector_path],
                            fill="#00e5ff", width=5)
                map_draw.line([map_point(item) for item in vehicle_plan.verified_path],
                              fill="white", width=7, joint="curve")
            else:
                map_draw.line([map_point(item) for item in route.path], fill="white", width=7, joint="curve")
        start_label = "挖掘机" if correspondence is not None else "起"
        markers = [(map_point(start), start_label, "#00e5ff")]
        if goal is not None:
            markers.append((map_point(goal), "A" if correspondence is not None else "终", "#ff55d5"))
        for point, label, color in markers:
            x, y = point
            map_draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="white", width=2)
            map_draw.text((x + 10, y - 11), label, fill="white", font=font(16))
        if args.chassis_forward is not None:
            direction = np.asarray(args.chassis_forward, dtype=np.float64)
            direction /= np.linalg.norm(direction)
            origin_px = metric_point(0.0, 0.0)
            front_px = metric_point(*(direction * 2.5))
            map_draw.line((origin_px, front_px), fill="#00e5ff", width=4)
            map_draw.polygon((front_px,
                              (front_px[0] - 7, front_px[1] - 4),
                              (front_px[0] - 7, front_px[1] + 4)), fill="#00e5ff")
        if correspondence is not None and goal is not None:
            for bev in (semantic_map, geometry_map):
                bd = ImageDraw.Draw(bev)
                x, y = map_point(goal)
                bd.ellipse((x-9, y-9, x+9, y+9), fill='#ff55d5', outline='white', width=2)
                bd.text((x+10, y-11), 'A', fill='white', font=font(16))

        canvas = Image.new("RGB", (1920, 620), "#081019")
        draw = ImageDraw.Draw(canvas)
        elapsed = (int(by_name[frame_name]["camera_timestamp_ns"]) - first_timestamp) / 1e9
        draw.text((25, 18), "SiteMind · 下一作业点安全移位预演", fill="white", font=font(30))
        draw.text(
            (25, 58),
            f"Sequence02  t={elapsed:05.1f}s · " + (
                ("目标 A 当前可见，位置已锁定" if target_pixel is not None else
                 "目标 A 已锁定在地图，当前相机未直接看见") if world_goal is not None else
                "单帧选点样例：五图中的 A 为同一目标地面位置" if correspondence is not None else
                f"{index + 1:03d}/{len(layer_paths):03d} · 最近{int(grid_layers.get('temporal_window_size', 1))}帧融合"
            ),
            fill="#a9bdca",
            font=font(18),
        )
        titles = (
            ("① 相机选择下一作业点",
             "粉色 A：真实地面目标" if target_pixel is not None else "目标已固定，不跟随车辆移动")
            if correspondence is not None else ("① 现场相机", "观察工地场景"),
            ("② 标定投影", "相机语义贴到真实点云"),
            ("③ 相机语义 BEV", "识别材质、设备与障碍"),
            ("④ LiDAR 几何 BEV", "测量高度、坡度与台阶"),
            ("⑤ 融合 BEV + 局部规划", "青色虚线为近场 · 白线经过已观测地面"),
        )
        images = (camera, projection, semantic_map, geometry_map, fused_map)
        for x, image, (title, subtitle) in zip(panel_x, images, titles):
            draw.rounded_rectangle((x, 98, x + 350, 530), radius=16, fill="#121d27")
            draw.text((x + 12, 108), title, fill="white", font=font(19))
            draw.text((x + 12, 137), subtitle, fill="#a9bdca", font=font(13))
            canvas.paste(image, (x, 170))

        status = "局部路线建议（未执行）" if route is not None else "未找到路线：不建议移位"
        status_color = "#35df9a" if route is not None else "#ffbd45"
        draw.text(
            (25, 552),
            ("目标 A 由相机选定并固定；路线从挖掘机当前位置开始；青色虚线跨越传感器自遮挡近场；白线只经过已观测地面"
             if correspondence is not None else "相机识别材质与障碍，LiDAR估计地形风险，融合规划综合两类证据"),
            fill="#d8e4ec",
            font=font(15),
        )
        draw.rounded_rectangle((1545, 545, 1895, 590), radius=10, fill="#17273a")
        draw.text((1560, 556), status, fill=status_color, font=font(17))
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if world_goal_selection is not None:
        sequence_report = {
            "goal_selection": world_goal_selection,
            "world_fixed_tracking": True,
            "rendered_frames": len(frame_reports),
            "route_frames": route_count,
            "stop_frames": len(frame_reports) - route_count,
            "frames": frame_reports,
            "note": (
                "The camera selects one measured ground point once. Its world position remains "
                "fixed while every frame replans from the current excavator reference."
            ),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(sequence_report, indent=2), encoding="utf-8"
        )
    if args.frame is not None and args.output.suffix.lower() == '.png':
        frames[0].convert('RGB').save(args.output)
        print(f"output={args.output}")
        return
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(
        f"output={args.output} frames={len(frames)} routes={route_count} "
        f"stops={len(frames) - route_count} fps={args.fps}"
    )


if __name__ == "__main__":
    main()
