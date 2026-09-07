"""Audit platform motion and TF reference frames across an ALICE ROS1 bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from extract_alice_ros_frame import (
    CLOUD_TOPICS,
    IMAGE_TOPICS,
    collect_tf,
    nearest_message,
    pick_topic,
    register_bag_message_types,
)
from sitemind.fusion import planar_pose, resolve_frame_transform


def ranked_reference_frames(
    transforms: list[tuple[str, str, np.ndarray]], source_frame: str
) -> list[str]:
    """Return connected map-like frames in a deterministic preference order."""

    names = {name.lstrip("/") for edge in transforms for name in edge[:2]}
    source = source_frame.lstrip("/")

    def score(name: str) -> tuple[int, str]:
        lowered = name.lower()
        terminal = lowered.rsplit("/", 1)[-1]
        if terminal == "map":
            rank = 0
        elif terminal.endswith("_map"):
            rank = 1
        elif terminal == "odom":
            rank = 2
        elif terminal.endswith("_odom"):
            rank = 3
        elif terminal in {"world", "utm"}:
            rank = 4
        else:
            rank = 10
        return rank, lowered

    connected = []
    for name in sorted(names, key=score):
        if name == source or score(name)[0] >= 10:
            continue
        try:
            resolve_frame_transform(transforms, source, name)
        except ValueError:
            continue
        connected.append(name)
    return connected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--reference-frame")
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="number of evenly spaced camera timestamps; 0 audits every frame",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/alice_seq02_pose_audit.json")
    )
    args = parser.parse_args()
    if args.samples < 0:
        raise SystemExit("samples must be non-negative")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    window_ns = int(args.window_seconds * 1e9)
    records: list[dict[str, object]] = []
    chosen_reference = args.reference_frame
    available_candidates: list[str] = []

    with Reader(args.bag) as reader:
        register_bag_message_types(reader, typestore)
        image_topic = pick_topic(reader.connections, IMAGE_TOPICS)
        cloud_topic = pick_topic(reader.connections, CLOUD_TOPICS)
        image_connections = [c for c in reader.connections if c.topic == image_topic]
        timestamps = [timestamp for _, timestamp, _ in reader.messages(connections=image_connections)]
        if not timestamps:
            raise SystemExit("bag contains no camera frames")
        if args.samples and args.samples < len(timestamps):
            selected_indices = np.linspace(0, len(timestamps) - 1, args.samples).round().astype(int)
            selected = [timestamps[index] for index in np.unique(selected_indices)]
        else:
            selected = timestamps

        for index, camera_time in enumerate(selected):
            cloud_time, cloud_message = nearest_message(
                reader, cloud_topic, camera_time, window_ns, typestore
            )
            transforms = collect_tf(reader, camera_time, window_ns, typestore)
            cloud_frame = cloud_message.header.frame_id
            if chosen_reference is None:
                available_candidates = ranked_reference_frames(transforms, cloud_frame)
                if not available_candidates:
                    frames = sorted({name for edge in transforms for name in edge[:2]})
                    raise SystemExit(
                        "could not infer a map/odom/world reference frame; available TF frames: "
                        + ", ".join(frames)
                    )
                chosen_reference = available_candidates[0]
            world_from_cloud = resolve_frame_transform(
                transforms, cloud_frame, chosen_reference
            )
            pose = planar_pose(world_from_cloud)
            records.append(
                {
                    "index": index,
                    "camera_timestamp_ns": camera_time,
                    "pointcloud_timestamp_ns": cloud_time,
                    "pointcloud_delta_ms": (cloud_time - camera_time) / 1e6,
                    "x_m": float(pose[0, 2]),
                    "y_m": float(pose[1, 2]),
                    "yaw_rad": float(np.arctan2(pose[1, 0], pose[0, 0])),
                }
            )
            if (index + 1) % 25 == 0 or index + 1 == len(selected):
                print(f"poses {index + 1}/{len(selected)}")

    xy = np.array([[row["x_m"], row["y_m"]] for row in records], dtype=np.float64)
    yaw = np.unwrap(np.array([row["yaw_rad"] for row in records], dtype=np.float64))
    step_distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    displacement = np.linalg.norm(xy - xy[0], axis=1)
    summary = {
        "source_bag": args.bag.name,
        "camera_frame_count": len(timestamps),
        "audited_pose_count": len(records),
        "duration_seconds": (timestamps[-1] - timestamps[0]) / 1e9,
        "reference_frame": chosen_reference,
        "auto_reference_candidates": available_candidates,
        "start_xy_m": xy[0].tolist(),
        "end_xy_m": xy[-1].tolist(),
        "end_to_start_displacement_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "maximum_displacement_from_start_m": float(np.max(displacement)),
        "sampled_path_length_m": float(np.sum(step_distance)),
        "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "yaw_range_deg": float(np.degrees(np.max(yaw) - np.min(yaw))),
        "poses": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "poses"}, indent=2))


if __name__ == "__main__":
    main()
