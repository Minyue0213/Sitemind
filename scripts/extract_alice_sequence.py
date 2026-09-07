"""Extract a short synchronized ALICE camera/LiDAR sequence from a ROS1 bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from extract_alice_ros_frame import (
    CAMERA_INFO_TOPICS,
    CLOUD_TOPICS,
    IMAGE_TOPICS,
    collect_tf,
    decode_image,
    decode_pointcloud_xyz,
    nearest_message,
    pick_topic,
    register_bag_message_types,
)
from sitemind.fusion import planar_pose, resolve_frame_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--window-seconds", type=float, default=0.4)
    parser.add_argument(
        "--reference-frame",
        help="fixed TF frame (for example map/odom) used to save a pose for every local BEV",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/alice_seq02_sequence"))
    args = parser.parse_args()
    if args.start_index < 0 or args.count <= 0 or args.stride <= 0:
        raise SystemExit("start-index must be non-negative; count and stride must be positive")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    window_ns = int(args.window_seconds * 1e9)
    sequence_metadata: list[dict[str, object]] = []
    with Reader(args.bag) as reader:
        register_bag_message_types(reader, typestore)
        camera_info_topic = pick_topic(reader.connections, CAMERA_INFO_TOPICS)
        image_topic = pick_topic(reader.connections, IMAGE_TOPICS)
        cloud_topic = pick_topic(reader.connections, CLOUD_TOPICS)
        image_connections = [c for c in reader.connections if c.topic == image_topic]
        image_timestamps = [timestamp for _, timestamp, _ in reader.messages(connections=image_connections)]
        selected = image_timestamps[
            args.start_index : args.start_index + args.count * args.stride : args.stride
        ]
        if not selected:
            raise SystemExit("no camera frames selected")

        for index, target_ns in enumerate(selected):
            camera_time, camera_message = nearest_message(
                reader, image_topic, target_ns, window_ns, typestore
            )
            info_time, info_message = nearest_message(
                reader, camera_info_topic, target_ns, window_ns, typestore
            )
            cloud_time, cloud_message = nearest_message(
                reader, cloud_topic, target_ns, window_ns, typestore
            )
            transforms = collect_tf(reader, target_ns, window_ns, typestore)
            transforms = list(transforms)
            camera_frame = info_message.header.frame_id
            cloud_frame = cloud_message.header.frame_id
            lidar_to_camera = resolve_frame_transform(transforms, cloud_frame, camera_frame)
            world_from_pointcloud = None
            if args.reference_frame:
                world_from_pointcloud = resolve_frame_transform(
                    transforms, cloud_frame, args.reference_frame
                )
            image = decode_image(camera_message)
            points_xyz = decode_pointcloud_xyz(cloud_message)

            frame_name = f"frame_{index:03d}_{camera_time}"
            frame_dir = args.output / frame_name
            frame_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(frame_dir / "camera.png")
            calibrated_arrays = dict(
                points_xyz=points_xyz,
                projection=np.asarray(info_message.P, dtype=np.float64).reshape(3, 4),
                intrinsic=np.asarray(info_message.K, dtype=np.float64).reshape(3, 3),
                rectification=np.asarray(info_message.R, dtype=np.float64).reshape(3, 3),
                distortion=np.asarray(info_message.D, dtype=np.float64),
                distortion_model=np.array(info_message.distortion_model),
                lidar_to_camera=lidar_to_camera,
                image_size=np.array([info_message.width, info_message.height]),
            )
            if world_from_pointcloud is not None:
                calibrated_arrays["world_from_pointcloud"] = world_from_pointcloud
                calibrated_arrays["reference_frame"] = np.array(args.reference_frame)
            np.savez_compressed(frame_dir / "calibrated_frame.npz", **calibrated_arrays)
            metadata = {
                "index": index,
                "frame_name": frame_name,
                "camera_timestamp_ns": camera_time,
                "camera_info_delta_ms": (info_time - camera_time) / 1e6,
                "pointcloud_timestamp_ns": cloud_time,
                "pointcloud_delta_ms": (cloud_time - camera_time) / 1e6,
                "point_count": int(len(points_xyz)),
                "camera_frame": camera_frame,
                "pointcloud_frame": cloud_frame,
                # Preserve the resolved TF snapshot rather than only camera/cloud
                # products: later audits need to distinguish undercarriage and cab.
                "tf_snapshot": [
                    {"parent": parent, "child": child,
                     "parent_from_child": np.asarray(matrix).tolist()}
                    for parent, child, matrix in transforms
                ],
            }
            if world_from_pointcloud is not None:
                pose = planar_pose(world_from_pointcloud)
                metadata.update(
                    reference_frame=args.reference_frame,
                    position_xy_m=[float(pose[0, 2]), float(pose[1, 2])],
                    yaw_deg=float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))),
                )
            (frame_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            sequence_metadata.append(metadata)
            print(
                f"[{index + 1}/{len(selected)}] {frame_name} "
                f"cloud_delta_ms={metadata['pointcloud_delta_ms']:.1f} points={len(points_xyz)}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_bag": args.bag.name,
        "frame_count": len(sequence_metadata),
        "start_index": args.start_index,
        "stride": args.stride,
        "reference_frame": args.reference_frame,
        "duration_seconds": (
            int(sequence_metadata[-1]["camera_timestamp_ns"])
            - int(sequence_metadata[0]["camera_timestamp_ns"])
        )
        / 1e9,
        "frames": sequence_metadata,
    }
    (args.output / "sequence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
