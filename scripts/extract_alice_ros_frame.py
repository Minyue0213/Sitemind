"""Extract one calibrated ALICE camera/LiDAR frame from an official ROS1 bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from sitemind.fusion import resolve_frame_transform, rigid_transform


CAMERA_INFO_TOPICS = (
    "/stereo_camera/left/vis/camera_info",
    "/stero_camera/left/vis/camera_info",  # spelling used by the official docs
)
IMAGE_TOPICS = ("/stereo_camera/left/vis/image_color",)
CLOUD_TOPICS = ("/pcl_merging/merged_cloud",)


def pick_topic(connections: Iterable[Any], candidates: tuple[str, ...]) -> str:
    available = {connection.topic for connection in connections}
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"none of the required topics are present: {candidates}")


def register_bag_message_types(reader: Reader, typestore: Any) -> None:
    """Register message definitions embedded in the bag.

    The ROS1 Noetic store does not include every package (notably
    ``tf2_msgs/TFMessage``), while a ROS1 bag carries the definitions required
    to decode its own connections.
    """

    message_types: dict[str, Any] = {}
    for connection in reader.connections:
        message_types.update(get_types_from_msg(connection.msgdef, connection.msgtype))
    typestore.register(message_types)


def nearest_message(
    reader: Reader,
    topic: str,
    target_ns: int,
    window_ns: int,
    typestore: Any,
) -> tuple[int, Any]:
    connections = [connection for connection in reader.connections if connection.topic == topic]
    best: tuple[int, Any] | None = None
    for connection, timestamp, raw in reader.messages(
        connections=connections,
        start=target_ns - window_ns,
        stop=target_ns + window_ns + 1,
    ):
        message = typestore.deserialize_ros1(raw, connection.msgtype)
        if best is None or abs(timestamp - target_ns) < abs(best[0] - target_ns):
            best = (timestamp, message)
    if best is None:
        raise ValueError(f"no {topic} message within {window_ns / 1e9:.1f}s of target")
    return best


def transform_tuple(transform: Any) -> tuple[str, str, np.ndarray]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = rigid_transform(
        np.array([translation.x, translation.y, translation.z]),
        np.array([rotation.x, rotation.y, rotation.z, rotation.w]),
    )
    return transform.header.frame_id, transform.child_frame_id, matrix


def collect_tf(
    reader: Reader,
    target_ns: int,
    window_ns: int,
    typestore: Any,
) -> list[tuple[str, str, np.ndarray]]:
    selected: dict[tuple[str, str], tuple[int, tuple[str, str, np.ndarray]]] = {}
    static_connections = [c for c in reader.connections if c.topic == "/tf_static"]
    for connection, timestamp, raw in reader.messages(connections=static_connections):
        message = typestore.deserialize_ros1(raw, connection.msgtype)
        for transform in message.transforms:
            item = transform_tuple(transform)
            selected[(item[0].lstrip("/"), item[1].lstrip("/"))] = (timestamp, item)

    dynamic_connections = [c for c in reader.connections if c.topic == "/tf"]
    for connection, timestamp, raw in reader.messages(
        connections=dynamic_connections,
        start=target_ns - window_ns,
        stop=target_ns + window_ns + 1,
    ):
        message = typestore.deserialize_ros1(raw, connection.msgtype)
        for transform in message.transforms:
            item = transform_tuple(transform)
            key = (item[0].lstrip("/"), item[1].lstrip("/"))
            previous = selected.get(key)
            if previous is None or abs(timestamp - target_ns) < abs(previous[0] - target_ns):
                selected[key] = (timestamp, item)
    return [item for _, item in selected.values()]


def decode_image(message: Any) -> np.ndarray:
    encoding = message.encoding.lower()
    channels_by_encoding = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding: {message.encoding}")
    channels = channels_by_encoding[encoding]
    raw = np.asarray(message.data, dtype=np.uint8)
    rows = raw.reshape(int(message.height), int(message.step))
    image = rows[:, : int(message.width) * channels].reshape(
        int(message.height), int(message.width), channels
    )
    if encoding == "mono8":
        return image[..., 0]
    if encoding in {"bgr8", "bgra8"}:
        image = image[..., [2, 1, 0, 3] if channels == 4 else [2, 1, 0]]
    return image[..., :3].copy()


def decode_pointcloud_xyz(message: Any) -> np.ndarray:
    formats = {
        1: "i1",
        2: "u1",
        3: "i2",
        4: "u2",
        5: "i4",
        6: "u4",
        7: "f4",
        8: "f8",
    }
    byte_order = ">" if message.is_bigendian else "<"
    names, field_formats, offsets = [], [], []
    for field in message.fields:
        if field.datatype not in formats:
            raise ValueError(f"unsupported PointField datatype {field.datatype}")
        names.append(field.name)
        base = byte_order + formats[field.datatype]
        field_formats.append(base if field.count == 1 else (base, (field.count,)))
        offsets.append(field.offset)
    dtype = np.dtype(
        {"names": names, "formats": field_formats, "offsets": offsets, "itemsize": message.point_step}
    )
    count = int(message.width) * int(message.height)
    cloud = np.frombuffer(np.asarray(message.data, dtype=np.uint8), dtype=dtype, count=count)
    if not {"x", "y", "z"}.issubset(cloud.dtype.names or ()):
        raise ValueError("point cloud does not contain x/y/z fields")
    return np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("target_timestamp_ns", type=int)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/alice_seq02_raw"))
    args = parser.parse_args()

    typestore = get_typestore(Stores.ROS1_NOETIC)
    window_ns = int(args.window_seconds * 1e9)
    with Reader(args.bag) as reader:
        register_bag_message_types(reader, typestore)
        camera_info_topic = pick_topic(reader.connections, CAMERA_INFO_TOPICS)
        image_topic = pick_topic(reader.connections, IMAGE_TOPICS)
        cloud_topic = pick_topic(reader.connections, CLOUD_TOPICS)
        camera_time, camera_message = nearest_message(
            reader, image_topic, args.target_timestamp_ns, window_ns, typestore
        )
        info_time, info_message = nearest_message(
            reader, camera_info_topic, args.target_timestamp_ns, window_ns, typestore
        )
        cloud_time, cloud_message = nearest_message(
            reader, cloud_topic, args.target_timestamp_ns, window_ns, typestore
        )
        transforms = collect_tf(reader, args.target_timestamp_ns, window_ns, typestore)

    camera_frame = info_message.header.frame_id
    cloud_frame = cloud_message.header.frame_id
    lidar_to_camera = resolve_frame_transform(transforms, cloud_frame, camera_frame)
    projection = np.asarray(info_message.P, dtype=np.float64).reshape(3, 4)
    intrinsic = np.asarray(info_message.K, dtype=np.float64).reshape(3, 3)
    rectification = np.asarray(info_message.R, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(info_message.D, dtype=np.float64)
    image = decode_image(camera_message)
    points_xyz = decode_pointcloud_xyz(cloud_message)

    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output / "camera.png")
    np.savez_compressed(
        args.output / "calibrated_frame.npz",
        points_xyz=points_xyz,
        projection=projection,
        intrinsic=intrinsic,
        rectification=rectification,
        distortion=distortion,
        distortion_model=np.array(info_message.distortion_model),
        lidar_to_camera=lidar_to_camera,
        image_size=np.array([info_message.width, info_message.height]),
    )
    metadata = {
        "source_bag": args.bag.name,
        "target_timestamp_ns": args.target_timestamp_ns,
        "camera_timestamp_ns": camera_time,
        "camera_delta_ms": (camera_time - args.target_timestamp_ns) / 1e6,
        "camera_info_timestamp_ns": info_time,
        "pointcloud_timestamp_ns": cloud_time,
        "pointcloud_delta_ms": (cloud_time - args.target_timestamp_ns) / 1e6,
        "camera_topic": image_topic,
        "camera_info_topic": camera_info_topic,
        "pointcloud_topic": cloud_topic,
        "camera_frame": camera_frame,
        "pointcloud_frame": cloud_frame,
        "image_encoding": camera_message.encoding,
        "distortion_model": info_message.distortion_model,
        "image_size": [int(info_message.width), int(info_message.height)],
        "point_count": int(len(points_xyz)),
        "tf_edge_count": len(transforms),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
