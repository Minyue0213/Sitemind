"""Strict readers for GOOSE SemanticKITTI-style 3D annotations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TIMESTAMP = re.compile(r"(?<!\d)(\d{10,})(?!\d)")


@dataclass(frozen=True)
class PointCloudPair:
    scan_path: Path
    label_path: Path
    sequence: str
    timestamp: str


def _index(paths: Iterable[Path], root: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for path in paths:
        matches = TIMESTAMP.findall(path.stem)
        if not matches:
            continue
        key = (path.relative_to(root).parent.as_posix(), matches[-1])
        if key in result:
            raise ValueError(f"duplicate point-cloud identity {key}")
        result[key] = path
    return result


def discover_point_cloud_pairs(
    dataset_root: Path | str, split: str = "val"
) -> list[PointCloudPair]:
    root = Path(dataset_root)
    scan_roots = [candidate for name in ("lidar", "velodyne") if (candidate := root / name / split).is_dir()]
    label_root = root / "labels" / split
    if len(scan_roots) != 1 or not label_root.is_dir():
        raise FileNotFoundError(
            f"expected one lidar or velodyne directory plus labels for split {split}"
        )
    scan_root = scan_roots[0]
    scans = _index(scan_root.rglob("*.bin"), scan_root)
    labels = _index(label_root.rglob("*.label"), label_root)
    missing_labels = scans.keys() - labels.keys()
    missing_scans = labels.keys() - scans.keys()
    if missing_labels or missing_scans:
        raise ValueError(
            "incomplete point-cloud export: "
            f"{len(missing_labels)} scans lack labels and "
            f"{len(missing_scans)} labels lack scans"
        )
    return [
        PointCloudPair(scans[key], labels[key], key[0], key[1])
        for key in sorted(scans.keys() & labels.keys())
    ]


def load_point_cloud(path: Path | str) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"point cloud does not contain xyzi float32 rows: {path}")
    return values.reshape(-1, 4)


def load_point_labels(
    path: Path | str, *, expected_points: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    packed = np.fromfile(path, dtype=np.uint32)
    if expected_points is not None and packed.size != expected_points:
        raise ValueError(
            f"scan/label length mismatch: expected {expected_points}, got {packed.size}"
        )
    semantic = packed & np.uint32(0xFFFF)
    instance = packed >> np.uint32(16)
    return semantic, instance
