"""Readers for the documented GOOSE 2D directory layout."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image


TIMESTAMP = re.compile(r"(?<!\d)(\d{10,})(?!\d)")


@dataclass(frozen=True)
class FramePair:
    image_path: Path
    label_path: Path
    sequence: str
    timestamp: str


def _last_timestamp(path: Path) -> Optional[str]:
    matches = TIMESTAMP.findall(path.stem)
    return matches[-1] if matches else None


def _index_by_timestamp(
    paths: Iterable[Path], root: Path
) -> Dict[tuple[str, str], Path]:
    index: Dict[tuple[str, str], Path] = {}
    for path in paths:
        timestamp = _last_timestamp(path)
        if timestamp is None:
            continue
        sequence = path.relative_to(root).parent.as_posix()
        key = (sequence, timestamp)
        if key in index:
            raise ValueError(f"duplicate frame identity {key}: {index[key]} and {path}")
        index[key] = path
    return index


def discover_frame_pairs(dataset_root: Path | str, split: str = "val") -> list[FramePair]:
    """Pair RGB images and ``*_labelids.png`` masks using sequence + timestamp."""

    root = Path(dataset_root)
    image_root = root / "images" / split
    label_root = root / "labels" / split
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing image directory: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"missing label directory: {label_root}")

    images = _index_by_timestamp(image_root.rglob("*.png"), image_root)
    labels = _index_by_timestamp(label_root.rglob("*_labelids.png"), label_root)
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(
            "incomplete image/label export: "
            f"{len(missing_labels)} images lack labels and "
            f"{len(missing_images)} labels lack images"
        )
    common = sorted(images.keys() & labels.keys())
    return [
        FramePair(images[key], labels[key], sequence=key[0], timestamp=key[1])
        for key in common
    ]


def load_rgb(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_label_ids(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        labels = np.asarray(image)
    if labels.ndim == 3:
        labels = labels[..., 0]
    return labels.astype(np.int32, copy=False)


def load_label_mapping(path: Path | str) -> Dict[int, str]:
    """Read common GOOSE CSV variants without hard-coding column order."""

    id_candidates = ("label_key", "class_id", "label_id", "goose_id", "id")
    name_candidates = ("class_name", "label_name", "name", "class")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("label mapping CSV has no header")
        normalized = {field.strip().lower(): field for field in reader.fieldnames}
        id_field = next((normalized[name] for name in id_candidates if name in normalized), None)
        name_field = next((normalized[name] for name in name_candidates if name in normalized), None)
        if id_field is None or name_field is None:
            raise ValueError(
                f"cannot identify ID/name columns in {reader.fieldnames}; "
                "expected an id-like and a name-like column"
            )
        mapping: Dict[int, str] = {}
        for row in reader:
            mapping[int(row[id_field])] = row[name_field].strip().lower()
    return mapping
