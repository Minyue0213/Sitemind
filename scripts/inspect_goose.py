"""Inspect a downloaded GOOSE/GOOSE-Ex 2D split before model work."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sitemind.data import discover_frame_pairs, load_label_ids, load_label_mapping, load_rgb
from sitemind.fusion import load_risk_values


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml"
    )
    args = parser.parse_args()

    mapping_path = args.dataset_root / "goose_label_mapping.csv"
    mapping = load_label_mapping(mapping_path)
    pairs = discover_frame_pairs(args.dataset_root, args.split)
    print(f"classes={len(mapping)}")
    print(f"paired_frames={len(pairs)}")
    if not pairs:
        raise SystemExit("No image/label pairs found")
    sample = pairs[0]
    rgb = load_rgb(sample.image_path)
    labels = load_label_ids(sample.label_path)
    present_ids = np.unique(labels)
    unknown_ids = [int(label_id) for label_id in present_ids if int(label_id) not in mapping]
    uncovered_classes = sorted(set(mapping.values()) - set(load_risk_values(args.risk_config)))
    present_uncovered = sorted(
        {
            mapping[int(label_id)]
            for label_id in present_ids
            if int(label_id) in mapping
            and mapping[int(label_id)] not in load_risk_values(args.risk_config)
        }
    )
    print(f"sample_sequence={sample.sequence}")
    print(f"sample_timestamp={sample.timestamp}")
    print(f"rgb_shape={rgb.shape}")
    print(f"label_shape={labels.shape}")
    print(f"sample_label_ids={present_ids.tolist()}")
    print(f"unknown_label_ids={unknown_ids}")
    print(f"uncovered_mapping_classes={uncovered_classes}")
    print(f"present_uncovered_classes={present_uncovered}")


if __name__ == "__main__":
    main()
