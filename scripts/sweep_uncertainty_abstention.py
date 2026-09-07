"""Sweep entropy abstention thresholds for uncertainty-aware risk decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sitemind.data import discover_frame_pairs, load_label_ids, load_label_mapping
from sitemind.evaluation import risk_metrics_from_confusion
from sitemind.fusion import load_risk_values


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml")
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "abstention_sweep")
    args = parser.parse_args()

    pairs = discover_frame_pairs(args.dataset_root, args.split)
    mapping = load_label_mapping(args.dataset_root / "goose_label_mapping.csv")
    risk_values = load_risk_values(args.risk_config)
    class_risk = np.array(
        [risk_values.get(mapping.get(index, ""), 1.0) for index in range(args.classes)]
    )
    levels = np.unique(class_risk)
    class_to_level = np.searchsorted(levels, class_risk)
    blocked_index = int(np.argmax(levels))
    ignore_ids = [index for index, name in mapping.items() if name == "undefined"]
    image_root = args.dataset_root / "images" / args.split
    histograms: dict[str, np.ndarray] = {
        "overall": np.zeros((len(levels), len(levels), 256), dtype=np.int64)
    }

    for index, pair in enumerate(pairs, start=1):
        relative = pair.image_path.relative_to(image_root).with_suffix(".png")
        target = load_label_ids(pair.label_path)
        prediction = load_label_ids(args.prediction_root / "label_ids" / relative)
        entropy = load_label_ids(args.prediction_root / "uncertainty" / relative).clip(0, 255)
        if not (target.shape == prediction.shape == entropy.shape):
            raise SystemExit(f"shape mismatch for {relative}")
        valid = (
            (target >= 0)
            & (target < args.classes)
            & (prediction >= 0)
            & (prediction < args.classes)
        )
        for ignored in ignore_ids:
            valid &= target != ignored
        gt_level = class_to_level[target[valid]]
        predicted_level = class_to_level[prediction[valid]]
        encoded = (
            (gt_level * len(levels) + predicted_level) * 256
            + entropy[valid].astype(np.int64)
        )
        frame_histogram = np.bincount(
            encoded, minlength=len(levels) * len(levels) * 256
        ).reshape(len(levels), len(levels), 256)
        platform = pair.sequence.split("_", 1)[0].lower()
        histograms.setdefault(
            platform, np.zeros((len(levels), len(levels), 256), dtype=np.int64)
        )
        histograms["overall"] += frame_histogram
        histograms[platform] += frame_histogram
        if index % 50 == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}]", flush=True)

    if not 1 <= args.threshold_step <= 255:
        raise SystemExit("threshold step must be between 1 and 255")
    thresholds = list(range(args.threshold_step, 256, args.threshold_step)) + [256]
    rows: list[dict[str, object]] = []
    for platform, histogram in histograms.items():
        total = int(histogram.sum())
        for threshold in thresholds:
            retained = histogram[:, :, :threshold].sum(axis=2)
            abstained_by_gt = histogram[:, :, threshold:].sum(axis=(1, 2))
            adjusted = retained.copy()
            adjusted[:, blocked_index] += abstained_by_gt
            metrics = risk_metrics_from_confusion(adjusted, levels)
            rows.append(
                {
                    "platform": platform,
                    "entropy_threshold": threshold / 255 if threshold < 256 else 1.01,
                    "coverage": float(retained.sum() / total),
                    "abstention_rate": float(1 - retained.sum() / total),
                    "risk_accuracy": metrics["risk_accuracy"],
                    "hazard_recall": metrics["hazard_recall"],
                    "false_safe_rate": metrics["false_safe_rate"],
                    "catastrophic_false_safe_rate": metrics[
                        "catastrophic_false_safe_rate"
                    ],
                }
            )

    alice_candidates = [
        row
        for row in rows
        if row["platform"] == "alice"
        and float(row["coverage"]) >= args.minimum_coverage
    ]
    recommended = min(
        alice_candidates,
        key=lambda row: (
            float(row["false_safe_rate"])
            + float(row["catastrophic_false_safe_rate"]),
            -float(row["risk_accuracy"]),
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "recommended.json").write_text(
        json.dumps(recommended, indent=2) + "\n", encoding="utf-8"
    )
    print("recommended=" + json.dumps(recommended), flush=True)


if __name__ == "__main__":
    main()
