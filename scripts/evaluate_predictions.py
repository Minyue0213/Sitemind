"""Evaluate semantic predictions with segmentation and safety-oriented metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sitemind.data import discover_frame_pairs, load_label_ids, load_label_mapping
from sitemind.evaluation import (
    binary_auc_from_histograms,
    calibration_error_from_histograms,
    risk_metrics_from_confusion,
    segmentation_metrics_from_confusion,
    update_confusion_matrix,
)
from sitemind.fusion import load_risk_values


ROOT = Path(__file__).resolve().parents[1]


def load_mask_at_size(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.int32, copy=False)


def finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def approximate_aurc(correct_histogram: np.ndarray, error_histogram: np.ndarray) -> float:
    """Approximate area under risk-coverage while accepting low entropy first."""

    accepted = np.cumsum(correct_histogram + error_histogram).astype(np.float64)
    errors = np.cumsum(error_histogram).astype(np.float64)
    if not accepted[-1]:
        return float("nan")
    coverage = accepted / accepted[-1]
    selective_risk = np.divide(errors, accepted, out=np.zeros_like(errors), where=accepted > 0)
    previous_coverage = np.concatenate(([0.0], coverage[:-1]))
    return float(np.sum((coverage - previous_coverage) * selective_risk))


def save_class_report(
    path: Path,
    mapping: dict[int, str],
    class_risk: np.ndarray,
    metrics: dict[str, object],
) -> None:
    iou = np.asarray(metrics["per_class_iou"])
    recall = np.asarray(metrics["per_class_recall"])
    support = np.asarray(metrics["per_class_support"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "risk", "support", "iou", "recall"])
        for label_id in range(len(class_risk)):
            writer.writerow(
                [
                    label_id,
                    mapping.get(label_id, "unknown"),
                    f"{class_risk[label_id]:.2f}",
                    int(support[label_id]),
                    "" if not np.isfinite(iou[label_id]) else f"{iou[label_id]:.6f}",
                    "" if not np.isfinite(recall[label_id]) else f"{recall[label_id]:.6f}",
                ]
            )


def save_overview(
    path: Path,
    summary: dict[str, object],
    risk_confusion: np.ndarray,
    risk_levels: np.ndarray,
    frame_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(0.72, 1.28))
    cards = figure.add_subplot(grid[0, :])
    cards.axis("off")
    card_values = [
        ("mIoU", summary["mean_iou"]),
        ("Pixel accuracy", summary["pixel_accuracy"]),
        ("Hazard recall", summary["hazard_recall"]),
        ("False-safe rate", summary["false_safe_rate"]),
        ("Error AUROC", summary["uncertainty_error_auroc"]),
        ("ECE", summary["expected_calibration_error"]),
    ]
    for index, (label, value) in enumerate(card_values):
        x = (index + 0.5) / len(card_values)
        display = "N/A" if value is None else f"{100 * float(value):.1f}%"
        cards.text(x, 0.62, display, ha="center", va="center", fontsize=22, weight="bold")
        cards.text(x, 0.25, label, ha="center", va="center", fontsize=11, color="#455a64")
    cards.set_title(
        f"SiteMind safety evaluation — {summary['evaluated_frames']} GOOSE-Ex frames",
        fontsize=18,
        weight="bold",
        pad=12,
    )

    confusion_axis = figure.add_subplot(grid[1, 0])
    row_sum = risk_confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        risk_confusion,
        row_sum,
        out=np.zeros_like(risk_confusion, dtype=np.float64),
        where=row_sum > 0,
    )
    confusion_axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    names = [
        "Low" if value < 0.4 else "Medium" if value < 0.75 else "High" if value < 0.99 else "Blocked"
        for value in risk_levels
    ]
    confusion_axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
    confusion_axis.set_yticks(range(len(names)), names)
    confusion_axis.set_xlabel("Predicted risk")
    confusion_axis.set_ylabel("Ground-truth risk")
    confusion_axis.set_title("Risk confusion (row-normalized)")
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            color = "white" if normalized[row, col] > 0.55 else "black"
            confusion_axis.text(
                col,
                row,
                f"{100 * normalized[row, col]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    worst_axis = figure.add_subplot(grid[1, 1])
    eligible = [row for row in frame_rows if int(row["hazardous_pixels"]) > 0]
    worst = sorted(eligible, key=lambda row: float(row["false_safe_rate"]), reverse=True)[:8]
    worst.reverse()
    worst_axis.barh(
        [str(row["frame_index"]) for row in worst],
        [100 * float(row["false_safe_rate"]) for row in worst],
        color="#ef5350",
    )
    worst_axis.set_xlabel("False-safe rate (%)")
    worst_axis.set_ylabel("Frame index")
    worst_axis.set_title("Worst safety frames")
    worst_axis.set_xlim(0, 100)

    explanation = figure.add_subplot(grid[1, 2])
    explanation.axis("off")
    explanation.set_title("How to read this report", loc="left", weight="bold")
    explanation.text(
        0,
        0.94,
        "Hazard recall\nHigher is better. Did we find dangerous pixels?\n\n"
        "False-safe rate\nLower is better. Did danger look safer than it is?\n\n"
        "Error AUROC\nHigher is better. Does uncertainty reveal mistakes?\n\n"
        "ECE\nLower is better. Does confidence match accuracy?\n\n"
        "Decision gate\nFine-tune if hazardous regions are frequently missed\n"
        "or if representative scenes are visually unreliable.",
        va="top",
        fontsize=12,
        linespacing=1.35,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation_val")
    args = parser.parse_args()

    pairs = discover_frame_pairs(args.dataset_root, args.split)
    if not pairs:
        raise SystemExit("No image/label pairs found")
    mapping = load_label_mapping(args.dataset_root / "goose_label_mapping.csv")
    risk_values = load_risk_values(args.risk_config)
    class_risk = np.array(
        [risk_values.get(mapping.get(index, ""), 1.0) for index in range(args.classes)],
        dtype=np.float64,
    )
    ignore_ids = [index for index, name in mapping.items() if name == "undefined"]
    confusion = np.zeros((args.classes, args.classes), dtype=np.int64)
    uncertainty_correct = np.zeros(256, dtype=np.int64)
    uncertainty_error = np.zeros(256, dtype=np.int64)
    confidence_count = np.zeros(15, dtype=np.int64)
    confidence_correct = np.zeros(15, dtype=np.int64)
    confidence_sum = np.zeros(15, dtype=np.float64)
    platform_confusions: dict[str, np.ndarray] = {}
    platform_frame_counts: dict[str, int] = {}
    high_confidence_false_safe = 0
    missing_confidence = False
    resized_predictions = 0
    frame_rows: list[dict[str, object]] = []

    image_root = args.dataset_root / "images" / args.split
    for frame_index, pair in enumerate(pairs):
        relative = pair.image_path.relative_to(image_root).with_suffix(".png")
        prediction_path = args.prediction_root / "label_ids" / relative
        uncertainty_path = args.prediction_root / "uncertainty" / relative
        confidence_path = args.prediction_root / "confidence" / relative
        if not prediction_path.is_file() or not uncertainty_path.is_file():
            raise SystemExit(f"Missing prediction or uncertainty for {relative}")

        target = load_label_ids(pair.label_path)
        target_size = (target.shape[1], target.shape[0])
        with Image.open(prediction_path) as prediction_image:
            if prediction_image.size != target_size:
                resized_predictions += 1
        prediction = load_mask_at_size(prediction_path, target_size)
        uncertainty = load_mask_at_size(uncertainty_path, target_size).clip(0, 255).astype(np.uint8)
        confidence = None
        if confidence_path.is_file():
            confidence = load_mask_at_size(confidence_path, target_size).clip(0, 255).astype(np.uint8)
        else:
            missing_confidence = True

        valid = (
            (target >= 0)
            & (target < args.classes)
            & (prediction >= 0)
            & (prediction < args.classes)
        )
        for ignored in ignore_ids:
            valid &= target != ignored
        correct = prediction == target
        uncertainty_correct += np.bincount(
            uncertainty[valid & correct], minlength=256
        )
        uncertainty_error += np.bincount(
            uncertainty[valid & ~correct], minlength=256
        )

        if confidence is not None:
            bins = np.minimum((confidence.astype(np.int32) * 15) // 256, 14)
            confidence_count += np.bincount(bins[valid], minlength=15)
            confidence_correct += np.bincount(bins[valid & correct], minlength=15)
            confidence_sum += np.bincount(
                bins[valid], weights=confidence[valid] / 255.0, minlength=15
            )
            gt_risk = class_risk[target.clip(0, args.classes - 1)]
            predicted_risk = class_risk[prediction.clip(0, args.classes - 1)]
            high_confidence_false_safe += int(
                np.count_nonzero(
                    valid
                    & (gt_risk >= 0.75)
                    & (predicted_risk < gt_risk)
                    & (confidence >= round(0.9 * 255))
                )
            )

        frame_confusion = np.zeros_like(confusion)
        update_confusion_matrix(
            frame_confusion,
            target,
            prediction,
            ignore_ids=ignore_ids,
        )
        confusion += frame_confusion
        platform = pair.sequence.split("_", 1)[0].lower()
        platform_confusions.setdefault(platform, np.zeros_like(confusion))
        platform_confusions[platform] += frame_confusion
        platform_frame_counts[platform] = platform_frame_counts.get(platform, 0) + 1
        frame_safety = risk_metrics_from_confusion(frame_confusion, class_risk)
        frame_rows.append(
            {
                "frame_index": frame_index,
                "platform": platform,
                "sequence": pair.sequence,
                "timestamp": pair.timestamp,
                "false_safe_rate": frame_safety["false_safe_rate"],
                "hazard_recall": frame_safety["hazard_recall"],
                "hazardous_pixels": frame_safety["hazardous_pixels"],
                "relative_image": str(relative),
            }
        )
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(pairs):
            print(f"[{frame_index + 1}/{len(pairs)}]")

    segmentation = segmentation_metrics_from_confusion(
        confusion, ignore_class_ids=ignore_ids
    )
    safety = risk_metrics_from_confusion(confusion, class_risk)
    uncertainty_auc = binary_auc_from_histograms(
        uncertainty_correct, uncertainty_error
    )
    aurc = approximate_aurc(uncertainty_correct, uncertainty_error)
    ece = (
        float("nan")
        if missing_confidence
        else calibration_error_from_histograms(
            confidence_count, confidence_correct, confidence_sum
        )
    )

    platform_summaries: dict[str, object] = {}
    for platform, platform_confusion in sorted(platform_confusions.items()):
        platform_segmentation = segmentation_metrics_from_confusion(
            platform_confusion, ignore_class_ids=ignore_ids
        )
        platform_safety = risk_metrics_from_confusion(platform_confusion, class_risk)
        platform_summaries[platform] = {
            "frames": platform_frame_counts[platform],
            "pixels": int(platform_confusion.sum()),
            "pixel_accuracy": finite_number(float(platform_segmentation["pixel_accuracy"])),
            "mean_iou": finite_number(float(platform_segmentation["mean_iou"])),
            "risk_accuracy": finite_number(float(platform_safety["risk_accuracy"])),
            "hazard_recall": finite_number(float(platform_safety["hazard_recall"])),
            "false_safe_rate": finite_number(float(platform_safety["false_safe_rate"])),
            "catastrophic_false_safe_rate": finite_number(
                float(platform_safety["catastrophic_false_safe_rate"])
            ),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluated_frames": len(pairs),
        "evaluated_pixels": int(confusion.sum()),
        "ignored_class_ids": ignore_ids,
        "resized_prediction_frames": resized_predictions,
        "pixel_accuracy": finite_number(float(segmentation["pixel_accuracy"])),
        "mean_iou": finite_number(float(segmentation["mean_iou"])),
        "risk_accuracy": finite_number(float(safety["risk_accuracy"])),
        "mean_absolute_risk_error": finite_number(float(safety["mean_absolute_risk_error"])),
        "hazard_recall": finite_number(float(safety["hazard_recall"])),
        "forbidden_recall": finite_number(float(safety["forbidden_recall"])),
        "false_safe_rate": finite_number(float(safety["false_safe_rate"])),
        "catastrophic_false_safe_rate": finite_number(
            float(safety["catastrophic_false_safe_rate"])
        ),
        "uncertainty_error_auroc": finite_number(uncertainty_auc),
        "uncertainty_aurc": finite_number(aurc),
        "expected_calibration_error": finite_number(ece),
        "high_confidence_false_safe_pixels": (
            None if missing_confidence else high_confidence_false_safe
        ),
        "confidence_complete": not missing_confidence,
        "platforms": platform_summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savetxt(args.output / "class_confusion.csv", confusion, delimiter=",", fmt="%d")
    np.savetxt(
        args.output / "risk_confusion.csv",
        np.asarray(safety["confusion"]),
        delimiter=",",
        fmt="%.0f",
    )
    save_class_report(args.output / "per_class.csv", mapping, class_risk, segmentation)
    with (args.output / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    save_overview(
        args.output / "evaluation_overview.png",
        summary,
        np.asarray(safety["confusion"]),
        np.asarray(safety["levels"]),
        frame_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
