"""Streaming metrics used to judge whether model fine-tuning is necessary."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def update_confusion_matrix(
    confusion: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    ignore_ids: Iterable[int] = (),
) -> np.ndarray:
    """Add one label/prediction pair to a square class confusion matrix."""

    matrix = np.asarray(confusion)
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction must have the same shape")

    classes = matrix.shape[0]
    valid = (
        (target_array >= 0)
        & (target_array < classes)
        & (prediction_array >= 0)
        & (prediction_array < classes)
    )
    for label_id in ignore_ids:
        valid &= target_array != int(label_id)
    encoded = classes * target_array[valid].astype(np.int64) + prediction_array[
        valid
    ].astype(np.int64)
    matrix += np.bincount(encoded, minlength=classes * classes).reshape(
        classes, classes
    )
    return matrix


def segmentation_metrics_from_confusion(
    confusion: np.ndarray,
    *,
    ignore_class_ids: Iterable[int] = (),
) -> dict[str, object]:
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    true_pixels = matrix.sum(axis=1)
    predicted_pixels = matrix.sum(axis=0)
    intersection = np.diag(matrix)
    union = true_pixels + predicted_pixels - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan),
        where=union > 0,
    )
    recall = np.divide(
        intersection,
        true_pixels,
        out=np.full_like(intersection, np.nan),
        where=true_pixels > 0,
    )
    for label_id in ignore_class_ids:
        if 0 <= int(label_id) < matrix.shape[0]:
            iou[int(label_id)] = np.nan
            recall[int(label_id)] = np.nan
    total = matrix.sum()
    return {
        "pixel_accuracy": float(intersection.sum() / total) if total else float("nan"),
        "mean_iou": float(np.nanmean(iou)) if np.any(np.isfinite(iou)) else float("nan"),
        "per_class_iou": iou,
        "per_class_recall": recall,
        "per_class_support": true_pixels.astype(np.int64),
    }


def risk_metrics_from_confusion(
    class_confusion: np.ndarray,
    class_risk: np.ndarray,
    *,
    hazard_threshold: float = 0.75,
    forbidden_threshold: float = 0.99,
    catastrophic_gap: float = 0.5,
) -> dict[str, object]:
    """Aggregate class errors into safety-oriented risk metrics."""

    matrix = np.asarray(class_confusion, dtype=np.float64)
    risks = np.asarray(class_risk, dtype=np.float64)
    if matrix.shape != (risks.size, risks.size):
        raise ValueError("class_risk length must match confusion dimensions")
    levels = np.unique(risks)
    level_index = np.searchsorted(levels, risks)
    risk_confusion = np.zeros((levels.size, levels.size), dtype=np.float64)
    np.add.at(
        risk_confusion,
        (level_index[:, None], level_index[None, :]),
        matrix,
    )

    gt_risk = levels[:, None]
    predicted_risk = levels[None, :]
    hazardous = gt_risk >= hazard_threshold
    forbidden = gt_risk >= forbidden_threshold
    false_safe = predicted_risk < gt_risk
    catastrophic = (gt_risk - predicted_risk) >= catastrophic_gap
    predicted_hazard = predicted_risk >= hazard_threshold
    predicted_forbidden = predicted_risk >= forbidden_threshold

    hazardous_pixels = float(risk_confusion[hazardous[:, 0], :].sum())
    forbidden_pixels = float(risk_confusion[forbidden[:, 0], :].sum())
    false_safe_pixels = float(risk_confusion[(hazardous & false_safe)].sum())
    catastrophic_pixels = float(risk_confusion[(hazardous & catastrophic)].sum())
    hazard_true_positive = float(risk_confusion[(hazardous & predicted_hazard)].sum())
    forbidden_true_positive = float(
        risk_confusion[(forbidden & predicted_forbidden)].sum()
    )
    total = float(risk_confusion.sum())
    absolute_error = np.abs(gt_risk - predicted_risk)

    return {
        "levels": levels,
        "confusion": risk_confusion,
        "risk_accuracy": float(np.trace(risk_confusion) / total) if total else float("nan"),
        "mean_absolute_risk_error": (
            float((risk_confusion * absolute_error).sum() / total)
            if total
            else float("nan")
        ),
        "hazard_recall": (
            hazard_true_positive / hazardous_pixels
            if hazardous_pixels
            else float("nan")
        ),
        "forbidden_recall": (
            forbidden_true_positive / forbidden_pixels
            if forbidden_pixels
            else float("nan")
        ),
        "false_safe_rate": (
            false_safe_pixels / hazardous_pixels if hazardous_pixels else float("nan")
        ),
        "catastrophic_false_safe_rate": (
            catastrophic_pixels / hazardous_pixels
            if hazardous_pixels
            else float("nan")
        ),
        "hazardous_pixels": int(hazardous_pixels),
    }


def binary_auc_from_histograms(negative: np.ndarray, positive: np.ndarray) -> float:
    """AUROC for an ascending score histogram; positive should score higher."""

    negative_histogram = np.asarray(negative, dtype=np.float64)
    positive_histogram = np.asarray(positive, dtype=np.float64)
    if negative_histogram.shape != positive_histogram.shape:
        raise ValueError("histograms must have the same shape")
    negatives = negative_histogram.sum()
    positives = positive_histogram.sum()
    if not negatives or not positives:
        return float("nan")
    lower_negatives = np.cumsum(negative_histogram) - negative_histogram
    wins = (positive_histogram * (lower_negatives + 0.5 * negative_histogram)).sum()
    return float(wins / (positives * negatives))


def calibration_error_from_histograms(
    count: np.ndarray,
    correct: np.ndarray,
    confidence_sum: np.ndarray,
) -> float:
    """Expected calibration error from pre-aggregated confidence bins."""

    count_array = np.asarray(count, dtype=np.float64)
    correct_array = np.asarray(correct, dtype=np.float64)
    confidence_array = np.asarray(confidence_sum, dtype=np.float64)
    if not (count_array.shape == correct_array.shape == confidence_array.shape):
        raise ValueError("calibration histograms must have the same shape")
    total = count_array.sum()
    if not total:
        return float("nan")
    nonempty = count_array > 0
    accuracy = np.zeros_like(count_array)
    confidence = np.zeros_like(count_array)
    accuracy[nonempty] = correct_array[nonempty] / count_array[nonempty]
    confidence[nonempty] = confidence_array[nonempty] / count_array[nonempty]
    return float((count_array * np.abs(accuracy - confidence)).sum() / total)
