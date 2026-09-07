"""Evaluation helpers for semantic segmentation and safety-risk quality."""

from .metrics import (
    binary_auc_from_histograms,
    calibration_error_from_histograms,
    risk_metrics_from_confusion,
    segmentation_metrics_from_confusion,
    update_confusion_matrix,
)

__all__ = [
    "binary_auc_from_histograms",
    "calibration_error_from_histograms",
    "risk_metrics_from_confusion",
    "segmentation_metrics_from_confusion",
    "update_confusion_matrix",
]
