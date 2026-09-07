import numpy as np

from sitemind.evaluation import (
    binary_auc_from_histograms,
    calibration_error_from_histograms,
    risk_metrics_from_confusion,
    segmentation_metrics_from_confusion,
    update_confusion_matrix,
)


def test_segmentation_and_safety_metrics():
    target = np.array([[0, 1], [2, 2]])
    prediction = np.array([[0, 0], [2, 1]])
    confusion = np.zeros((3, 3), dtype=np.int64)
    update_confusion_matrix(confusion, target, prediction)

    segmentation = segmentation_metrics_from_confusion(confusion)
    assert segmentation["pixel_accuracy"] == 0.5
    np.testing.assert_allclose(segmentation["per_class_iou"], [0.5, 0.0, 0.5])

    safety = risk_metrics_from_confusion(confusion, np.array([0.05, 0.75, 1.0]))
    assert safety["hazard_recall"] == 2 / 3
    assert safety["false_safe_rate"] == 2 / 3
    assert safety["catastrophic_false_safe_rate"] == 1 / 3


def test_histogram_uncertainty_and_calibration_metrics():
    correct_by_uncertainty = np.array([2, 0])
    incorrect_by_uncertainty = np.array([0, 2])
    assert binary_auc_from_histograms(correct_by_uncertainty, incorrect_by_uncertainty) == 1.0

    count = np.array([2, 2])
    correct = np.array([1, 2])
    confidence_sum = np.array([1.0, 2.0])
    assert calibration_error_from_histograms(count, correct, confidence_sum) == 0.0


def test_ignored_class_is_excluded_from_mean_iou():
    confusion = np.array([[0, 0], [1, 3]])
    metrics = segmentation_metrics_from_confusion(confusion, ignore_class_ids=[0])
    assert np.isnan(metrics["per_class_iou"][0])
    assert metrics["mean_iou"] == 0.75
