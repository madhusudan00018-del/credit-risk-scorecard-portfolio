import numpy as np
import pandas as pd

from credit_risk.evaluation import (
    classification_metrics,
    find_cost_minimizing_threshold,
    threshold_cost_curve,
)


def test_classification_metrics_return_core_fields() -> None:
    y_true = np.array([0, 0, 1, 1])
    predicted_pd = np.array([0.05, 0.20, 0.70, 0.90])
    metrics = classification_metrics(y_true, predicted_pd, threshold=0.50)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_threshold_cost_curve_has_minimum() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1])
    predicted_pd = pd.Series([0.02, 0.10, 0.25, 0.55, 0.80])
    ead = pd.Series([100, 100, 100, 100, 100])
    curve = threshold_cost_curve(
        y_true,
        predicted_pd,
        ead,
        false_negative_cost_rate=0.45,
        false_positive_cost_rate=0.025,
        thresholds=np.array([0.10, 0.30, 0.60]),
    )
    threshold = find_cost_minimizing_threshold(curve)
    assert threshold in {0.10, 0.30, 0.60}
    assert {"approval_rate", "bad_capture_rate", "expected_cost"}.issubset(curve.columns)

