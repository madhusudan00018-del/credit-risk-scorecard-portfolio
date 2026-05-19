import pandas as pd

from credit_risk.governance import categorical_psi, numeric_psi, segment_performance_report


def test_numeric_psi_detects_stable_distribution() -> None:
    expected = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    actual = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    assert numeric_psi(expected, actual) < 0.01


def test_categorical_psi_returns_non_negative_value() -> None:
    expected = pd.Series(["a", "a", "b", "c"])
    actual = pd.Series(["a", "b", "b", "c"])
    assert categorical_psi(expected, actual) >= 0


def test_segment_performance_report_has_calibration_gap() -> None:
    scored = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "segment": ["x", "x", "y"],
            "predicted_pd": [0.1, 0.2, 0.4],
            "default_next_month": [0, 1, 1],
            "decision": ["approve", "decline_or_review", "decline_or_review"],
            "ead": [100, 200, 300],
            "expected_loss": [4.5, 18.0, 54.0],
        }
    )
    report = segment_performance_report(scored, ["segment"])
    assert "calibration_gap" in report.columns
    assert report["accounts"].sum() == 3

