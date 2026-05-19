from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_risk.features import assign_risk_band


def ks_statistic(y_true: pd.Series | np.ndarray, predicted_pd: pd.Series | np.ndarray) -> float:
    data = pd.DataFrame({"target": y_true, "pd": predicted_pd}).sort_values("pd")
    positives = data["target"].sum()
    negatives = len(data) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    data["cum_bad"] = data["target"].cumsum() / positives
    data["cum_good"] = ((1 - data["target"]).cumsum()) / negatives
    return float((data["cum_bad"] - data["cum_good"]).abs().max())


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    predicted_pd: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    y_pred = (np.asarray(predicted_pd) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, predicted_pd)),
        "average_precision": float(average_precision_score(y_true, predicted_pd)),
        "brier_score": float(brier_score_loss(y_true, predicted_pd)),
        "ks_statistic": ks_statistic(y_true, predicted_pd),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def threshold_cost_curve(
    y_true: pd.Series | np.ndarray,
    predicted_pd: pd.Series | np.ndarray,
    ead: pd.Series | np.ndarray,
    false_negative_cost_rate: float,
    false_positive_cost_rate: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.03, 0.60, 116)
    y = np.asarray(y_true)
    pd_values = np.asarray(predicted_pd)
    ead_values = np.asarray(ead)
    rows = []
    for threshold in thresholds:
        predicted_bad = (pd_values >= threshold).astype(int)
        false_negative = (y == 1) & (predicted_bad == 0)
        false_positive = (y == 0) & (predicted_bad == 1)
        expected_cost = (
            false_negative_cost_rate * ead_values[false_negative].sum()
            + false_positive_cost_rate * ead_values[false_positive].sum()
        )
        rows.append(
            {
                "threshold": float(threshold),
                "approval_rate": float((predicted_bad == 0).mean()),
                "bad_capture_rate": float(((y == 1) & (predicted_bad == 1)).sum() / max((y == 1).sum(), 1)),
                "expected_cost": float(expected_cost),
                "false_negative_count": int(false_negative.sum()),
                "false_positive_count": int(false_positive.sum()),
            }
        )
    return pd.DataFrame(rows)


def find_cost_minimizing_threshold(cost_curve: pd.DataFrame) -> float:
    return float(cost_curve.loc[cost_curve["expected_cost"].idxmin(), "threshold"])


def build_scored_frame(
    frame: pd.DataFrame,
    predicted_pd: np.ndarray,
    threshold: float,
    risk_bands: list[dict[str, float | str]],
) -> pd.DataFrame:
    scored = frame.copy()
    scored["predicted_pd"] = predicted_pd
    scored["decision"] = np.where(scored["predicted_pd"] >= threshold, "decline_or_review", "approve")
    scored["risk_band"] = assign_risk_band(scored["predicted_pd"], risk_bands)
    scored["ead"] = scored["limit_bal"]
    scored["expected_loss"] = scored["predicted_pd"] * scored["ead"] * 0.45
    return scored


def risk_band_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby("risk_band", observed=True)
        .agg(
            accounts=("customer_id", "count"),
            exposure=("ead", "sum"),
            avg_pd=("predicted_pd", "mean"),
            observed_default_rate=("default_next_month", "mean"),
            expected_loss=("expected_loss", "sum"),
            approval_rate=("decision", lambda x: (x == "approve").mean()),
        )
        .reset_index()
    )
    summary["exposure_share"] = summary["exposure"] / summary["exposure"].sum()
    summary["loss_share"] = summary["expected_loss"] / summary["expected_loss"].sum()
    return summary.sort_values("avg_pd")


def decile_lift_table(scored: pd.DataFrame) -> pd.DataFrame:
    table = scored.copy()
    table["risk_decile"] = pd.qcut(
        table["predicted_pd"].rank(method="first"),
        q=10,
        labels=False,
    ) + 1
    grouped = (
        table.groupby("risk_decile", observed=True)
        .agg(
            accounts=("customer_id", "count"),
            avg_pd=("predicted_pd", "mean"),
            observed_default_rate=("default_next_month", "mean"),
            exposure=("ead", "sum"),
            expected_loss=("expected_loss", "sum"),
        )
        .reset_index()
        .sort_values("risk_decile", ascending=False)
    )
    base_rate = table["default_next_month"].mean()
    grouped["lift_vs_portfolio"] = grouped["observed_default_rate"] / base_rate
    return grouped

