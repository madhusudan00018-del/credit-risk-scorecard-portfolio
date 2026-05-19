from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_share(values: pd.Series) -> pd.Series:
    total = values.sum()
    if total == 0:
        return values * 0
    return values / total


def numeric_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected_clean = pd.to_numeric(expected, errors="coerce").dropna()
    actual_clean = pd.to_numeric(actual, errors="coerce").dropna()
    if expected_clean.nunique() < 2 or actual_clean.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected_clean.quantile(quantiles).to_numpy())
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_bins = pd.cut(expected_clean, bins=breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual_clean, bins=breakpoints, include_lowest=True)
    expected_dist = _safe_share(expected_bins.value_counts(sort=False)).replace(0, 1e-6)
    actual_dist = _safe_share(actual_bins.value_counts(sort=False)).replace(0, 1e-6)
    return float(((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum())


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    expected_counts = expected.fillna("missing").astype(str).value_counts()
    actual_counts = actual.fillna("missing").astype(str).value_counts()
    categories = expected_counts.index.union(actual_counts.index)
    expected_dist = _safe_share(expected_counts.reindex(categories, fill_value=0)).replace(0, 1e-6)
    actual_dist = _safe_share(actual_counts.reindex(categories, fill_value=0)).replace(0, 1e-6)
    return float(((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum())


def psi_report(
    expected_frame: pd.DataFrame,
    actual_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        expected = expected_frame[feature]
        actual = actual_frame[feature]
        if pd.api.types.is_numeric_dtype(expected):
            value = numeric_psi(expected, actual)
        else:
            value = categorical_psi(expected, actual)
        if value < 0.10:
            status = "stable"
        elif value < 0.25:
            status = "watch"
        else:
            status = "material_shift"
        rows.append({"feature": feature, "psi": value, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def segment_performance_report(scored: pd.DataFrame, segment_columns: list[str]) -> pd.DataFrame:
    frames = []
    for segment in segment_columns:
        grouped = (
            scored.groupby(segment, observed=True)
            .agg(
                accounts=("customer_id", "count"),
                avg_pd=("predicted_pd", "mean"),
                observed_default_rate=("default_next_month", "mean"),
                approval_rate=("decision", lambda x: (x == "approve").mean()),
                exposure=("ead", "sum"),
                expected_loss=("expected_loss", "sum"),
            )
            .reset_index()
            .rename(columns={segment: "segment_value"})
        )
        grouped["segment"] = segment
        grouped["exposure_share"] = grouped["exposure"] / grouped["exposure"].sum()
        frames.append(grouped)
    report = pd.concat(frames, ignore_index=True)
    report["calibration_gap"] = report["observed_default_rate"] - report["avg_pd"]
    return report[
        [
            "segment",
            "segment_value",
            "accounts",
            "avg_pd",
            "observed_default_rate",
            "calibration_gap",
            "approval_rate",
            "exposure",
            "exposure_share",
            "expected_loss",
        ]
    ].sort_values(["segment", "expected_loss"], ascending=[True, False])

