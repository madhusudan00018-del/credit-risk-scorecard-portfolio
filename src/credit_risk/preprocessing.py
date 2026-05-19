from __future__ import annotations

import numpy as np
import pandas as pd


SEX_MAP = {1: "male", 2: "female"}
EDUCATION_MAP = {
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "other",
}
MARRIAGE_MAP = {
    1: "married",
    2: "single",
    3: "other",
}


def clean_credit_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)

    cleaned["education"] = cleaned["education"].replace({0: 4, 5: 4, 6: 4})
    cleaned["marriage"] = cleaned["marriage"].replace({0: 3})
    cleaned["sex_label"] = cleaned["sex"].map(SEX_MAP).fillna("unknown")
    cleaned["education_label"] = cleaned["education"].map(EDUCATION_MAP).fillna("other")
    cleaned["marriage_label"] = cleaned["marriage"].map(MARRIAGE_MAP).fillna("other")

    financial_cols = [col for col in cleaned.columns if col.startswith(("bill_amt", "pay_amt"))]
    for col in ["limit_bal", *financial_cols]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    repayment_cols = [col for col in cleaned.columns if col.startswith("pay_")]
    repayment_cols = [col for col in repayment_cols if col not in financial_cols]
    for col in repayment_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").clip(-2, 9)

    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce").clip(18, 100)
    cleaned["limit_bal"] = cleaned["limit_bal"].clip(lower=1)
    cleaned["default_next_month"] = cleaned["default_next_month"].astype(int)
    return cleaned


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    low, high = series.quantile([lower, upper])
    return series.clip(low, high)


def add_age_and_limit_segments(frame: pd.DataFrame) -> pd.DataFrame:
    segmented = frame.copy()
    segmented["age_band"] = pd.cut(
        segmented["age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=["18_25", "26_35", "36_45", "46_55", "56_plus"],
        include_lowest=True,
    ).astype(str)
    segmented["limit_band"] = pd.qcut(
        segmented["limit_bal"].rank(method="first"),
        q=5,
        labels=["limit_q1_low", "limit_q2", "limit_q3", "limit_q4", "limit_q5_high"],
    ).astype(str)
    return segmented


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    num = pd.Series(numerator)
    den = pd.Series(denominator).replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)

