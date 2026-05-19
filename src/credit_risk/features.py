from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.preprocessing import add_age_and_limit_segments, safe_divide, winsorize_series


TARGET_COLUMN = "default_next_month"
ID_COLUMN = "customer_id"
REPAYMENT_COLUMNS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLUMNS = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
PAYMENT_COLUMNS = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]
CATEGORICAL_FEATURES = [
    "sex_label",
    "education_label",
    "marriage_label",
    "age_band",
    "limit_band",
]


def add_credit_risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    featured = add_age_and_limit_segments(frame)

    featured["avg_bill_amt"] = featured[BILL_COLUMNS].mean(axis=1)
    featured["max_bill_amt"] = featured[BILL_COLUMNS].max(axis=1)
    featured["min_bill_amt"] = featured[BILL_COLUMNS].min(axis=1)
    featured["total_bill_amt"] = featured[BILL_COLUMNS].sum(axis=1)
    featured["total_payment_amt"] = featured[PAYMENT_COLUMNS].sum(axis=1)
    featured["avg_payment_amt"] = featured[PAYMENT_COLUMNS].mean(axis=1)

    featured["avg_utilization"] = safe_divide(featured["avg_bill_amt"], featured["limit_bal"]).clip(-1, 5)
    featured["max_utilization"] = safe_divide(featured["max_bill_amt"], featured["limit_bal"]).clip(-1, 5)
    featured["recent_utilization"] = safe_divide(featured["bill_amt1"], featured["limit_bal"]).clip(-1, 5)
    featured["payment_to_bill_ratio"] = safe_divide(
        featured["total_payment_amt"],
        featured["total_bill_amt"].abs() + 1,
    ).clip(0, 20)
    featured["payment_to_limit_ratio"] = safe_divide(
        featured["total_payment_amt"],
        featured["limit_bal"],
    ).clip(0, 20)

    # Positive repayment status values are months of payment delay.
    delinquency = featured[REPAYMENT_COLUMNS].clip(lower=0)
    featured["months_delinquent"] = (delinquency > 0).sum(axis=1)
    featured["max_delinquency"] = delinquency.max(axis=1)
    featured["avg_delinquency"] = delinquency.mean(axis=1)
    featured["recent_delinquency"] = delinquency["pay_0"]
    featured["severe_delinquency_flag"] = (featured["max_delinquency"] >= 3).astype(int)

    # The dataset is ordered from recent month to older month for bill amounts.
    featured["bill_amount_trend"] = safe_divide(
        featured["bill_amt1"] - featured["bill_amt6"],
        featured["limit_bal"],
    ).clip(-5, 5)
    featured["payment_momentum"] = safe_divide(
        featured["pay_amt1"] + featured["pay_amt2"] - featured["pay_amt5"] - featured["pay_amt6"],
        featured["limit_bal"],
    ).clip(-5, 5)

    featured["revolving_balance_flag"] = (featured["recent_utilization"] > 0.80).astype(int)
    featured["low_payment_high_util_flag"] = (
        (featured["recent_utilization"] > 0.75) & (featured["payment_to_bill_ratio"] < 0.08)
    ).astype(int)
    featured["young_high_util_flag"] = (
        (featured["age"] <= 30) & (featured["recent_utilization"] > 0.70)
    ).astype(int)

    for col in [
        "avg_bill_amt",
        "max_bill_amt",
        "total_bill_amt",
        "total_payment_amt",
        "avg_payment_amt",
    ]:
        featured[f"{col}_winsor"] = winsorize_series(featured[col])
        featured[f"log_{col}_abs"] = np.log1p(featured[col].abs())

    return featured


def get_model_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COLUMN, ID_COLUMN}
    return [col for col in frame.columns if col not in excluded]


def assign_risk_band(pd_values: pd.Series, band_config: list[dict[str, float | str]]) -> pd.Series:
    labels = []
    for value in pd_values:
        matched = "unbanded"
        for band in band_config:
            if float(band["min_pd"]) <= float(value) < float(band["max_pd"]):
                matched = str(band["name"])
                break
        labels.append(matched)
    return pd.Series(labels, index=pd_values.index)


def feature_type_split(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [col for col in CATEGORICAL_FEATURES if col in feature_columns]
    numeric = [col for col in feature_columns if col not in categorical]
    return numeric, categorical

