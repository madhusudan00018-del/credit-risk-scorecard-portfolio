import pandas as pd

from credit_risk.features import add_credit_risk_features, get_model_feature_columns
from credit_risk.preprocessing import clean_credit_data


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2],
            "limit_bal": [100000, 200000],
            "sex": [1, 2],
            "education": [2, 5],
            "marriage": [1, 0],
            "age": [28, 41],
            "pay_0": [0, 2],
            "pay_2": [0, 2],
            "pay_3": [-1, 0],
            "pay_4": [0, 3],
            "pay_5": [0, 0],
            "pay_6": [0, 0],
            "bill_amt1": [35000, 180000],
            "bill_amt2": [32000, 170000],
            "bill_amt3": [30000, 160000],
            "bill_amt4": [28000, 150000],
            "bill_amt5": [26000, 140000],
            "bill_amt6": [24000, 130000],
            "pay_amt1": [5000, 1000],
            "pay_amt2": [5000, 1000],
            "pay_amt3": [5000, 1000],
            "pay_amt4": [5000, 1000],
            "pay_amt5": [5000, 1000],
            "pay_amt6": [5000, 1000],
            "default_next_month": [0, 1],
        }
    )


def test_feature_engineering_creates_credit_risk_variables() -> None:
    clean = clean_credit_data(sample_raw_frame())
    features = add_credit_risk_features(clean)
    assert "avg_utilization" in features.columns
    assert "months_delinquent" in features.columns
    assert "payment_to_bill_ratio" in features.columns
    assert features.loc[1, "months_delinquent"] == 3


def test_model_features_exclude_id_and_target() -> None:
    clean = clean_credit_data(sample_raw_frame())
    features = add_credit_risk_features(clean)
    feature_columns = get_model_feature_columns(features)
    assert "customer_id" not in feature_columns
    assert "default_next_month" not in feature_columns

