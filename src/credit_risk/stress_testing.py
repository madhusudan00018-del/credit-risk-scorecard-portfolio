from __future__ import annotations

import pandas as pd

from credit_risk.features import BILL_COLUMNS, PAYMENT_COLUMNS, REPAYMENT_COLUMNS, add_credit_risk_features
from credit_risk.models import predict_pd


def apply_stress_scenario(frame: pd.DataFrame, scenario: dict[str, float | int]) -> pd.DataFrame:
    stressed = frame.copy()
    bill_multiplier = float(scenario.get("bill_amount_multiplier", 1.0))
    payment_multiplier = float(scenario.get("payment_amount_multiplier", 1.0))
    delinquency_shift = int(scenario.get("delinquency_shift", 0))

    stressed[BILL_COLUMNS] = stressed[BILL_COLUMNS] * bill_multiplier
    stressed[PAYMENT_COLUMNS] = stressed[PAYMENT_COLUMNS] * payment_multiplier
    for col in REPAYMENT_COLUMNS:
        stressed[col] = stressed[col].where(stressed[col] < -1, stressed[col] + delinquency_shift).clip(-2, 9)
    return stressed


def simulate_stress_tests(
    model: object,
    clean_base_frame: pd.DataFrame,
    feature_columns: list[str],
    scenarios: dict[str, dict[str, float | int]],
    threshold: float,
    lgd: float,
) -> pd.DataFrame:
    rows = []
    for scenario_name, scenario_config in scenarios.items():
        stressed_raw = apply_stress_scenario(clean_base_frame, scenario_config)
        stressed_features = add_credit_risk_features(stressed_raw)
        predicted_pd = predict_pd(model, stressed_features, feature_columns)
        exposure = stressed_features["limit_bal"]
        expected_loss = predicted_pd * exposure * lgd
        approval_rate = (predicted_pd < threshold).mean()
        rows.append(
            {
                "scenario": scenario_name,
                "accounts": int(len(stressed_features)),
                "total_exposure": float(exposure.sum()),
                "avg_pd": float(predicted_pd.mean()),
                "median_pd": float(pd.Series(predicted_pd).median()),
                "p90_pd": float(pd.Series(predicted_pd).quantile(0.90)),
                "expected_loss": float(expected_loss.sum()),
                "expected_loss_rate": float(expected_loss.sum() / exposure.sum()),
                "approval_rate": float(approval_rate),
            }
        )
    results = pd.DataFrame(rows)
    base_loss = float(results.loc[results["scenario"] == "base", "expected_loss"].iloc[0])
    results["incremental_expected_loss"] = results["expected_loss"] - base_loss
    return results

