from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.config import (
    economics_config,
    ensure_project_directories,
    load_config,
    project_config,
    resolve_path,
)
from credit_risk.data import (
    download_dataset,
    load_raw_credit_data,
    split_train_validation_test,
    validate_raw_schema,
)
from credit_risk.evaluation import (
    build_scored_frame,
    classification_metrics,
    decile_lift_table,
    find_cost_minimizing_threshold,
    risk_band_summary,
    threshold_cost_curve,
)
from credit_risk.explainability import permutation_importance_table, try_write_shap_summary
from credit_risk.features import TARGET_COLUMN, add_credit_risk_features, get_model_feature_columns
from credit_risk.governance import psi_report, segment_performance_report
from credit_risk.models import calibrate_model, fit_and_select_model, predict_pd
from credit_risk.preprocessing import clean_credit_data
from credit_risk.stress_testing import simulate_stress_tests
from credit_risk.visualization import (
    save_calibration_curve,
    save_confusion_matrix,
    save_feature_importance,
    save_precision_recall_curve,
    save_roc_curve,
    save_stress_test_chart,
)


def run_pipeline(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    ensure_project_directories(config)
    project = project_config(config)
    economics = economics_config(config)
    paths = config["paths"]

    raw_path = resolve_path(paths["raw_data"])
    if not raw_path.exists():
        download_dataset(raw_path)

    raw = load_raw_credit_data(raw_path)
    validate_raw_schema(raw)
    clean = clean_credit_data(raw)
    features = add_credit_risk_features(clean)
    feature_columns = get_model_feature_columns(features)
    resolve_path(paths["features"]).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(resolve_path(paths["features"]), index=False)

    train, validation, test = split_train_validation_test(
        features,
        target=TARGET_COLUMN,
        test_size=project.test_size,
        validation_size=project.validation_size,
        random_state=project.random_state,
    )
    selection = fit_and_select_model(
        train=train,
        validation=validation,
        feature_columns=feature_columns,
        target=TARGET_COLUMN,
        random_state=project.random_state,
    )
    train_validation = pd.concat([train, validation], ignore_index=True)
    calibrated_model = calibrate_model(selection.best_model, train_validation, feature_columns, TARGET_COLUMN)

    validation_pd = predict_pd(calibrated_model, validation, feature_columns)
    cost_curve = threshold_cost_curve(
        validation[TARGET_COLUMN],
        validation_pd,
        validation["limit_bal"],
        economics.false_negative_cost_rate,
        economics.false_positive_cost_rate,
    )
    threshold = find_cost_minimizing_threshold(cost_curve)

    test_pd = predict_pd(calibrated_model, test, feature_columns)
    metrics = classification_metrics(test[TARGET_COLUMN], test_pd, threshold)
    scored = build_scored_frame(test, test_pd, threshold, config["risk_bands"])
    band_summary = risk_band_summary(scored)
    deciles = decile_lift_table(scored)

    stress_results = simulate_stress_tests(
        model=calibrated_model,
        clean_base_frame=clean,
        feature_columns=feature_columns,
        scenarios=config["stress_scenarios"],
        threshold=threshold,
        lgd=economics.lgd,
    )

    importance = permutation_importance_table(
        calibrated_model,
        test,
        test[TARGET_COLUMN],
        feature_columns,
        random_state=project.random_state,
    )
    psi = psi_report(train, test, feature_columns)
    segment_monitoring = segment_performance_report(
        scored,
        ["sex_label", "education_label", "marriage_label", "age_band", "limit_band"],
    )

    scored.to_csv(resolve_path(paths["model_scores"]), index=False)
    band_summary.to_csv(resolve_path(paths["risk_band_summary"]), index=False)
    stress_results.to_csv(resolve_path(paths["stress_results"]), index=False)
    deciles.to_csv(resolve_path("data/processed/decile_lift_table.csv"), index=False)
    selection.leaderboard.to_csv(resolve_path("data/processed/model_leaderboard.csv"), index=False)
    importance.to_csv(resolve_path("data/processed/feature_importance.csv"), index=False)
    cost_curve.to_csv(resolve_path("data/processed/threshold_cost_curve.csv"), index=False)
    psi.to_csv(resolve_path("data/processed/model_governance_psi.csv"), index=False)
    segment_monitoring.to_csv(resolve_path("data/processed/segment_performance.csv"), index=False)

    model_artifact = {
        "model": calibrated_model,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "risk_bands": config["risk_bands"],
        "metrics": metrics,
        "selected_model": selection.best_name,
    }
    joblib.dump(model_artifact, resolve_path(paths["model"]))

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "UCI Default of Credit Card Clients",
        "source_url": "https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients",
        "rows": int(len(features)),
        "target_rate": float(features[TARGET_COLUMN].mean()),
        "feature_count": int(len(feature_columns)),
        "selected_model": selection.best_name,
        "threshold": threshold,
    }
    resolve_path(paths["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    resolve_path(paths["metadata"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_roc_curve(test[TARGET_COLUMN], test_pd, resolve_path("images/roc_curve.png"))
    save_precision_recall_curve(
        test[TARGET_COLUMN],
        test_pd,
        resolve_path("images/precision_recall_curve.png"),
    )
    save_confusion_matrix(test[TARGET_COLUMN], test_pd, threshold, resolve_path("images/confusion_matrix.png"))
    save_calibration_curve(test[TARGET_COLUMN], test_pd, resolve_path("images/calibration_curve.png"))
    save_feature_importance(importance, resolve_path("images/feature_importance.png"))
    save_stress_test_chart(stress_results, resolve_path("images/stress_test_expected_loss.png"))
    try_write_shap_summary(calibrated_model, test, feature_columns, resolve_path("images/shap_summary.png"))

    return {
        "metrics": metrics,
        "metadata": metadata,
        "leaderboard": selection.leaderboard,
        "risk_band_summary": band_summary,
        "stress_results": stress_results,
    }
