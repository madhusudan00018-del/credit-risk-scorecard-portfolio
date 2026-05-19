from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.features import feature_type_split


@dataclass
class ModelSelectionResult:
    best_name: str
    best_model: Pipeline
    leaderboard: pd.DataFrame


def make_preprocessor(frame: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_features, categorical_features = feature_type_split(frame, feature_columns)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_model_grid(random_state: int) -> dict[str, tuple[object, list[dict[str, object]]]]:
    return {
        "logistic_scorecard": (
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
            [
                {"model__C": 0.25, "model__penalty": "l2"},
                {"model__C": 0.75, "model__penalty": "l2"},
                {"model__C": 1.50, "model__penalty": "l2"},
            ],
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=260,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=random_state,
            ),
            [
                {"model__max_depth": 6, "model__min_samples_leaf": 80},
                {"model__max_depth": 8, "model__min_samples_leaf": 60},
                {"model__max_depth": 10, "model__min_samples_leaf": 40},
            ],
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=220,
                early_stopping=True,
                validation_fraction=0.15,
                l2_regularization=0.03,
                random_state=random_state,
            ),
            [
                {"model__learning_rate": 0.03, "model__max_leaf_nodes": 15},
                {"model__learning_rate": 0.05, "model__max_leaf_nodes": 31},
                {"model__learning_rate": 0.08, "model__max_leaf_nodes": 31},
            ],
        ),
    }


def fit_and_select_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    random_state: int,
) -> ModelSelectionResult:
    preprocessor = make_preprocessor(train, feature_columns)
    records: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    x_train, y_train = train[feature_columns], train[target]
    x_valid, y_valid = validation[feature_columns], validation[target]

    for model_name, (estimator, grid) in candidate_model_grid(random_state).items():
        for params in grid:
            pipeline = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )
            pipeline.set_params(**params)
            run_name = f"{model_name}:" + ",".join(f"{key}={value}" for key, value in params.items())
            pipeline.fit(x_train, y_train)
            predicted_pd = pipeline.predict_proba(x_valid)[:, 1]
            auc = roc_auc_score(y_valid, predicted_pd)
            average_precision = average_precision_score(y_valid, predicted_pd)
            brier = brier_score_loss(y_valid, predicted_pd)
            records.append(
                {
                    "model_name": model_name,
                    "run_name": run_name,
                    "roc_auc": auc,
                    "average_precision": average_precision,
                    "brier_score": brier,
                }
            )
            fitted_models[run_name] = pipeline

    leaderboard = pd.DataFrame(records).sort_values(
        ["roc_auc", "average_precision", "brier_score"],
        ascending=[False, False, True],
    )
    best_name = str(leaderboard.iloc[0]["run_name"])
    return ModelSelectionResult(
        best_name=best_name,
        best_model=fitted_models[best_name],
        leaderboard=leaderboard.reset_index(drop=True),
    )


def calibrate_model(
    model: Pipeline,
    train_validation: pd.DataFrame,
    feature_columns: list[str],
    target: str,
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(estimator=clone(model), method="sigmoid", cv=3)
    calibrated.fit(train_validation[feature_columns], train_validation[target])
    return calibrated


def predict_pd(model: object, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict_proba(frame[feature_columns])[:, 1]
