from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_roc_curve(y_true, predicted_pd, output_path: str | Path) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    RocCurveDisplay.from_predictions(y_true, predicted_pd, ax=ax, color="#1F7A8C")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#778899", linewidth=1)
    ax.set_title("ROC Curve - Probability of Default Model")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_precision_recall_curve(y_true, predicted_pd, output_path: str | Path) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    PrecisionRecallDisplay.from_predictions(y_true, predicted_pd, ax=ax, color="#D1495B")
    ax.set_title("Precision-Recall Curve - Default Detection")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_confusion_matrix(y_true, predicted_pd, threshold: float, output_path: str | Path) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    y_pred = (pd.Series(predicted_pd) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix at PD Threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_calibration_curve(y_true, predicted_pd, output_path: str | Path) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    CalibrationDisplay.from_predictions(y_true, predicted_pd, n_bins=10, ax=ax, color="#1F7A8C")
    ax.set_title("Calibration Curve - Predicted PD vs Observed Default")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_feature_importance(importance: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    output = _prepare_output(output_path)
    top = importance.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(top["feature"], top["importance_mean"], color="#2A9D8F")
    ax.set_title("Top Feature Importance - Permutation AUC Impact")
    ax.set_xlabel("Mean AUC decrease")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_stress_test_chart(stress_results: pd.DataFrame, output_path: str | Path) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    sns.barplot(
        data=stress_results,
        x="scenario",
        y="expected_loss",
        hue="scenario",
        palette=["#1F7A8C", "#72B7B2", "#E9C46A", "#D1495B"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Expected Loss Under Stress Scenarios")
    ax.set_xlabel("")
    ax.set_ylabel("Expected loss")
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

