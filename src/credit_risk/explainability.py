from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(
    model: object,
    x: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    n_repeats: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x[feature_columns],
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def try_write_shap_summary(
    model: object,
    x: pd.DataFrame,
    feature_columns: list[str],
    output_path: str | Path,
    max_rows: int = 800,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return None

    sample = x[feature_columns].sample(min(len(x), max_rows), random_state=42)
    explainer = shap.Explainer(model.predict_proba, sample)
    shap_values = explainer(sample)
    shap.plots.beeswarm(shap_values[:, :, 1], show=False, max_display=20)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()
    return str(output)
