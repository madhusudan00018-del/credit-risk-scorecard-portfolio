from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from credit_risk.config import PROJECT_ROOT
from credit_risk.data import standardize_credit_columns
from credit_risk.evaluation import build_scored_frame, threshold_cost_curve
from credit_risk.features import add_credit_risk_features
from credit_risk.models import predict_pd
from credit_risk.preprocessing import clean_credit_data


ARTIFACT_PATHS = {
    "scores": PROJECT_ROOT / "data/processed/model_scores.csv",
    "bands": PROJECT_ROOT / "data/processed/risk_band_summary.csv",
    "stress": PROJECT_ROOT / "data/processed/stress_test_results.csv",
    "psi": PROJECT_ROOT / "data/processed/model_governance_psi.csv",
    "segment_performance": PROJECT_ROOT / "data/processed/segment_performance.csv",
    "metrics": PROJECT_ROOT / "models/model_metrics.json",
    "metadata": PROJECT_ROOT / "models/model_metadata.json",
    "model": PROJECT_ROOT / "models/pd_model.joblib",
}


def _page_config() -> None:
    st.set_page_config(
        page_title="Credit Risk Command Center",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return joblib.load(path)


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def artifacts_ready() -> bool:
    return all(path.exists() for path in [ARTIFACT_PATHS["scores"], ARTIFACT_PATHS["model"]])


def image_full_width(container, path: Path) -> None:
    try:
        container.image(str(path), use_container_width=True)
    except TypeError:
        container.image(str(path), use_column_width=True)


def metric_row(scores: pd.DataFrame, metrics: dict[str, object]) -> None:
    total_exposure = scores["ead"].sum()
    expected_loss = scores["expected_loss"].sum()
    avg_pd = scores["predicted_pd"].mean()
    approval_rate = (scores["decision"] == "approve").mean()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accounts", f"{len(scores):,.0f}")
    col2.metric("Exposure", f"{total_exposure:,.0f}")
    col3.metric("Avg PD", f"{avg_pd:.2%}")
    col4.metric("Expected Loss", f"{expected_loss:,.0f}")
    col5.metric("Approval Rate", f"{approval_rate:.1%}", help=f"AUC {metrics.get('roc_auc', 0):.3f}")


def executive_page(scores: pd.DataFrame, bands: pd.DataFrame, metrics: dict[str, object]) -> None:
    metric_row(scores, metrics)
    left, right = st.columns([1.15, 0.85])
    with left:
        band_order = bands.sort_values("avg_pd")
        fig = px.bar(
            band_order,
            x="risk_band",
            y="expected_loss",
            color="avg_pd",
            color_continuous_scale=["#2A9D8F", "#E9C46A", "#D1495B"],
            title="Expected Loss by Risk Band",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Expected loss", height=420)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(
            bands,
            values="exposure",
            names="risk_band",
            title="Exposure Mix",
            color_discrete_sequence=px.colors.qualitative.Safe,
            hole=0.52,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    segment_cols = st.columns(3)
    for idx, dimension in enumerate(["education_label", "marriage_label", "age_band"]):
        segment = (
            scores.groupby(dimension, observed=True)
            .agg(avg_pd=("predicted_pd", "mean"), accounts=("customer_id", "count"))
            .reset_index()
            .sort_values("avg_pd", ascending=False)
        )
        fig = px.bar(
            segment,
            x=dimension,
            y="avg_pd",
            color="accounts",
            color_continuous_scale=["#EAF4F4", "#1F7A8C"],
            title=f"PD by {dimension.replace('_', ' ').title()}",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Average PD", height=340)
        segment_cols[idx].plotly_chart(fig, use_container_width=True)


def policy_page(scores: pd.DataFrame) -> None:
    threshold = st.slider("PD Cutoff", min_value=0.03, max_value=0.60, value=0.18, step=0.01)
    cost_curve = threshold_cost_curve(
        scores["default_next_month"],
        scores["predicted_pd"],
        scores["ead"],
        false_negative_cost_rate=0.45,
        false_positive_cost_rate=0.025,
        thresholds=np.linspace(0.03, 0.60, 116),
    )
    selected = cost_curve.iloc[(cost_curve["threshold"] - threshold).abs().argsort()[:1]].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Approval Rate", f"{selected['approval_rate']:.1%}")
    col2.metric("Bad Capture", f"{selected['bad_capture_rate']:.1%}")
    col3.metric("Expected Cost", f"{selected['expected_cost']:,.0f}")
    col4.metric("False Negatives", f"{selected['false_negative_count']:,.0f}")

    fig = px.line(
        cost_curve,
        x="threshold",
        y=["expected_cost", "approval_rate", "bad_capture_rate"],
        title="Policy Tradeoff Curve",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#D1495B")
    fig.update_layout(height=470, yaxis_title="Value", legend_title="")
    st.plotly_chart(fig, use_container_width=True)


def stress_page(stress: pd.DataFrame) -> None:
    fig = px.bar(
        stress,
        x="scenario",
        y="expected_loss",
        color="expected_loss_rate",
        color_continuous_scale=["#2A9D8F", "#E9C46A", "#D1495B"],
        title="Expected Loss Under Portfolio Stress",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Expected loss", height=440)
    st.plotly_chart(fig, use_container_width=True)
    display_cols = [
        "scenario",
        "avg_pd",
        "p90_pd",
        "expected_loss_rate",
        "approval_rate",
        "incremental_expected_loss",
    ]
    st.dataframe(
        stress[display_cols].style.format(
            {
                "avg_pd": "{:.2%}",
                "p90_pd": "{:.2%}",
                "expected_loss_rate": "{:.2%}",
                "approval_rate": "{:.1%}",
                "incremental_expected_loss": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )


def governance_page(psi: pd.DataFrame, segment_performance: pd.DataFrame) -> None:
    st.subheader("Model Governance Monitor")
    if psi.empty or segment_performance.empty:
        st.warning("Governance artifacts are unavailable. Re-run the pipeline.")
        return

    top_psi = psi.head(15).sort_values("psi")
    left, right = st.columns([0.95, 1.05])
    with left:
        fig = px.bar(
            top_psi,
            x="psi",
            y="feature",
            color="status",
            orientation="h",
            title="Population Stability Index by Feature",
            color_discrete_map={
                "stable": "#2A9D8F",
                "watch": "#E9C46A",
                "material_shift": "#D1495B",
            },
        )
        fig.update_layout(height=520, xaxis_title="PSI", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        segment_choice = st.selectbox(
            "Segment View",
            options=sorted(segment_performance["segment"].dropna().unique()),
            index=0,
        )
        selected = segment_performance[segment_performance["segment"] == segment_choice].copy()
        selected = selected.sort_values("expected_loss", ascending=False)
        fig = px.bar(
            selected,
            x="segment_value",
            y="calibration_gap",
            color="expected_loss",
            color_continuous_scale=["#2A9D8F", "#E9C46A", "#D1495B"],
            title="Calibration Gap by Segment",
        )
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Observed default - predicted PD")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            selected[
                [
                    "segment_value",
                    "accounts",
                    "avg_pd",
                    "observed_default_rate",
                    "calibration_gap",
                    "approval_rate",
                    "expected_loss",
                ]
            ].style.format(
                {
                    "avg_pd": "{:.2%}",
                    "observed_default_rate": "{:.2%}",
                    "calibration_gap": "{:.2%}",
                    "approval_rate": "{:.1%}",
                    "expected_loss": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )


def diagnostics_page(metrics: dict[str, object]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
    col2.metric("Avg Precision", f"{metrics.get('average_precision', 0):.3f}")
    col3.metric("Brier Score", f"{metrics.get('brier_score', 0):.3f}")
    col4.metric("KS", f"{metrics.get('ks_statistic', 0):.3f}")

    image_paths = [
        PROJECT_ROOT / "images/roc_curve.png" if (PROJECT_ROOT / "images/roc_curve.png").exists() else PROJECT_ROOT / "assets/roc_curve.png",
        PROJECT_ROOT / "images/precision_recall_curve.png" if (PROJECT_ROOT / "images/precision_recall_curve.png").exists() else PROJECT_ROOT / "assets/precision_recall_curve.png",
        PROJECT_ROOT / "images/calibration_curve.png" if (PROJECT_ROOT / "images/calibration_curve.png").exists() else PROJECT_ROOT / "assets/calibration_curve.png",
        PROJECT_ROOT / "images/confusion_matrix.png" if (PROJECT_ROOT / "images/confusion_matrix.png").exists() else PROJECT_ROOT / "assets/confusion_matrix.png",
        PROJECT_ROOT / "images/feature_importance.png" if (PROJECT_ROOT / "images/feature_importance.png").exists() else PROJECT_ROOT / "assets/feature_importance.png",
        PROJECT_ROOT / "images/stress_test_expected_loss.png" if (PROJECT_ROOT / "images/stress_test_expected_loss.png").exists() else PROJECT_ROOT / "assets/stress_test_expected_loss.png",
    ]
    for row_start in range(0, len(image_paths), 2):
        left, right = st.columns(2)
        for column, path in zip([left, right], image_paths[row_start : row_start + 2]):
            if path.exists():
                image_full_width(column, path)


def batch_scoring_page(model_artifact: dict[str, object] | None) -> None:
    if model_artifact is None:
        st.warning("Model artifact is unavailable.")
        return
    uploaded = st.file_uploader("Applicant CSV", type=["csv"])
    if uploaded is None:
        return
    incoming = pd.read_csv(uploaded)
    standardized = standardize_credit_columns(incoming)
    clean = clean_credit_data(standardized)
    features = add_credit_risk_features(clean)
    feature_columns = model_artifact["feature_columns"]
    predicted_pd = predict_pd(model_artifact["model"], features, feature_columns)
    scored = build_scored_frame(
        features,
        predicted_pd,
        float(model_artifact["threshold"]),
        model_artifact["risk_bands"],
    )
    st.dataframe(
        scored[["customer_id", "predicted_pd", "risk_band", "decision", "expected_loss"]].sort_values(
            "predicted_pd", ascending=False
        ),
        use_container_width=True,
    )


def main() -> None:
    _page_config()
    st.title("Credit Risk Command Center")

    if not artifacts_ready():
        st.error("Dashboard artifacts are not available. Run the modeling pipeline first.")
        return

    scores = load_csv(ARTIFACT_PATHS["scores"])
    bands = load_csv(ARTIFACT_PATHS["bands"]) if ARTIFACT_PATHS["bands"].exists() else pd.DataFrame()
    stress = load_csv(ARTIFACT_PATHS["stress"]) if ARTIFACT_PATHS["stress"].exists() else pd.DataFrame()
    psi = load_csv(ARTIFACT_PATHS["psi"]) if ARTIFACT_PATHS["psi"].exists() else pd.DataFrame()
    segment_performance = (
        load_csv(ARTIFACT_PATHS["segment_performance"])
        if ARTIFACT_PATHS["segment_performance"].exists()
        else pd.DataFrame()
    )
    metrics = load_json(ARTIFACT_PATHS["metrics"])
    model_artifact = load_model(ARTIFACT_PATHS["model"])

    with st.sidebar:
        st.caption("Portfolio Filters")
        selected_bands = st.multiselect(
            "Risk Band",
            options=sorted(scores["risk_band"].dropna().unique()),
            default=sorted(scores["risk_band"].dropna().unique()),
        )
        selected_decision = st.multiselect(
            "Decision",
            options=sorted(scores["decision"].dropna().unique()),
            default=sorted(scores["decision"].dropna().unique()),
        )

    filtered = scores[
        scores["risk_band"].isin(selected_bands) & scores["decision"].isin(selected_decision)
    ].copy()
    filtered_bands = (
        filtered.groupby("risk_band", observed=True)
        .agg(
            accounts=("customer_id", "count"),
            exposure=("ead", "sum"),
            avg_pd=("predicted_pd", "mean"),
            observed_default_rate=("default_next_month", "mean"),
            expected_loss=("expected_loss", "sum"),
            approval_rate=("decision", lambda x: (x == "approve").mean()),
        )
        .reset_index()
        if len(filtered)
        else bands.head(0)
    )

    tabs = st.tabs(
        [
            "Executive",
            "Policy Simulator",
            "Stress Testing",
            "Governance",
            "Model Diagnostics",
            "Batch Scoring",
        ]
    )
    with tabs[0]:
        executive_page(filtered, filtered_bands, metrics)
    with tabs[1]:
        policy_page(filtered)
    with tabs[2]:
        stress_page(stress)
    with tabs[3]:
        governance_page(psi, segment_performance)
    with tabs[4]:
        diagnostics_page(metrics)
    with tabs[5]:
        batch_scoring_page(model_artifact)


if __name__ == "__main__":
    main()
