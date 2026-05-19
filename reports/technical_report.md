# Technical Report

## 1. Problem Definition

The target variable is `default_next_month`, a binary indicator for whether a customer defaults in the following month. The modeling task is supervised binary classification, but the business output is calibrated probability of default and expected loss.

## 2. Data

The project uses the UCI Default of Credit Card Clients dataset with 30,000 customers and repayment, bill, payment, credit limit, and demographic variables.

Data treatment:

- Standardize column names.
- Validate required schema.
- Collapse undocumented categorical values.
- Retain negative bill amounts because they can indicate overpayment or credit balances.
- Engineer utilization, payment intensity, delinquency, trend, and segment features.

## 3. Feature Engineering

The feature engineering layer creates:

- utilization ratios,
- payment-to-bill and payment-to-limit ratios,
- count and severity of delinquent months,
- recent delinquency flags,
- bill amount trend,
- payment momentum,
- high-risk interaction flags,
- age and limit segments,
- winsorized and log-transformed financial features.

## 4. Modeling

Candidate models:

- Logistic regression with class balancing.
- Random forest with balanced subsampling.
- Histogram gradient boosting.

Model selection uses validation ROC AUC, average precision, and Brier score. The selected model is calibrated using sigmoid calibration and evaluated on a held-out test set.

Top validation challengers from the verified run:

| Rank | Model | Validation ROC AUC | Average Precision | Brier Score |
|---:|---|---:|---:|---:|
| 1 | Histogram gradient boosting | 0.787 | 0.560 | 0.134 |
| 2 | Random forest | 0.785 | 0.556 | 0.173 |
| 3 | Histogram gradient boosting | 0.785 | 0.553 | 0.134 |

## 5. Evaluation

Verified test metrics:

| Metric | Value |
|---|---:|
| ROC AUC | 0.781 |
| Average precision | 0.560 |
| Brier score | 0.134 |
| KS statistic | 0.427 |
| Precision | 0.280 |
| Recall | 0.928 |
| F1 score | 0.431 |
| Selected PD threshold | 0.094 |

Confusion matrix at the selected threshold:

| | Predicted good | Predicted bad |
|---|---:|---:|
| Actual good | 1,512 | 3,161 |
| Actual bad | 95 | 1,232 |

The project writes:

- `data/processed/model_leaderboard.csv`
- `data/processed/threshold_cost_curve.csv`
- `data/processed/decile_lift_table.csv`
- `data/processed/feature_importance.csv`
- `data/processed/model_governance_psi.csv`
- `data/processed/segment_performance.csv`

## 6. Threshold Optimization

Thresholds are evaluated using a business cost function:

```text
False negative cost = bad borrower approved x EAD x loss cost rate
False positive cost = good borrower declined x EAD x opportunity cost rate
```

This is more appropriate than maximizing accuracy because the cost of approving a future defaulter is much larger than the cost of reviewing a good customer.

## 7. Explainability

Permutation importance is computed for every feature using AUC degradation. SHAP summary charts are generated when the optional `shap` package is installed.

The explanation layer supports:

- model validation,
- stakeholder communication,
- reason-code thinking,
- monitoring of unstable or non-actionable signals.

## 8. Stress Testing

The stress module applies scenario shocks:

- increased bill amounts,
- reduced payment amounts,
- worse repayment status.

The model re-scores the stressed portfolio and calculates:

- average PD,
- p90 PD,
- expected loss,
- expected loss rate,
- approval rate,
- incremental expected loss.

## 9. Deployment

The dashboard is implemented in Streamlit and can be deployed locally, through Docker, or on Render. The app reads saved model artifacts and scoring files from the pipeline.

## 10. Model Risk Notes

This is a portfolio demonstration model. Production use would require:

- out-of-time validation,
- population stability monitoring,
- adverse-action reason codes,
- fairness review,
- data lineage,
- approval from model risk management,
- challenger monitoring after deployment.

The repository now includes first-pass model governance outputs:

- population stability index for feature distribution shift,
- segment-level calibration gaps,
- approval-rate monitoring by demographic and risk segments,
- dashboard governance tab for analyst review.
