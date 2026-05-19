# Executive Summary

## Objective

This project builds a credit risk decisioning framework for estimating probability of default, ranking customers by risk, optimizing approval policy, and quantifying expected loss under stress scenarios.

## Business Context

For credit card and fintech lenders, risk decisions require more than classification accuracy. The core management questions are:

- Which accounts are likely to default?
- Where is expected loss concentrated?
- What approval cutoff balances loss prevention and profitable growth?
- How does expected loss change under adverse repayment conditions?

## Approach

The project uses the UCI Default of Credit Card Clients dataset and constructs a professional risk workflow:

- Data quality checks and schema validation.
- Credit risk feature engineering from repayment, utilization, and payment behavior.
- Champion-challenger PD modeling using logistic regression, random forest, and gradient boosting.
- Probability calibration and threshold optimization.
- Risk-band reporting, SQL analytics, and Streamlit dashboarding.
- Stress testing for mild, adverse, and severe repayment deterioration.

## Management Insights

The verified model run selected histogram gradient boosting and produced:

| Metric | Value |
|---|---:|
| Test ROC AUC | 0.781 |
| Average precision | 0.560 |
| Brier score | 0.134 |
| KS statistic | 0.427 |
| Cost-optimized PD threshold | 0.094 |

The framework surfaces four high-value insights:

1. Recent delinquency is the strongest risk signal, followed by six-month delinquency count and utilization.
2. Expected loss is usually concentrated in a small number of high-PD accounts and bands.
3. A model threshold should be selected using credit economics, not a default ML cutoff.
4. Stress testing can materially change portfolio loss estimates and approval strategy.

## Stress Test Results

| Scenario | Avg PD | Expected Loss Rate | Incremental Expected Loss |
|---|---:|---:|---:|
| Base | 22.2% | 7.8% | 0 |
| Mild recession | 22.8% | 8.0% | 12.5M |
| Adverse | 36.0% | 13.3% | 277.3M |
| Severe | 56.9% | 23.6% | 798.0M |

## Recommendations

- Use risk bands for policy communication and governance.
- Route high-risk bands to manual review or stricter limit management.
- Add early-warning treatment for recent delinquency plus high utilization.
- Maintain monthly calibration and population-stability monitoring.
- Add fairness, explainability, and adverse-action review before any production use.

## Deliverables

- Python PD model pipeline.
- SQL analytics mart and monitoring queries.
- Streamlit dashboard.
- Comprehensive executive and technical documentation.
- Reproducible model training and reporting assets.
