# Model Card

## Model Name

Credit Card Probability of Default Model

## Intended Use

Estimate next-month probability of default for credit card customers and support portfolio monitoring, risk banding, threshold simulation, and stress testing.

## Not Intended For

- Fully automated production underwriting without governance.
- Legal adverse-action notices without validated reason-code mapping.
- Use on populations that materially differ from the source dataset without recalibration.

## Training Data

UCI Default of Credit Card Clients dataset.

## Inputs

Credit limit, demographic variables, six months of repayment status, six months of bill amounts, six months of payment amounts, and engineered behavior features.

## Output

`predicted_pd`, a calibrated probability of next-month default.

## Evaluation

Metrics are written to `models/model_metrics.json`:

- ROC AUC
- Average precision
- Brier score
- KS statistic
- Precision, recall, F1
- Confusion matrix
- Cost-minimizing threshold

## Ethical and Compliance Considerations

Demographic variables can create fairness and compliance concerns. A production implementation should include bias testing, explainability, adverse-action reason code validation, and policy approval.

## Monitoring Plan

- Monthly calibration by PD decile.
- Population distribution by risk band.
- Approval rate and bad capture rate.
- Feature drift for delinquency, utilization, payment ratios, and limit segments.
- Expected loss under base and adverse scenarios.

