# Streamlit Dashboard

The dashboard is the presentation layer for the credit risk project. It uses the artifacts written by:

```bash
python scripts/run_pipeline.py --config config/model_config.yaml
```

Launch it with:

```bash
streamlit run app.py
```

## Pages

- **Executive:** portfolio PD, expected loss, exposure, risk-band mix.
- **Policy Simulator:** PD cutoff slider with approval rate, captured bads, and expected loss.
- **Stress Testing:** base, mild, adverse, and severe loss scenarios.
- **Governance:** population stability, segment calibration, and monitoring views.
- **Model Diagnostics:** ROC, PR, calibration, confusion matrix, feature importance.
- **Batch Scoring:** upload a CSV with the same schema to score new accounts.
