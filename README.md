# Credit Risk Scorecard & Portfolio Analytics Platform

## Live Demo
https://credit-risk-scorecard-portfolio.onrender.com/

## GitHub Repository
https://github.com/madhusudan00018-del/credit-risk-scorecard-portfolio

---

## Overview

Production-grade machine learning credit risk platform for:

- Probability of Default (PD) estimation
- Risk segmentation
- Batch applicant scoring
- Portfolio stress testing
- Governance analytics
- Executive dashboards

The platform simulates operational banking risk infrastructure using machine learning, portfolio analytics, and deployment engineering.

---

## Features

### Credit Risk Modeling
- HistGradientBoostingClassifier PD model
- Risk band classification
- Approval/review decision engine
- Expected loss estimation

### Governance & Monitoring
- Population Stability Index (PSI)
- Drift monitoring
- Calibration diagnostics
- Segment performance analysis

### Portfolio Analytics
- Stress testing scenarios
- Risk concentration analysis
- Portfolio segmentation
- Executive KPI dashboards

### Deployment
- Public cloud deployment on Render
- Streamlit interactive dashboard
- Batch scoring workflow
- Exportable scoring outputs

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | ~0.78 |
| KS Statistic | ~0.42 |
| Deployment | Render |
| Dataset | UCI Credit Default Dataset |

---

## Architecture

```text
Raw Data
   ↓
Feature Engineering
   ↓
ML Pipeline
   ↓
Governance & Monitoring
   ↓
Interactive Dashboard
   ↓
Cloud Deployment
```

---

## Repository Structure

```text
├── app.py
├── dashboard/
├── src/
├── models/
├── data/
├── reports/
├── sql/
├── tests/
└── assets/
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Sample Dataset

Example upload file for batch scoring:

```text
data/sample/sample_applicants.csv
```

---

## Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas
- Plotly
- SQL
- Render
- GitHub

---

## Author

Madhusudan Yadav
