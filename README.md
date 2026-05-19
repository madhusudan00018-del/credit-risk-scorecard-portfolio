# Credit Risk Scorecard & Portfolio Analytics Platform

[![Streamlit App](https://static.streamlit.io/badge-hosted-badge.svg)](https://credit-risk-scorecard-portfolio-zunilcwekhsefzybspcwts.streamlit.app/)

An end-to-end, production-grade credit risk modeling and portfolio analytics platform. The system simulates a retail bank's risk decisioning infrastructure—incorporating machine learning Probability of Default (PD) scorecards, cost-optimized underwriting decision boundaries, portfolio stress testing under adverse macroeconomic scenarios, and automated data monitoring (PSI drift) with SQL.

**🔗 Live Interactive Dashboard:** [Credit Risk Command Center](https://credit-risk-scorecard-portfolio-zunilcwekhsefzybspcwts.streamlit.app/)

---

## 💻 Architecture & Data Flow

```text
       [Raw Customer Data] (UCI Credit Card Clients Dataset)
                │
                ▼ (Download & Clean)
      [Feature Engineering Layer] (Utilization, trend, & payment momentum)
                │
                ▼ (Train & Calibrate)
     [HistGradientBoosting PD Model] ───► [Sigmoid Calibration Layer]
                │
                ▼ (Batch Scoring)
     [SQLite Risk Analytics Mart] ◄───── [Governance & Stability Index (PSI)]
                │
                ▼ (Interactive Deployment)
     [Streamlit Command Center App] ◄─── [Macroeconomic Stress Simulator]
```

---

## 🚀 Core Platform Features

### 1. Advanced Credit Risk Modeling (PD Scorecard)
* **Model Pipeline:** `HistGradientBoostingClassifier` optimized using scikit-learn pipelines with custom feature engineering.
* **Probability Calibration:** Applied Sigmoid calibration to convert model raw scores into true probabilities of default (PD) for regulatory and financial reporting.
* **Explainability:** Calculated permutation feature importance using validation AUC degradation to provide industry-standard risk explanations.

### 2. Business Decisioning & Policy Simulation
* **Threshold Optimization:** Implemented a financial cost-minimization function comparing the opportunity cost of credit declines (False Positives) against actual default charge-offs (False Negatives).
* **Policy Simulator:** An interactive widget to simulate changes in credit policy cutoffs and dynamically report expected approval rates, bad capture rates, and portfolio expected losses.

### 3. Macroeconomic Portfolio Stress Testing
* **Stress Scenarios:** Simulated mild, adverse, and severe risk shocks (escalating billing, reducing payment ratios, and degrading repayment status).
* **Impact Assessment:** Calculates stressed portfolio PD distributions, Value-at-Risk (90th percentile PD), expected loss rate, and incremental credit reserves required under stress.

### 4. Data Governance & Model Monitoring
* **Population Stability Index (PSI):** Automated computation of PSI drift to detect feature distribution shifts between training and production runs.
* **SQL Mart:** An analytical data mart constructed in SQLite, tracking segment-level calibration, approval-rate compliance, and key performance indicators.

---

## 🛠️ Replicating & Running the Project

Follow these step-by-step instructions to set up your environment, download the source data, run the ML pipeline, build the analytics database, and launch the interactive dashboard locally.

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed on your system.

### 2. Clone the Repository & Install Dependencies
Clone the repository and install the required libraries:
```bash
# Clone the repository
git clone https://github.com/madhusudan00018-del/credit-risk-scorecard-portfolio.git
cd credit-risk-scorecard-portfolio

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies and local packages
pip install -r requirements.txt
pip install -e .
```

### 3. Step 1: Download the Source Dataset
Download the original UCI Default of Credit Card Clients dataset using the automated utility script:
```bash
python src/utils/download_data.py
```
*This downloads the raw CSV files into `data/raw/`.*

### 4. Step 2: Run the Machine Learning Pipeline
Clean, engineer features, train, calibrate, explain, and evaluate the model:
```bash
python src/utils/run_pipeline.py --config config/model_config.yaml
```
*This runs the full train-test split, serializes the pre-trained model into `models/pd_model.joblib`, generates evaluations, and saves calculated outputs to `data/processed/`.*

### 5. Step 3: Build the SQLite Analytics Mart
Construct the SQLite database and extract operational KPIs:
```bash
python src/utils/build_sqlite_mart.py
```
*This creates `data/processed/credit_risk.sqlite` and runs the schema definitions and KPI queries located in `sql/`.*

### 6. Step 4: Run the Interactive Streamlit Dashboard
Launch the dashboard command center locally:
```bash
streamlit run app.py
```
*The application will open automatically in your browser at `http://localhost:8501`.*

### 7. Step 5: Verify with Unit Tests
Execute the pytest suite to ensure all modeling, feature engineering, and governance layers compile and function correctly:
```bash
pytest
```

---

## 📊 Tech Stack & Infrastructure

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-learn, Joblib
* **Data Processing & Analytics:** Pandas, NumPy, SQLite, SQL
* **Interactive UI:** Streamlit, Plotly Express
* **Testing & Quality Assurance:** Pytest, Ruff
* **Environment:** VS Code Dev Containers (configured in `.devcontainer/`)

---

## 📄 License & Attribution

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

* **Author:** Madhusudan Yadav
* **Email:** [madhusudan00018@gmail.com](mailto:madhusudan00018@gmail.com)
* **GitHub:** [madhusudan00018-del](https://github.com/madhusudan00018-del)
