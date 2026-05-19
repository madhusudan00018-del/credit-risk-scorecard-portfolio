-- Credit risk analytics mart schema.
-- The Python mart builder writes equivalent tables into SQLite.

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    limit_bal REAL NOT NULL,
    sex INTEGER,
    sex_label TEXT,
    education INTEGER,
    education_label TEXT,
    marriage INTEGER,
    marriage_label TEXT,
    age INTEGER,
    default_next_month INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_statements (
    customer_id INTEGER NOT NULL,
    statement_month DATE NOT NULL,
    recency_rank INTEGER NOT NULL,
    repayment_status INTEGER,
    bill_amount REAL,
    payment_amount REAL,
    delinquent_flag INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS model_scores (
    customer_id INTEGER,
    predicted_pd REAL,
    risk_band TEXT,
    decision TEXT,
    ead REAL,
    expected_loss REAL,
    default_next_month INTEGER,
    limit_bal REAL,
    age INTEGER,
    education_label TEXT,
    marriage_label TEXT,
    sex_label TEXT
);

CREATE TABLE IF NOT EXISTS risk_band_summary (
    risk_band TEXT,
    accounts INTEGER,
    exposure REAL,
    avg_pd REAL,
    observed_default_rate REAL,
    expected_loss REAL,
    approval_rate REAL,
    exposure_share REAL,
    loss_share REAL
);

CREATE TABLE IF NOT EXISTS stress_test_results (
    scenario TEXT,
    accounts INTEGER,
    total_exposure REAL,
    avg_pd REAL,
    median_pd REAL,
    p90_pd REAL,
    expected_loss REAL,
    expected_loss_rate REAL,
    approval_rate REAL,
    incremental_expected_loss REAL
);

CREATE INDEX IF NOT EXISTS idx_monthly_customer
ON monthly_statements(customer_id);

CREATE INDEX IF NOT EXISTS idx_model_scores_risk_band
ON model_scores(risk_band);

