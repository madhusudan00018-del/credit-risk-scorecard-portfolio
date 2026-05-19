-- Segment, cohort, and customer behavior analytics.

-- 1. Default and PD by demographic and credit limit cohort.
WITH limit_cohorts AS (
    SELECT
        customer_id,
        CASE
            WHEN limit_bal < 50000 THEN '01_under_50k'
            WHEN limit_bal < 100000 THEN '02_50k_100k'
            WHEN limit_bal < 200000 THEN '03_100k_200k'
            WHEN limit_bal < 350000 THEN '04_200k_350k'
            ELSE '05_350k_plus'
        END AS limit_cohort
    FROM customers
),
segment_base AS (
    SELECT
        lc.limit_cohort,
        c.education_label,
        c.marriage_label,
        ms.risk_band,
        ms.predicted_pd,
        ms.default_next_month,
        ms.ead,
        ms.expected_loss
    FROM model_scores ms
    JOIN customers c
        ON ms.customer_id = c.customer_id
    JOIN limit_cohorts lc
        ON ms.customer_id = lc.customer_id
)
SELECT
    limit_cohort,
    education_label,
    marriage_label,
    COUNT(*) AS accounts,
    ROUND(AVG(predicted_pd), 4) AS avg_pd,
    ROUND(AVG(default_next_month * 1.0), 4) AS observed_default_rate,
    ROUND(SUM(ead), 2) AS exposure,
    ROUND(SUM(expected_loss), 2) AS expected_loss
FROM segment_base
GROUP BY limit_cohort, education_label, marriage_label
HAVING COUNT(*) >= 100
ORDER BY expected_loss DESC;

-- 2. Rolling six-month delinquency behavior using window functions.
WITH monthly AS (
    SELECT
        customer_id,
        statement_month,
        recency_rank,
        repayment_status,
        bill_amount,
        payment_amount,
        delinquent_flag,
        SUM(delinquent_flag) OVER (
            PARTITION BY customer_id
            ORDER BY recency_rank
            ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING
        ) AS six_month_delinquent_months,
        MAX(repayment_status) OVER (
            PARTITION BY customer_id
            ORDER BY recency_rank
            ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING
        ) AS worst_six_month_status
    FROM monthly_statements
)
SELECT
    CASE
        WHEN six_month_delinquent_months = 0 THEN 'clean'
        WHEN six_month_delinquent_months BETWEEN 1 AND 2 THEN 'early_warning'
        WHEN six_month_delinquent_months BETWEEN 3 AND 4 THEN 'persistent_delinquency'
        ELSE 'severe_delinquency'
    END AS repayment_cohort,
    COUNT(DISTINCT m.customer_id) AS customers,
    ROUND(AVG(ms.predicted_pd), 4) AS avg_pd,
    ROUND(AVG(ms.default_next_month * 1.0), 4) AS observed_default_rate,
    ROUND(SUM(ms.expected_loss), 2) AS expected_loss
FROM monthly m
JOIN model_scores ms
    ON m.customer_id = ms.customer_id
WHERE m.recency_rank = 1
GROUP BY repayment_cohort
ORDER BY avg_pd DESC;

-- 3. Rank customers by expected loss within risk band for collections prioritization.
WITH ranked AS (
    SELECT
        customer_id,
        risk_band,
        predicted_pd,
        ead,
        expected_loss,
        ROW_NUMBER() OVER (
            PARTITION BY risk_band
            ORDER BY expected_loss DESC
        ) AS loss_rank_in_band,
        NTILE(10) OVER (
            ORDER BY predicted_pd DESC
        ) AS pd_decile_desc
    FROM model_scores
)
SELECT
    customer_id,
    risk_band,
    ROUND(predicted_pd, 4) AS predicted_pd,
    ROUND(ead, 2) AS ead,
    ROUND(expected_loss, 2) AS expected_loss,
    loss_rank_in_band,
    pd_decile_desc
FROM ranked
WHERE loss_rank_in_band <= 20
ORDER BY risk_band, loss_rank_in_band;

