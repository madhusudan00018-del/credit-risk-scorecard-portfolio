-- Monitoring queries for production-style model governance.

-- 1. Calibration by PD decile.
WITH deciles AS (
    SELECT
        customer_id,
        predicted_pd,
        default_next_month,
        NTILE(10) OVER (ORDER BY predicted_pd) AS pd_decile
    FROM model_scores
),
calibration AS (
    SELECT
        pd_decile,
        COUNT(*) AS accounts,
        AVG(predicted_pd) AS avg_predicted_pd,
        AVG(default_next_month * 1.0) AS observed_default_rate
    FROM deciles
    GROUP BY pd_decile
)
SELECT
    pd_decile,
    accounts,
    ROUND(avg_predicted_pd, 4) AS avg_predicted_pd,
    ROUND(observed_default_rate, 4) AS observed_default_rate,
    ROUND(observed_default_rate - avg_predicted_pd, 4) AS calibration_gap
FROM calibration
ORDER BY pd_decile;

-- 2. Population stability style distribution by risk band and segment.
WITH band_distribution AS (
    SELECT
        risk_band,
        education_label,
        COUNT(*) AS accounts,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS population_share,
        AVG(predicted_pd) AS avg_pd
    FROM model_scores
    GROUP BY risk_band, education_label
)
SELECT
    risk_band,
    education_label,
    accounts,
    ROUND(100.0 * population_share, 2) AS population_share_pct,
    ROUND(avg_pd, 4) AS avg_pd
FROM band_distribution
ORDER BY risk_band, population_share DESC;

-- 3. Stress scenario ranking.
SELECT
    scenario,
    ROUND(avg_pd, 4) AS avg_pd,
    ROUND(expected_loss_rate, 4) AS expected_loss_rate,
    ROUND(incremental_expected_loss, 2) AS incremental_expected_loss,
    RANK() OVER (ORDER BY expected_loss DESC) AS loss_rank
FROM stress_test_results
ORDER BY loss_rank;

