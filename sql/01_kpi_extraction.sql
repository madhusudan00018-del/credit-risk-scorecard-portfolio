-- Executive KPIs: portfolio risk, losses, and decisioning performance.

WITH portfolio AS (
    SELECT
        COUNT(*) AS accounts,
        SUM(ead) AS exposure,
        AVG(predicted_pd) AS avg_pd,
        SUM(expected_loss) AS expected_loss,
        AVG(CASE WHEN decision = 'approve' THEN 1.0 ELSE 0.0 END) AS approval_rate,
        AVG(default_next_month * 1.0) AS observed_default_rate
    FROM model_scores
),
loss_density AS (
    SELECT
        risk_band,
        COUNT(*) AS accounts,
        SUM(ead) AS exposure,
        SUM(expected_loss) AS expected_loss,
        AVG(predicted_pd) AS avg_pd,
        AVG(default_next_month * 1.0) AS observed_default_rate
    FROM model_scores
    GROUP BY risk_band
)
SELECT
    p.accounts,
    ROUND(p.exposure, 2) AS exposure,
    ROUND(p.avg_pd, 4) AS avg_pd,
    ROUND(p.observed_default_rate, 4) AS observed_default_rate,
    ROUND(p.expected_loss, 2) AS expected_loss,
    ROUND(p.expected_loss / NULLIF(p.exposure, 0), 4) AS expected_loss_rate,
    ROUND(p.approval_rate, 4) AS approval_rate,
    ld.risk_band AS highest_loss_band,
    ROUND(ld.expected_loss, 2) AS highest_band_expected_loss
FROM portfolio p
CROSS JOIN (
    SELECT *
    FROM loss_density
    ORDER BY expected_loss DESC
    LIMIT 1
) ld;

-- Risk-band concentration.
SELECT
    risk_band,
    accounts,
    ROUND(exposure, 2) AS exposure,
    ROUND(100.0 * exposure_share, 2) AS exposure_share_pct,
    ROUND(avg_pd, 4) AS avg_pd,
    ROUND(observed_default_rate, 4) AS observed_default_rate,
    ROUND(expected_loss, 2) AS expected_loss,
    ROUND(100.0 * loss_share, 2) AS loss_share_pct
FROM risk_band_summary
ORDER BY avg_pd;

