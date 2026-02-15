-- PostgreSQL schema for ATP betting model
-- Tracks all bets, predictions, and performance metrics

CREATE TABLE IF NOT EXISTS bets (
    id              SERIAL PRIMARY KEY,
    match_id        VARCHAR(64) NOT NULL,
    date            DATE NOT NULL,
    tournament      VARCHAR(128),
    player_name     VARCHAR(128) NOT NULL,
    opponent_name   VARCHAR(128),
    model_prob      DECIMAL(5,4) NOT NULL,
    decimal_odds    DECIMAL(6,2) NOT NULL,
    implied_prob    DECIMAL(5,4),
    expected_value  DECIMAL(6,4),
    kelly_fraction  DECIMAL(6,4),
    stake           DECIMAL(10,2) NOT NULL,
    bankroll_before DECIMAL(12,2),

    -- Result (filled after match)
    won             BOOLEAN,
    profit          DECIMAL(10,2),
    bankroll_after  DECIMAL(12,2),

    -- Metadata
    model_version   VARCHAR(64),
    threshold       DECIMAL(5,4),
    flag            VARCHAR(32),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at      TIMESTAMP,

    UNIQUE (match_id, player_name)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(date);
CREATE INDEX IF NOT EXISTS idx_bets_match ON bets(match_id);
CREATE INDEX IF NOT EXISTS idx_bets_settled ON bets(settled_at);

-- Performance summary view
CREATE OR REPLACE VIEW bet_performance AS
SELECT
    DATE_TRUNC('month', date) AS month,
    COUNT(*)                   AS total_bets,
    SUM(CASE WHEN won THEN 1 ELSE 0 END) AS wins,
    ROUND(AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) * 100, 1) AS win_rate_pct,
    SUM(profit)                AS total_profit,
    ROUND(SUM(profit) / NULLIF(SUM(stake), 0) * 100, 1) AS roi_pct,
    MAX(bankroll_after)        AS max_bankroll,
    MIN(bankroll_after)        AS min_bankroll
FROM bets
WHERE settled_at IS NOT NULL
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;
