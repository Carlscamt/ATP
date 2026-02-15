"""
Walk-Forward Backtester

Expanding-window backtesting with weekly retraining.
Computes ROI, Sharpe ratio, max drawdown, and flags leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from betting.strategy import BettingStrategy, calculate_ev, fractional_kelly

logger = logging.getLogger("betting")

EXCLUDE_COLS = {
    "match_id", "date", "label", "winner", "player1_id", "player2_id",
    "player1_name", "player2_name", "player1_seed", "player2_seed",
    "score", "tournament_id", "tournament_name", "tournament_slug",
    "surface", "tier", "round", "location", "start_date", "year",
    "winner_flag", "scraped_at", "player1_odds", "player2_odds",
}


class WalkForwardBacktest:
    """
    Walk-forward backtesting with expanding training window.

    For each test period:
    1. Train on all data before test period
    2. Predict on test period
    3. Simulate bets using model predictions + odds
    4. Record P&L
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        retrain_frequency: str = "monthly",
        kelly_fraction: float = 0.25,
        max_stake_pct: float = 0.05,
        xgb_params: dict | None = None,
    ):
        self.initial_bankroll = initial_bankroll
        self.retrain_frequency = retrain_frequency
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.xgb_params = xgb_params or {
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "max_delta_step": 5,
            "learning_rate": 0.01,
            "max_depth": 6,
            "eval_metric": ["aucpr", "logloss"],
        }

        self.results: list[dict[str, Any]] = []
        self.bankroll_history: list[float] = []

    def run(self, data_path: str) -> dict[str, Any]:
        """
        Execute walk-forward backtest.

        Requires columns: date, label, player1_odds, player2_odds,
        plus all feature columns.
        """
        df = pl.read_parquet(data_path).sort("date")

        if "player1_odds" not in df.columns:
            logger.error("No odds data for backtesting — need 'player1_odds', 'player2_odds'")
            return {}

        # Determine feature columns
        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
        ]

        # Get unique test periods
        df = df.with_columns(
            pl.col("date").str.slice(0, 7).alias("year_month")
        )
        periods = df["year_month"].unique().sort().to_list()

        strategy = BettingStrategy(
            bankroll=self.initial_bankroll,
            kelly_fraction=self.kelly_fraction,
            max_stake_pct=self.max_stake_pct,
        )
        self.bankroll_history = [self.initial_bankroll]

        # Minimum training data: first 50% of periods
        min_train_periods = max(len(periods) // 2, 3)

        for i in range(min_train_periods, len(periods)):
            test_period = periods[i]
            train_periods = periods[:i]

            # Split
            train_df = df.filter(pl.col("year_month").is_in(train_periods))
            test_df = df.filter(pl.col("year_month") == test_period)

            if len(test_df) == 0:
                continue

            # Train model
            X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
            y_train = train_df["label"].to_numpy().astype(np.float32)
            X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)

            # Auto scale_pos_weight
            params = dict(self.xgb_params)
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            if n_pos > 0:
                params["scale_pos_weight"] = n_neg / n_pos

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
            dtest = xgb.DMatrix(X_test, feature_names=feature_cols)

            model = xgb.train(
                params, dtrain,
                num_boost_round=300,
                verbose_eval=False,
            )

            preds = model.predict(dtest)

            # Simulate bets
            test_rows = test_df.to_dicts()
            period_bets = 0
            period_profit = 0.0

            for j, row in enumerate(test_rows):
                prob_p1 = float(preds[j])
                p1_odds = float(row.get("player1_odds", 0))
                p2_odds = float(row.get("player2_odds", 0))
                actual_winner = row.get("winner", "")

                # Evaluate both sides
                for side, prob, odds in [
                    ("player1", prob_p1, p1_odds),
                    ("player2", 1 - prob_p1, p2_odds),
                ]:
                    if odds <= 0:
                        continue

                    ev = calculate_ev(prob, odds)
                    if ev <= 0:
                        continue

                    stake_pct = fractional_kelly(
                        prob, odds,
                        fraction=self.kelly_fraction,
                        max_stake_pct=self.max_stake_pct,
                    )
                    if stake_pct <= 0:
                        continue

                    stake = strategy.bankroll * stake_pct
                    won = actual_winner == side
                    strategy.record_result(
                        row.get("match_id", ""), won, stake, odds
                    )
                    period_bets += 1
                    period_profit += stake * (odds - 1) if won else -stake

            self.bankroll_history.append(strategy.bankroll)

            self.results.append({
                "period": test_period,
                "bets": period_bets,
                "profit": round(period_profit, 2),
                "bankroll": round(strategy.bankroll, 2),
                "train_size": len(train_df),
            })

            logger.info(
                "Period %s: %d bets, P&L=%.2f, Bankroll=%.2f",
                test_period, period_bets, period_profit, strategy.bankroll,
            )

        return self._compute_summary(strategy)

    def _compute_summary(self, strategy: BettingStrategy) -> dict[str, Any]:
        """Compute backtest performance metrics."""
        stats = strategy.get_stats()

        # Sharpe ratio (annualised)
        if len(self.results) > 1:
            returns = [r["profit"] for r in self.results]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (mean_ret / std_ret * np.sqrt(12)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = self.bankroll_history[0]
        max_dd = 0.0
        for b in self.bankroll_history:
            peak = max(peak, b)
            dd = (peak - b) / peak
            max_dd = max(max_dd, dd)

        total_profit = strategy.bankroll - self.initial_bankroll
        roi = total_profit / self.initial_bankroll * 100

        summary = {
            **stats,
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_roi_pct": round(roi, 2),
            "final_bankroll": round(strategy.bankroll, 2),
            "periods_tested": len(self.results),
        }

        # Flag potential leakage
        if roi > 15:
            logger.warning(
                "ROI %.1f%% > 15%% — investigate for potential data leakage!", roi
            )
            summary["leakage_flag"] = True

        logger.info("Backtest summary: %s", summary)
        return summary


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_path = "data/processed/features.parquet"
    if Path(data_path).exists():
        bt = WalkForwardBacktest()
        results = bt.run(data_path)
        print(f"\nBacktest results: {results}")
    else:
        print("No feature data found. Run processor first.")
