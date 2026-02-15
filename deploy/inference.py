"""
Production Inference Pipeline

Daily pipeline: scrape upcoming → scrape odds → engineer features →
predict → calculate EV → recommend bets → save to Parquet & DB.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import polars as pl
import xgboost as xgb
import yaml

from betting.strategy import BettingStrategy, calculate_ev, fractional_kelly
from data.processor import SafeFeatureEngineer
from scraper.odds_scraper import OddsScraper
from scraper.schedule_scraper import ScheduleScraper

logger = logging.getLogger("deploy")


def _load_config() -> dict:
    cfg_path = Path("config/model_config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


EXCLUDE_COLS = {
    "match_id", "date", "label", "winner", "player1_id", "player2_id",
    "player1_name", "player2_name", "player1_seed", "player2_seed",
    "score", "tournament_id", "tournament_name", "tournament_slug",
    "surface", "tier", "round", "location", "start_date", "year",
    "winner_flag", "scraped_at", "player1_odds", "player2_odds",
    "event_id", "bookmaker", "commence_time",
}


class ProductionInference:
    """Daily prediction pipeline for upcoming matches."""

    def __init__(self, config: dict | None = None):
        self.config = config or _load_config()
        self.model: xgb.Booster | None = None
        self.feature_cols: list[str] = []
        self.threshold: float = 0.5
        self.feature_engineer = SafeFeatureEngineer(config=self.config)

        betting_cfg = self.config.get("betting", {})
        self.strategy = BettingStrategy(
            bankroll=betting_cfg.get("initial_bankroll", 10000),
            kelly_fraction=betting_cfg.get("kelly_fraction", 0.25),
            max_stake_pct=betting_cfg.get("max_stake_pct", 0.05),
            min_odds=betting_cfg.get("min_odds", 1.5),
            max_odds=betting_cfg.get("max_odds", 10.0),
        )

    def load_model(self, model_path: str = "model/xgboost_model.json"):
        """Load trained model."""
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        # Load feature columns from training data
        train_path = self.config.get("paths", {}).get(
            "processed_features", "data/processed/features.parquet"
        )
        if Path(train_path).exists():
            df = pl.read_parquet(train_path, n_rows=1)
            self.feature_cols = [
                c for c in df.columns
                if c not in EXCLUDE_COLS
                and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
            ]

        logger.info("Model loaded with %d features", len(self.feature_cols))

    def run_daily_predictions(self, days_ahead: int = 2) -> list[dict[str, Any]]:
        """
        Full daily prediction pipeline.

        1. Scrape upcoming matches (next 24-48 hours)
        2. Scrape current odds
        3. Engineer features for upcoming matches
        4. Generate predictions
        5. Calculate EV and recommend bets
        6. Save recommendations
        """
        if self.model is None:
            self.load_model()

        # 1. Get upcoming matches
        logger.info("Step 1: Scraping upcoming matches...")
        paths_cfg = self.config.get("paths", {})
        schedule_scraper = ScheduleScraper(config=paths_cfg)
        upcoming = schedule_scraper.scrape_upcoming_matches(days_ahead=days_ahead)

        if len(upcoming) == 0:
            logger.info("No upcoming matches found")
            return []

        logger.info("Found %d upcoming matches", len(upcoming))

        # 2. Get current odds
        logger.info("Step 2: Scraping current odds...")
        scraping_cfg = self.config.get("scraping", {})
        odds_scraper = OddsScraper(
            config=paths_cfg,
            api_key=scraping_cfg.get("odds_api_key", ""),
        )
        odds = odds_scraper.scrape_odds()

        # 3. Merge upcoming with odds
        if len(odds) > 0:
            matches_with_odds = upcoming.join(
                odds.select([
                    "player1_name", "player2_name",
                    "player1_odds", "player2_odds"
                ]),
                on=["player1_name", "player2_name"],
                how="left",
            )
        else:
            logger.warning("No odds data — predictions without stake sizing")
            matches_with_odds = upcoming.with_columns([
                pl.lit(0.0).alias("player1_odds"),
                pl.lit(0.0).alias("player2_odds"),
            ])

        # 4. Engineer features
        logger.info("Step 3: Engineering features for upcoming matches...")
        features = self.feature_engineer.engineer_features_for_upcoming(
            matches_with_odds
        )

        # 5. Generate predictions
        logger.info("Step 4: Generating predictions...")
        available_features = [c for c in self.feature_cols if c in features.columns]
        if not available_features:
            logger.error("No matching feature columns found")
            return []

        X = features.select(available_features).to_numpy().astype(np.float32)
        dmatrix = xgb.DMatrix(X, feature_names=available_features)
        predictions = self.model.predict(dmatrix)

        # 6. Calculate EV and recommend bets
        logger.info("Step 5: Evaluating betting opportunities...")
        bets: list[dict[str, Any]] = []
        match_rows = matches_with_odds.to_dicts()

        for idx, row in enumerate(match_rows):
            prob_p1_win = float(predictions[idx])
            p1_odds = float(row.get("player1_odds", 0))
            p2_odds = float(row.get("player2_odds", 0))

            # Evaluate player 1
            if p1_odds > 0:
                bet = self.strategy.evaluate_bet(
                    row.get("match_id", ""),
                    row.get("player1_name", ""),
                    prob_p1_win,
                    p1_odds,
                )
                if bet:
                    bet["tournament"] = row.get("tournament", "")
                    bet["opponent"] = row.get("player2_name", "")
                    bet["date"] = row.get("date", "")
                    bets.append(bet)

            # Evaluate player 2
            if p2_odds > 0:
                bet = self.strategy.evaluate_bet(
                    row.get("match_id", ""),
                    row.get("player2_name", ""),
                    1 - prob_p1_win,
                    p2_odds,
                )
                if bet:
                    bet["tournament"] = row.get("tournament", "")
                    bet["opponent"] = row.get("player1_name", "")
                    bet["date"] = row.get("date", "")
                    bets.append(bet)

        # 7. Save recommendations
        if bets:
            self._save_recommendations(bets)
            logger.info("Recommended %d bets", len(bets))
        else:
            logger.info("No positive EV bets found today")

        return bets

    def _save_recommendations(self, bets: list[dict[str, Any]]):
        """Save daily recommendations to Parquet."""
        pred_dir = Path(
            self.config.get("paths", {}).get("predictions_dir", "data/predictions")
        )
        pred_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        path = pred_dir / f"{date_str}.parquet"
        pl.DataFrame(bets).write_parquet(path)
        logger.info("Recommendations saved → %s", path)


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    inference = ProductionInference()
    bets = inference.run_daily_predictions()
    if bets:
        print(f"\n{len(bets)} betting opportunities found:")
        for b in bets:
            print(f"  {b['player_name']} @ {b['decimal_odds']} "
                  f"(EV={b['expected_value']:.2%}, Stake={b['stake']:.2f})")
    else:
        print("No positive EV bets today.")
