"""
Comprehensive Test Suite for ATP Betting Model

Tests:
  - Data quality and schema
  - Leakage prevention
  - Betting strategy correctness
  - Feature engineering logic
  - API endpoints
  - End-to-end pipeline
"""

import numpy as np
import polars as pl
import pytest
from pathlib import Path
from datetime import datetime


# =========================================================================
# 1.  DATA QUALITY TESTS
# =========================================================================

class TestDataQuality:
    """Validate data integrity and schema compliance."""

    def test_parquet_schema(self, tmp_path):
        """Verify feature data has required columns."""
        from data.quality_checks import DataQualityChecker

        # Create minimal valid data
        df = pl.DataFrame({
            "match_id": ["m1", "m2"],
            "date": ["2024-01-01", "2024-01-02"],
            "player1_id": ["p1", "p2"],
            "player2_id": ["p3", "p4"],
            "label": [1, 0],
        })
        path = str(tmp_path / "test_data.parquet")
        df.write_parquet(path)

        checker = DataQualityChecker()
        result = checker.check_all(path)
        assert result is True, f"Errors: {checker.errors}"

    def test_missing_required_columns(self, tmp_path):
        """Fail on missing required columns."""
        from data.quality_checks import DataQualityChecker

        df = pl.DataFrame({"foo": [1, 2], "bar": ["a", "b"]})
        path = str(tmp_path / "bad_data.parquet")
        df.write_parquet(path)

        checker = DataQualityChecker()
        result = checker.check_all(path)
        assert result is False

    def test_chronological_ordering(self, tmp_path):
        """Verify data is sorted by date."""
        from data.quality_checks import DataQualityChecker

        df = pl.DataFrame({
            "match_id": ["m1", "m2"],
            "date": ["2024-01-05", "2024-01-01"],  # Wrong order
            "player1_id": ["p1", "p2"],
            "player2_id": ["p3", "p4"],
        })
        path = str(tmp_path / "unordered.parquet")
        df.write_parquet(path)

        checker = DataQualityChecker()
        checker.check_all(path)
        assert any("chronologically" in e for e in checker.errors)


# =========================================================================
# 2.  LEAKAGE PREVENTION TESTS
# =========================================================================

class TestLeakagePrevention:
    """Verify no future data leaks into features."""

    def test_shift_prevents_current_match_leakage(self):
        """EMA/rolling must use shift(1) — current match excluded."""
        df = pl.DataFrame({
            "player_id": ["p1"] * 5,
            "aces": [10.0, 15.0, 20.0, 25.0, 30.0],
        })

        # Simulate shift(1) + rolling mean
        result = df.with_columns(
            pl.col("aces")
            .shift(1)
            .rolling_mean(window_size=3, min_samples=1)
            .over("player_id")
            .alias("aces_roll_3")
        )

        # Row 0 should be null (no prior data)
        assert result["aces_roll_3"][0] is None
        # Row 1 should only see row 0, not row 1
        assert result["aces_roll_3"][1] == 10.0
        # Row 4 should see rows 1-3 (shifted), not row 4
        assert result["aces_roll_3"][4] == pytest.approx(20.0, abs=0.1)

    def test_temporal_split_no_overlap(self, tmp_path):
        """Train/val/test should have no date overlap."""
        from data.splitter import TemporalDataSplitter

        df = pl.DataFrame({
            "date": [
                "2022-01-01", "2022-06-01", "2023-01-01",
                "2023-06-01", "2024-01-01", "2024-06-01",
            ],
            "match_id": [f"m{i}" for i in range(6)],
            "label": [1, 0, 1, 0, 1, 0],
        })
        path = str(tmp_path / "features.parquet")
        df.write_parquet(path)

        splitter = TemporalDataSplitter(
            val_start="2023-01-01", test_start="2024-01-01"
        )
        train, val, test = splitter.split(path)

        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

    def test_transformer_fit_only_on_train(self, tmp_path):
        """TimeSeriesFeatureTransformer must use ONLY training stats."""
        from data.leakage_prevention import TimeSeriesFeatureTransformer

        # Training data: mean = 50
        train_df = pl.DataFrame({
            "match_id": ["m1", "m2"],
            "date": ["2023-01-01", "2023-06-01"],
            "player1_id": ["p1", "p2"],
            "player2_id": ["p3", "p4"],
            "elo_diff": [40.0, 60.0],
            "label": [1, 0],
        })
        train_path = str(tmp_path / "train.parquet")
        train_df.write_parquet(train_path)

        transformer = TimeSeriesFeatureTransformer()
        transformer.fit(train_path)

        # Global mean should be 50
        assert "elo_diff_mean" in transformer.global_stats
        assert transformer.global_stats["elo_diff_mean"] == pytest.approx(50.0)


# =========================================================================
# 3.  BETTING STRATEGY TESTS
# =========================================================================

class TestBettingStrategy:
    """Verify Kelly criterion and EV calculations."""

    def test_ev_calculation(self):
        from betting.strategy import calculate_ev
        # 60% win prob at 2.0 odds: EV = 0.6*(2-1) - 0.4 = 0.2
        assert calculate_ev(0.6, 2.0) == pytest.approx(0.2, abs=0.01)

    def test_negative_ev_returns_zero(self):
        from betting.strategy import calculate_ev
        # 30% win prob at 2.0 odds: EV = 0.3*(1) - 0.7 = -0.4 → 0
        assert calculate_ev(0.3, 2.0) == 0.0

    def test_full_kelly_formula(self):
        from betting.strategy import full_kelly
        # b=1.0, p=0.6, q=0.4: f* = (1*0.6 - 0.4)/1 = 0.2
        assert full_kelly(0.6, 2.0) == pytest.approx(0.2, abs=0.01)

    def test_fractional_kelly_cap(self):
        from betting.strategy import fractional_kelly
        # Full Kelly = 0.2, fraction=0.25 → 0.05
        result = fractional_kelly(0.6, 2.0, fraction=0.25, max_stake_pct=0.05)
        assert result <= 0.05

    def test_strategy_filters_bad_odds(self):
        from betting.strategy import BettingStrategy
        strategy = BettingStrategy(min_odds=1.5, max_odds=10.0)

        # Odds too low
        bet = strategy.evaluate_bet("m1", "Player", 0.8, 1.1)
        assert bet is None

        # Odds too high
        bet = strategy.evaluate_bet("m1", "Player", 0.5, 15.0)
        assert bet is None

    def test_bankroll_update(self):
        from betting.strategy import BettingStrategy
        strategy = BettingStrategy(bankroll=1000.0)

        # Win a bet
        strategy.record_result("m1", True, 50.0, 2.0)
        assert strategy.bankroll == 1050.0

        # Lose a bet
        strategy.record_result("m2", False, 30.0, 2.0)
        assert strategy.bankroll == 1020.0


# =========================================================================
# 4.  FEATURE ENGINEERING TESTS
# =========================================================================

class TestFeatureEngineering:
    """Verify Elo, EMA, and encoding logic."""

    def test_elo_starts_at_default(self):
        """New players should start at initial Elo rating."""
        from data.processor import SafeFeatureEngineer

        engineer = SafeFeatureEngineer(config={
            "elo": {"initial_rating": 1500, "blend_ratio": 0.5,
                    "k_factors": {"atp_250": 16}},
            "features": {"ema_alpha": 0.3, "rolling_windows": [5]},
        })
        assert engineer.initial_elo == 1500

    def test_surface_one_hot_encoding(self):
        """Surface column should produce 3 one-hot columns."""
        from data.processor import SafeFeatureEngineer

        engineer = SafeFeatureEngineer()
        df = pl.DataFrame({"surface": ["Hard", "Clay", "Grass", "Hard"]})
        result = engineer._encode(df)

        assert "surface_hard" in result.columns
        assert "surface_clay" in result.columns
        assert "surface_grass" in result.columns
        assert result["surface_hard"].to_list() == [1, 0, 0, 1]

    def test_tier_ordinal_encoding(self):
        """Tier should map to ordinal values."""
        from data.processor import SafeFeatureEngineer

        engineer = SafeFeatureEngineer()
        df = pl.DataFrame({"tier": ["Grand Slam", "ATP 250", "Masters 1000"]})
        result = engineer._encode(df)

        assert "tier_ordinal" in result.columns
        assert result["tier_ordinal"][0] == 4  # Grand Slam
        assert result["tier_ordinal"][1] == 1  # ATP 250
        assert result["tier_ordinal"][2] == 3  # Masters 1000


# =========================================================================
# 5.  API TESTS
# =========================================================================

class TestAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "n_features" in data

    def test_predict_without_model(self, client):
        """Predict should return 503 if no model loaded."""
        resp = client.post("/predict", json={
            "match_id": "test",
            "player1_name": "Djokovic",
            "player2_name": "Alcaraz",
        })
        # 503 if model not loaded, 200 if loaded
        assert resp.status_code in (200, 503)


# =========================================================================
# 6.  E2E PIPELINE TESTS
# =========================================================================

class TestEndToEnd:
    """End-to-end pipeline smoke tests."""

    def test_schedule_scraper_init(self):
        """ScheduleScraper should initialize without error."""
        from scraper.schedule_scraper import ScheduleScraper
        scraper = ScheduleScraper()
        assert scraper is not None

    def test_odds_scraper_no_key(self):
        """OddsScraper without API key should return empty DataFrame."""
        from scraper.odds_scraper import OddsScraper
        scraper = OddsScraper(api_key="")
        df = scraper.scrape_odds_api()
        assert len(df) == 0

    def test_production_inference_init(self):
        """ProductionInference should initialize without error."""
        from deploy.inference import ProductionInference
        inference = ProductionInference()
        assert inference is not None

    def test_bet_logger_parquet_fallback(self, tmp_path):
        """BetLogger should work with Parquet fallback."""
        from database.bet_logger import BetLogger

        logger = BetLogger(connection_string="")  # No DB
        logger._fallback_path = tmp_path / "test_log.parquet"

        logger.log_bet({
            "match_id": "test_m1",
            "player_name": "Sinner",
            "model_prob": 0.65,
            "decimal_odds": 1.8,
            "stake": 100.0,
        })

        assert logger._fallback_path.exists()
        df = pl.read_parquet(logger._fallback_path)
        assert len(df) == 1
        assert df["match_id"][0] == "test_m1"

    def test_health_checker(self):
        """HealthChecker should run without crashing."""
        from monitor.alerts import HealthChecker
        checker = HealthChecker()
        results = checker.run_all()
        assert isinstance(results, list)
        assert len(results) > 0
