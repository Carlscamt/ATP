"""
Bet Logger — PostgreSQL Integration

Records bets and results to the database for tracking and reporting.
Falls back to Parquet logging if PostgreSQL is unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger("database")


class BetLogger:
    """Log bets and results to PostgreSQL or Parquet fallback."""

    def __init__(self, connection_string: str = ""):
        self.connection_string = connection_string
        self.conn = None
        self._fallback_path = Path("data/predictions/bet_log.parquet")

        if connection_string:
            try:
                import psycopg2
                self.conn = psycopg2.connect(connection_string)
                logger.info("Connected to PostgreSQL")
            except Exception as e:
                logger.warning("PostgreSQL unavailable: %s. Using Parquet fallback.", e)
                self.conn = None

    def log_bet(self, bet: dict[str, Any], model_version: str = ""):
        """Record a new bet (before match result)."""
        record = {
            "match_id": bet.get("match_id", ""),
            "date": bet.get("date", datetime.now().strftime("%Y-%m-%d")),
            "tournament": bet.get("tournament", ""),
            "player_name": bet.get("player_name", ""),
            "opponent_name": bet.get("opponent", ""),
            "model_prob": bet.get("model_prob", 0.0),
            "decimal_odds": bet.get("decimal_odds", 0.0),
            "implied_prob": bet.get("implied_prob", 0.0),
            "expected_value": bet.get("expected_value", 0.0),
            "kelly_fraction": bet.get("kelly_fraction", 0.0),
            "stake": bet.get("stake", 0.0),
            "bankroll_before": bet.get("bankroll", 0.0),
            "model_version": model_version,
            "flag": bet.get("flag", ""),
            "created_at": datetime.now().isoformat(),
        }

        if self.conn:
            self._insert_to_db(record)
        else:
            self._append_to_parquet(record)

    def settle_bet(self, match_id: str, player_name: str, won: bool, profit: float, bankroll_after: float):
        """Update bet with match result."""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """UPDATE bets SET won = %s, profit = %s,
                       bankroll_after = %s, settled_at = NOW()
                       WHERE match_id = %s AND player_name = %s""",
                    (won, profit, bankroll_after, match_id, player_name),
                )
                self.conn.commit()
                cur.close()
            except Exception as e:
                logger.error("Failed to settle bet: %s", e)
        else:
            logger.info(
                "Settled (Parquet): %s/%s won=%s profit=%.2f",
                match_id, player_name, won, profit,
            )

    def get_performance(self, months: int = 6) -> pl.DataFrame:
        """Get recent performance summary."""
        if self.conn:
            try:
                query = """
                    SELECT * FROM bet_performance
                    WHERE month >= CURRENT_DATE - INTERVAL '%s months'
                    ORDER BY month
                """
                return pl.read_database(query % months, self.conn)
            except Exception as e:
                logger.error("Failed to query performance: %s", e)

        # Parquet fallback
        if self._fallback_path.exists():
            return pl.read_parquet(self._fallback_path)
        return pl.DataFrame()

    def _insert_to_db(self, record: dict):
        """Insert bet record into PostgreSQL."""
        try:
            cur = self.conn.cursor()
            cols = list(record.keys())
            placeholders = ", ".join(["%s"] * len(cols))
            col_names = ", ".join(cols)
            cur.execute(
                f"INSERT INTO bets ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT (match_id, player_name) DO NOTHING",
                list(record.values()),
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            logger.error("Failed to insert bet: %s", e)

    def _append_to_parquet(self, record: dict):
        """Append to Parquet file (fallback)."""
        self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pl.DataFrame([record])

        if self._fallback_path.exists():
            existing = pl.read_parquet(self._fallback_path)
            combined = pl.concat([existing, new_df], how="diagonal")
            combined.write_parquet(self._fallback_path)
        else:
            new_df.write_parquet(self._fallback_path)

    def close(self):
        if self.conn:
            self.conn.close()
