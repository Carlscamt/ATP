"""
Data Quality Checks

Uses Polars expressions (and optionally Pandera) to validate
data integrity before model training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("data")


class DataQualityChecker:
    """Validate data quality before training."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check_all(self, df_path: str) -> bool:
        """Run all quality checks. Returns True if data is clean."""
        self.errors = []
        self.warnings = []

        df = pl.read_parquet(df_path)
        logger.info("Running quality checks on %s (%d rows)", df_path, len(df))

        self._check_schema(df)
        self._check_date_ordering(df)
        self._check_missing_values(df)
        self._check_value_ranges(df)
        self._check_duplicates(df)
        self._check_label_distribution(df)

        # Report
        for w in self.warnings:
            logger.warning("QUALITY WARNING: %s", w)
        for e in self.errors:
            logger.error("QUALITY ERROR: %s", e)

        passed = len(self.errors) == 0
        logger.info(
            "Quality check %s: %d errors, %d warnings",
            "PASSED" if passed else "FAILED",
            len(self.errors), len(self.warnings),
        )
        return passed

    def _check_schema(self, df: pl.DataFrame):
        """Verify required columns exist."""
        required = ["match_id", "date", "player1_id", "player2_id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")

    def _check_date_ordering(self, df: pl.DataFrame):
        """Verify data is sorted chronologically."""
        if "date" not in df.columns:
            return
        dates = df["date"].to_list()
        if dates != sorted(dates):
            self.errors.append("Data is not sorted chronologically by date")

    def _check_missing_values(self, df: pl.DataFrame):
        """Check missing value percentages."""
        null_counts = df.null_count()
        n = len(df)
        for col in null_counts.columns:
            null_pct = null_counts[col][0] / n * 100
            if null_pct > 50:
                self.errors.append(f"Column '{col}' has {null_pct:.1f}% missing values")
            elif null_pct > 20:
                self.warnings.append(f"Column '{col}' has {null_pct:.1f}% missing values")

    def _check_value_ranges(self, df: pl.DataFrame):
        """Verify stat values are within reasonable ranges."""
        range_checks = {
            "p1_first_serve_pct": (0, 100),
            "p2_first_serve_pct": (0, 100),
            "p1_aces": (0, 80),
            "p2_aces": (0, 80),
            "elo_diff": (-1000, 1000),
        }

        for col, (low, high) in range_checks.items():
            if col not in df.columns:
                continue
            col_data = df[col].drop_nulls()
            if len(col_data) == 0:
                continue
            min_val = col_data.min()
            max_val = col_data.max()
            if min_val < low or max_val > high:
                self.warnings.append(
                    f"Column '{col}' range [{min_val}, {max_val}] "
                    f"exceeds expected [{low}, {high}]"
                )

    def _check_duplicates(self, df: pl.DataFrame):
        """Check for duplicate match IDs."""
        if "match_id" not in df.columns:
            return
        n_unique = df["match_id"].n_unique()
        n_total = len(df)
        if n_unique < n_total:
            dup_count = n_total - n_unique
            self.warnings.append(f"{dup_count} duplicate match_id entries found")

    def _check_label_distribution(self, df: pl.DataFrame):
        """Verify label balance is reasonable."""
        if "label" not in df.columns:
            return
        counts = df["label"].value_counts()
        if len(counts) < 2:
            self.errors.append("Only one label class present in data")
            return

        total = len(df)
        for row in counts.to_dicts():
            pct = row["count"] / total * 100
            if pct < 10:
                self.warnings.append(
                    f"Label {row['label']} only {pct:.1f}% of data — severe imbalance"
                )


# CLI
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.parquet"
    checker = DataQualityChecker()
    passed = checker.check_all(path)
    print(f"\nQuality check: {'PASSED' if passed else 'FAILED'}")
