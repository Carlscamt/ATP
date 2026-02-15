"""
Leakage Prevention — TimeSeriesFeatureTransformer

Ensures that any statistics used for imputation or normalisation
are computed ONLY from training data and never from the test set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("data")


class TimeSeriesFeatureTransformer:
    """
    Fit on training data, transform train + test identically.
    Prevents target / future information leakage in normalisation.
    """

    def __init__(self):
        self.player_stats: pl.DataFrame | None = None
        self.global_stats: dict[str, float] = {}
        self.feature_cols: list[str] = []

    # ---------------------------------------------------------
    # FIT (train only)
    # ---------------------------------------------------------
    def fit(self, train_path: str) -> "TimeSeriesFeatureTransformer":
        """
        Compute per-player and global statistics from training data.
        Uses lazy evaluation for memory efficiency.
        """
        logger.info("Fitting transformer on %s", train_path)
        df = pl.scan_parquet(train_path)

        # Identify numeric feature columns
        schema = pl.read_parquet_schema(train_path)
        self.feature_cols = [
            col for col, dtype in schema.items()
            if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
            and col not in ("label", "year")
        ]

        # Per-player statistics (for imputation)
        if "player1_id" in schema:
            self.player_stats = (
                df.group_by("player1_id")
                .agg([
                    pl.col(c).mean().alias(f"{c}_mean")
                    for c in self.feature_cols
                    if c in schema
                ] + [
                    pl.col(c).std().alias(f"{c}_std")
                    for c in self.feature_cols
                    if c in schema
                ])
                .collect()
            )

        # Global statistics (fallback)
        global_df = (
            df.select([
                pl.col(c).mean().alias(f"{c}_mean")
                for c in self.feature_cols
                if c in schema
            ] + [
                pl.col(c).std().alias(f"{c}_std")
                for c in self.feature_cols
                if c in schema
            ] + [
                pl.col(c).quantile(0.99).alias(f"{c}_p99")
                for c in self.feature_cols
                if c in schema
            ])
            .collect()
        )

        for col in global_df.columns:
            self.global_stats[col] = global_df[col][0]

        logger.info(
            "Fitted on %d feature columns, %d player groups",
            len(self.feature_cols),
            len(self.player_stats) if self.player_stats is not None else 0,
        )
        return self

    # ---------------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------------
    def transform(self, df_path: str, *, is_training: bool = True) -> pl.DataFrame:
        """
        Apply transformations:
        1. Impute missing values (forward-fill then player/global mean)
        2. Cap outliers at 99th percentile (from training)
        3. Drop rows with >30% missing features
        """
        logger.info("Transforming %s (training=%s)", df_path, is_training)
        df = pl.read_parquet(df_path)

        # 1. Forward-fill within player (temporal fill)
        for col in self.feature_cols:
            if col in df.columns:
                if "player1_id" in df.columns:
                    df = df.with_columns(
                        pl.col(col)
                        .forward_fill()
                        .over("player1_id")
                        .alias(col)
                    )

        # 2. Fill remaining nulls with global mean from training
        for col in self.feature_cols:
            if col in df.columns:
                mean_key = f"{col}_mean"
                fill_val = self.global_stats.get(mean_key, 0.0)
                if fill_val is not None:
                    df = df.with_columns(
                        pl.col(col).fill_null(fill_val)
                    )

        # 3. Cap outliers at 99th percentile
        for col in self.feature_cols:
            if col in df.columns:
                p99_key = f"{col}_p99"
                cap = self.global_stats.get(p99_key)
                if cap is not None:
                    df = df.with_columns(
                        pl.when(pl.col(col) > cap)
                        .then(cap)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )

        # 4. Drop rows with too many missing features
        null_threshold = int(len(self.feature_cols) * 0.3)
        existing_features = [c for c in self.feature_cols if c in df.columns]
        if existing_features:
            null_count_expr = sum(
                pl.col(c).is_null().cast(pl.Int32) for c in existing_features
            )
            df = df.filter(null_count_expr <= null_threshold)

        logger.info("Transform complete: %d rows remaining", len(df))
        return df


# CLI
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python leakage_prevention.py <train.parquet> <test.parquet>")
        sys.exit(1)

    transformer = TimeSeriesFeatureTransformer()
    transformer.fit(sys.argv[1])

    train_out = transformer.transform(sys.argv[1], is_training=True)
    test_out = transformer.transform(sys.argv[2], is_training=False)

    print(f"Train: {len(train_out)} rows, Test: {len(test_out)} rows")
