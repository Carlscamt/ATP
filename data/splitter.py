"""
Temporal Data Splitter

Time-series aware train/val/test splitting.
Never shuffles — always respects chronological order.
Walk-forward CV with expanding training window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("data")


class TemporalDataSplitter:
    """
    Split data into train/val/test using temporal boundaries.
    Supports walk-forward cross-validation.
    """

    def __init__(
        self,
        val_start: str = "2023-01-01",
        test_start: str = "2024-01-01",
    ):
        self.val_start = val_start
        self.test_start = test_start

    # ---------------------------------------------------------
    # Fixed split
    # ---------------------------------------------------------
    def split(self, df_path: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split into train / validation / test by date.

        Train: < val_start
        Val:   >= val_start AND < test_start
        Test:  >= test_start
        """
        df = pl.scan_parquet(df_path).sort("date")

        train = df.filter(pl.col("date") < self.val_start).collect()
        val = df.filter(
            (pl.col("date") >= self.val_start) & (pl.col("date") < self.test_start)
        ).collect()
        test = df.filter(pl.col("date") >= self.test_start).collect()

        # Validate no temporal leakage
        if len(train) > 0 and len(val) > 0:
            assert train["date"].max() < val["date"].min(), "Train/Val overlap!"
        if len(val) > 0 and len(test) > 0:
            assert val["date"].max() < test["date"].min(), "Val/Test overlap!"

        # Save
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        train.write_parquet(out_dir / "train.parquet")
        val.write_parquet(out_dir / "val.parquet")
        test.write_parquet(out_dir / "test.parquet")

        logger.info(
            "Split: Train=%d (%s), Val=%d (%s), Test=%d (%s)",
            len(train), f"< {self.val_start}",
            len(val), f"{self.val_start} – {self.test_start}",
            len(test), f">= {self.test_start}",
        )
        return train, val, test

    # ---------------------------------------------------------
    # Walk-forward CV
    # ---------------------------------------------------------
    def time_series_cv(
        self,
        train_path: str,
        n_splits: int = 5,
    ):
        """
        Walk-forward (expanding window) cross-validation.

        Each fold expands the training window while keeping a fixed-size
        validation window that always comes chronologically after training.

        Yields (train_path, val_path) tuples.
        """
        df = pl.read_parquet(train_path).sort("date")
        n = len(df)

        if n < n_splits + 1:
            logger.warning("Not enough data for %d splits", n_splits)
            return

        folds_dir = Path("data/folds")
        folds_dir.mkdir(parents=True, exist_ok=True)

        # Calculate split points
        val_size = n // (n_splits + 1)

        for fold_idx in range(n_splits):
            train_end = val_size * (fold_idx + 1)
            val_end = min(train_end + val_size, n)

            train_fold = df[:train_end]
            val_fold = df[train_end:val_end]

            if len(train_fold) == 0 or len(val_fold) == 0:
                continue

            # Verify temporal ordering
            assert train_fold["date"].max() <= val_fold["date"].min(), \
                f"Fold {fold_idx}: temporal ordering violated!"

            t_path = folds_dir / f"fold_{fold_idx}_train.parquet"
            v_path = folds_dir / f"fold_{fold_idx}_val.parquet"
            train_fold.write_parquet(t_path)
            val_fold.write_parquet(v_path)

            logger.info(
                "Fold %d: Train=%d (%s→%s), Val=%d (%s→%s)",
                fold_idx,
                len(train_fold), train_fold["date"].min(), train_fold["date"].max(),
                len(val_fold), val_fold["date"].min(), val_fold["date"].max(),
            )

            yield str(t_path), str(v_path)


# CLI
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.parquet"
    splitter = TemporalDataSplitter()
    train, val, test = splitter.split(path)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
