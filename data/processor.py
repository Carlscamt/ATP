"""
Leakage-Safe Feature Engineering — Polars

SafeFeatureEngineer: All features use shift(1) to exclude the current match.
Elo calculation is purely chronological via eager to_dicts() loop.

Features produced:
  - Surface-specific + blended Elo ratings
  - EMA (α=0.3) of serve %, return %, break points
  - Rolling-window (5, 10, 22) averages
  - Head-to-head records (career + surface)
  - Fatigue: days since last match, matches in 7/14/30 days
  - Tournament context: round (ordinal), tier (ordinal), surface (one-hot)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl
import yaml

logger = logging.getLogger("data")

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    cfg_path = Path("config/model_config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAT_COLS = [
    "p1_aces", "p1_double_faults", "p1_first_serve_pct",
    "p1_first_serve_won_pct", "p1_second_serve_won_pct",
    "p1_break_points_saved", "p1_break_points_converted",
    "p1_service_points_won", "p1_return_points_won",
    "p1_total_points_won", "p1_winners", "p1_unforced_errors",
]

TIER_ORDER = {
    "Grand Slam": 4,
    "ATP Finals": 4,
    "Masters 1000": 3,
    "ATP 500": 2,
    "ATP 250": 1,
}

ROUND_ORDER = {
    "Final": 7, "Semifinals": 6, "Semi-Finals": 6,
    "Quarterfinals": 5, "Quarter-Finals": 5,
    "Round of 16": 4, "4th Round": 4,
    "Round of 32": 3, "3rd Round": 3,
    "Round of 64": 2, "2nd Round": 2,
    "Round of 128": 1, "1st Round": 1,
    "Qualifying": 0,
}


class SafeFeatureEngineer:
    """
    Polars-based, leakage-free feature engineering.

    All rolling / EMA features use shift(1) — the current match's result
    is never included in any feature that describes a player's form.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or _load_config()
        feat_cfg = self.config.get("features", {})
        self.alpha: float = feat_cfg.get("ema_alpha", 0.3)
        self.windows: list[int] = feat_cfg.get("rolling_windows", [5, 10, 22])
        elo_cfg = self.config.get("elo", {})
        self.initial_elo: float = elo_cfg.get("initial_rating", 1500)
        self.blend_ratio: float = elo_cfg.get("blend_ratio", 0.5)
        self.k_factors: dict[str, int] = elo_cfg.get("k_factors", {
            "grand_slam": 32, "masters": 24, "atp_500": 20, "atp_250": 16,
        })

    # =========================================================
    # PUBLIC API
    # =========================================================
    def create_features(self, df_path: str) -> pl.DataFrame:
        """
        Full feature engineering pipeline.

        1. Normalise raw data into per-player rows
        2. Compute Elo (chronological, eager)
        3. Compute EMA / rolling / H2H / fatigue (lazy where possible)
        4. Encode categoricals
        5. Return feature-ready DataFrame
        """
        raw = pl.read_parquet(df_path).sort("date")
        logger.info("Loaded %d raw matches from %s", len(raw), df_path)

        # --- Normalise to one row per player per match ---
        player_df = self._normalise_to_player_rows(raw)

        # --- Elo (must be eager — stateful computation) ---
        player_df = self._compute_elo(player_df)

        # --- EMA, rolling, fatigue (Polars expressions) ---
        player_df = self._compute_ema(player_df)
        player_df = self._compute_rolling(player_df)
        player_df = self._compute_fatigue(player_df)

        # --- Pivot back to match-level (1 row per match) ---
        match_df = self._pivot_to_match_rows(player_df, raw)

        # --- H2H features ---
        match_df = self._compute_h2h(match_df)

        # --- Encode categoricals ---
        match_df = self._encode(match_df)

        logger.info("Feature engineering complete: %d rows, %d cols",
                     len(match_df), len(match_df.columns))
        return match_df

    def engineer_features_for_upcoming(
        self, upcoming_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Produce features for upcoming (unplayed) matches.
        Uses only latest historical stats — no current-match data exists.
        """
        # Load latest Elo
        elo_path = self.config.get("paths", {}).get(
            "elo_history", "data/elo/ratings_history.parquet"
        )
        if not Path(elo_path).exists():
            logger.warning("No Elo history found; using default %d", self.initial_elo)
            latest_elo = pl.DataFrame(schema={
                "player_id": pl.Utf8, "elo": pl.Float64
            })
        else:
            elo_all = pl.read_parquet(elo_path).sort("date")
            latest_elo = (
                elo_all
                .group_by("player_id")
                .agg(pl.col("elo").last())
            )

        # Join
        features = (
            upcoming_df
            .join(latest_elo, left_on="player1_id", right_on="player_id", how="left")
            .rename({"elo": "p1_elo"})
            .join(latest_elo, left_on="player2_id", right_on="player_id", how="left")
            .rename({"elo": "p2_elo"})
        )

        # Fill unknown players with default
        features = features.with_columns([
            pl.col("p1_elo").fill_null(self.initial_elo),
            pl.col("p2_elo").fill_null(self.initial_elo),
        ])

        features = features.with_columns([
            (pl.col("p1_elo") - pl.col("p2_elo")).alias("elo_diff"),
        ])

        features = self._encode(features)
        return features

    # =========================================================
    # INTERNAL — normalise
    # =========================================================
    def _normalise_to_player_rows(self, raw: pl.DataFrame) -> pl.DataFrame:
        """
        Convert each match row into two player-perspective rows.
        This makes per-player rolling/EMA computations natural.
        """
        # Ensure numeric stat columns exist (fill missing with null)
        for col in STAT_COLS:
            if col not in raw.columns:
                raw = raw.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        p1_cols = {c: c.replace("p1_", "") for c in STAT_COLS}
        p2_cols = {c: c.replace("p1_", "") for c in [
            c.replace("p1_", "p2_") for c in STAT_COLS
        ]}

        common_cols = ["match_id", "date", "surface", "tier", "round",
                       "tournament_id", "year"]
        existing_common = [c for c in common_cols if c in raw.columns]

        p1 = (
            raw.select(
                existing_common +
                [pl.col("player1_id").alias("player_id"),
                 pl.col("player1_name").alias("player_name"),
                 pl.col("player2_id").alias("opponent_id"),
                 pl.col("winner").alias("winner_flag")] +
                [pl.col(c).cast(pl.Float64).alias(p1_cols[c]) for c in STAT_COLS if c in raw.columns]
            )
            .with_columns(
                (pl.col("winner_flag") == "player1").cast(pl.Int8).alias("won")
            )
        )

        p2_stat_cols = [c.replace("p1_", "p2_") for c in STAT_COLS]
        p2 = (
            raw.select(
                existing_common +
                [pl.col("player2_id").alias("player_id"),
                 pl.col("player2_name").alias("player_name"),
                 pl.col("player1_id").alias("opponent_id"),
                 pl.col("winner").alias("winner_flag")] +
                [pl.col(c).cast(pl.Float64).alias(c.replace("p2_", ""))
                 for c in p2_stat_cols if c in raw.columns]
            )
            .with_columns(
                (pl.col("winner_flag") == "player2").cast(pl.Int8).alias("won")
            )
        )

        combined = pl.concat([p1, p2], how="diagonal").sort(["player_id", "date"])
        return combined

    # =========================================================
    # INTERNAL — Elo
    # =========================================================
    def _compute_elo(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Chronological Elo calculation.
        Must be eager (stateful) — cannot use lazy API.
        """
        logger.info("Computing Elo ratings...")

        # Work from match-level to avoid double-counting
        matches = (
            df.select(["match_id", "player_id", "opponent_id", "won",
                        "date", "surface", "tier"])
            .unique(subset=["match_id", "player_id"])
            .sort("date")
        )

        elo_overall: dict[str, float] = {}
        elo_surface: dict[str, dict[str, float]] = {}
        elo_records: list[dict[str, Any]] = []

        for row in matches.to_dicts():
            pid = row["player_id"]
            oid = row["opponent_id"]
            surface = row.get("surface", "Hard")
            tier = row.get("tier", "ATP 250")

            # Current ratings (BEFORE this match)
            e_overall = elo_overall.get(pid, self.initial_elo)
            o_overall = elo_overall.get(oid, self.initial_elo)

            if surface not in elo_surface:
                elo_surface[surface] = {}
            e_surf = elo_surface[surface].get(pid, self.initial_elo)
            o_surf = elo_surface[surface].get(oid, self.initial_elo)

            # Blended
            e_blend = self.blend_ratio * e_surf + (1 - self.blend_ratio) * e_overall
            o_blend = self.blend_ratio * o_surf + (1 - self.blend_ratio) * o_overall

            elo_records.append({
                "match_id": row["match_id"],
                "player_id": pid,
                "elo": e_blend,
                "elo_overall": e_overall,
                "elo_surface": e_surf,
            })

            # Update
            expected = 1 / (1 + 10 ** ((o_blend - e_blend) / 400))
            actual = float(row["won"])
            k = self._get_k(tier)

            elo_overall[pid] = e_overall + k * (actual - expected)
            elo_overall[oid] = o_overall + k * ((1 - actual) - (1 - expected))
            elo_surface[surface][pid] = e_surf + k * (actual - expected)
            elo_surface[surface][oid] = o_surf + k * ((1 - actual) - (1 - expected))

        # Save full Elo history
        elo_df = pl.DataFrame(elo_records)
        elo_history = (
            matches.select(["match_id", "player_id", "date"])
            .join(elo_df, on=["match_id", "player_id"], how="left")
        )
        elo_out = self.config.get("paths", {}).get(
            "elo_history", "data/elo/ratings_history.parquet"
        )
        Path(elo_out).parent.mkdir(parents=True, exist_ok=True)
        elo_history.write_parquet(elo_out)
        logger.info("Elo history saved → %s", elo_out)

        # Join back
        result = df.join(
            elo_df, on=["match_id", "player_id"], how="left"
        )
        return result

    def _get_k(self, tier: str) -> int:
        tier_lower = tier.lower().replace(" ", "_")
        for key, val in self.k_factors.items():
            if key in tier_lower:
                return val
        return 16

    # =========================================================
    # INTERNAL — EMA
    # =========================================================
    def _compute_ema(self, df: pl.DataFrame) -> pl.DataFrame:
        """EMA with shift(1) — current match excluded."""
        ema_cols = ["aces", "double_faults", "first_serve_pct",
                    "service_points_won", "return_points_won"]
        existing = [c for c in ema_cols if c in df.columns]

        exprs = []
        for col in existing:
            exprs.append(
                pl.col(col)
                .shift(1)
                .ewm_mean(alpha=self.alpha, ignore_nulls=True)
                .over("player_id")
                .alias(f"{col}_ema")
            )

        if exprs:
            df = df.with_columns(exprs)
        return df

    # =========================================================
    # INTERNAL — Rolling averages
    # =========================================================
    def _compute_rolling(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rolling windows (5, 10, 22) with shift(1)."""
        roll_cols = ["aces", "first_serve_pct",
                     "service_points_won", "return_points_won", "won"]
        existing = [c for c in roll_cols if c in df.columns]

        for window in self.windows:
            exprs = []
            for col in existing:
                exprs.append(
                    pl.col(col)
                    .shift(1)
                    .rolling_mean(window_size=window, min_samples=1)
                    .over("player_id")
                    .alias(f"{col}_roll_{window}")
                )
            if exprs:
                df = df.with_columns(exprs)
        return df

    # =========================================================
    # INTERNAL — Fatigue
    # =========================================================
    def _compute_fatigue(self, df: pl.DataFrame) -> pl.DataFrame:
        """Days since last match & match counts in 7/14/30 days."""
        if "date" not in df.columns:
            return df

        df = df.with_columns(
            pl.col("date").str.to_date().alias("date_dt")
        )

        # Days since last match
        df = df.with_columns(
            (pl.col("date_dt") - pl.col("date_dt").shift(1).over("player_id"))
            .dt.total_days()
            .alias("days_since_last")
        )

        # Match count in rolling windows
        # Using a simple shift-and-count approach
        for days in [7, 14, 30]:
            df = df.with_columns(
                pl.col("match_id")
                .shift(1)
                .rolling_count(window_size=days)
                .over("player_id")
                .alias(f"matches_last_{days}d")
            )

        df = df.drop("date_dt")
        return df

    # =========================================================
    # INTERNAL — Pivot back to match rows
    # =========================================================
    def _pivot_to_match_rows(
        self, player_df: pl.DataFrame, raw: pl.DataFrame
    ) -> pl.DataFrame:
        """Combine two player-perspective rows back into one match row."""
        feature_cols = [c for c in player_df.columns
                        if c.endswith(("_ema", "_roll_5", "_roll_10", "_roll_22"))
                        or c in ("elo", "elo_overall", "elo_surface",
                                 "days_since_last",
                                 "matches_last_7d", "matches_last_14d",
                                 "matches_last_30d")]

        # Player 1 features
        p1_features = (
            player_df
            .join(
                raw.select(["match_id", "player1_id"]).unique(),
                left_on=["match_id", "player_id"],
                right_on=["match_id", "player1_id"],
                how="inner",
            )
            .select(
                ["match_id"] +
                [pl.col(c).alias(f"p1_{c}") for c in feature_cols if c in player_df.columns]
            )
        )

        # Player 2 features
        p2_features = (
            player_df
            .join(
                raw.select(["match_id", "player2_id"]).unique(),
                left_on=["match_id", "player_id"],
                right_on=["match_id", "player2_id"],
                how="inner",
            )
            .select(
                ["match_id"] +
                [pl.col(c).alias(f"p2_{c}") for c in feature_cols if c in player_df.columns]
            )
        )

        # Join features onto raw match data
        result = (
            raw.join(p1_features, on="match_id", how="left")
            .join(p2_features, on="match_id", how="left")
        )

        # Elo difference
        if "p1_elo" in result.columns and "p2_elo" in result.columns:
            result = result.with_columns(
                (pl.col("p1_elo") - pl.col("p2_elo")).alias("elo_diff")
            )

        # Label
        if "winner" in result.columns:
            result = result.with_columns(
                (pl.col("winner") == "player1").cast(pl.Int8).alias("label")
            )

        return result

    # =========================================================
    # INTERNAL — H2H
    # =========================================================
    def _compute_h2h(self, df: pl.DataFrame) -> pl.DataFrame:
        """Head-to-head win counts (career + surface-specific)."""
        if "player1_id" not in df.columns or "player2_id" not in df.columns:
            return df

        rows = df.sort("date").to_dicts()
        h2h_career: dict[tuple[str, str], int] = {}
        h2h_surface: dict[tuple[str, str, str], int] = {}
        h2h_records = []

        for row in rows:
            p1 = row.get("player1_id", "")
            p2 = row.get("player2_id", "")
            surface = row.get("surface", "")
            winner = row.get("winner", "")

            # Record BEFORE match
            p1_career_wins = h2h_career.get((p1, p2), 0)
            p2_career_wins = h2h_career.get((p2, p1), 0)
            p1_surf_wins = h2h_surface.get((p1, p2, surface), 0)
            p2_surf_wins = h2h_surface.get((p2, p1, surface), 0)

            h2h_records.append({
                "match_id": row["match_id"],
                "p1_h2h_wins": p1_career_wins,
                "p2_h2h_wins": p2_career_wins,
                "p1_h2h_surface_wins": p1_surf_wins,
                "p2_h2h_surface_wins": p2_surf_wins,
                "h2h_total": p1_career_wins + p2_career_wins,
            })

            # Update AFTER match
            if winner == "player1":
                h2h_career[(p1, p2)] = p1_career_wins + 1
                h2h_surface[(p1, p2, surface)] = p1_surf_wins + 1
            elif winner == "player2":
                h2h_career[(p2, p1)] = p2_career_wins + 1
                h2h_surface[(p2, p1, surface)] = p2_surf_wins + 1

        h2h_df = pl.DataFrame(h2h_records)
        return df.join(h2h_df, on="match_id", how="left")

    # =========================================================
    # INTERNAL — Encoding
    # =========================================================
    def _encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ordinal + one-hot encoding for categoricals."""
        # Surface one-hot
        if "surface" in df.columns:
            surfaces = ["Hard", "Clay", "Grass"]
            for s in surfaces:
                df = df.with_columns(
                    (pl.col("surface") == s).cast(pl.Int8).alias(f"surface_{s.lower()}")
                )

        # Tier ordinal
        if "tier" in df.columns:
            df = df.with_columns(
                pl.col("tier")
                .replace_strict(TIER_ORDER, default=0)
                .cast(pl.Int8)
                .alias("tier_ordinal")
            )

        # Round ordinal
        if "round" in df.columns:
            df = df.with_columns(
                pl.col("round")
                .replace_strict(ROUND_ORDER, default=0)
                .cast(pl.Int8)
                .alias("round_ordinal")
            )

        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = _load_config()
    paths = cfg.get("paths", {})
    engineer = SafeFeatureEngineer(config=cfg)

    raw_path = paths.get("raw_matches", "data/raw/matches.parquet")
    if Path(raw_path).exists():
        features = engineer.create_features(raw_path)
        out = paths.get("processed_features", "data/processed/features.parquet")
        features.write_parquet(out)
        print(f"Features written: {len(features)} rows → {out}")
        print(features.head(5))
    else:
        print(f"No raw data found at {raw_path}. Run scraper first.")
