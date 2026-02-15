"""
Model Explainability — SHAP & Feature Drift Detection

Generates SHAP summary plots, logs to MLflow, and monitors
feature importance drift between training and production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import shap
import xgboost as xgb
import yaml

logger = logging.getLogger("model")

EXCLUDE_COLS = {
    "match_id", "date", "label", "winner", "player1_id", "player2_id",
    "player1_name", "player2_name", "player1_seed", "player2_seed",
    "score", "tournament_id", "tournament_name", "tournament_slug",
    "surface", "tier", "round", "location", "start_date", "year",
    "winner_flag", "scraped_at",
}


class ModelExplainer:
    """SHAP-based model explainability and drift detection."""

    def __init__(self, model: xgb.Booster, feature_cols: list[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.baseline_importance: dict[str, float] | None = None

    def compute_shap(
        self, df_path: str, *, max_samples: int = 1000
    ) -> np.ndarray:
        """
        Compute SHAP values for a dataset.
        Uses TreeExplainer (fast, exact for tree models).
        """
        df = pl.read_parquet(df_path)
        feature_df = df.select(self.feature_cols)

        # Sample if too large
        if len(feature_df) > max_samples:
            feature_df = feature_df.sample(max_samples, seed=42)

        X = feature_df.to_numpy().astype(np.float32)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        logger.info("SHAP values computed for %d samples", len(X))
        return shap_values

    def generate_summary(
        self, df_path: str, *, save_dir: str = "model/shap"
    ):
        """
        Generate and save SHAP summary plot.
        Logs artifacts to MLflow.
        """
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        df = pl.read_parquet(df_path)
        feature_df = df.select(self.feature_cols)

        if len(feature_df) > 1000:
            feature_df = feature_df.sample(1000, seed=42)

        X = feature_df.to_numpy().astype(np.float32)
        shap_values = self.compute_shap(df_path)

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Summary plot (bar)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_cols,
            plot_type="bar",
            show=False,
        )
        bar_path = f"{save_dir}/shap_importance_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_cols,
            show=False,
        )
        bee_path = f"{save_dir}/shap_beeswarm.png"
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Log to MLflow
        try:
            mlflow.log_artifact(bar_path)
            mlflow.log_artifact(bee_path)
        except Exception:
            logger.warning("MLflow artifact logging failed (no active run?)")

        logger.info("SHAP summaries saved → %s", save_dir)

    def set_importance_baseline(self):
        """
        Store current feature importances as the baseline
        for drift detection.
        """
        importance = self.model.get_score(importance_type="gain")
        total = sum(importance.values()) or 1.0
        self.baseline_importance = {
            k: v / total for k, v in importance.items()
        }
        logger.info("Baseline importance set (%d features)", len(self.baseline_importance))

    def detect_drift(
        self, new_model: xgb.Booster, *, threshold: float = 0.30
    ) -> dict[str, float]:
        """
        Compare feature importances between baseline and new model.
        Returns dict of features whose importance changed > threshold.

        If any feature drifts > 30%, flag for investigation.
        """
        if self.baseline_importance is None:
            logger.warning("No baseline set — call set_importance_baseline() first")
            return {}

        new_importance = new_model.get_score(importance_type="gain")
        total = sum(new_importance.values()) or 1.0
        new_norm = {k: v / total for k, v in new_importance.items()}

        drifted: dict[str, float] = {}
        all_features = set(self.baseline_importance) | set(new_norm)

        for feat in all_features:
            old = self.baseline_importance.get(feat, 0.0)
            new = new_norm.get(feat, 0.0)

            if old > 0:
                change = abs(new - old) / old
            elif new > 0:
                change = 1.0  # New feature entirely
            else:
                change = 0.0

            if change > threshold:
                drifted[feat] = change
                logger.warning(
                    "Feature drift: %s (%.1f%% change)", feat, change * 100
                )

        if drifted:
            logger.warning(
                "%d features drifted > %.0f%%: %s",
                len(drifted), threshold * 100, list(drifted.keys()),
            )
        else:
            logger.info("No significant feature drift detected")

        return drifted


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_path = "model/xgboost_model.json"
    val_path = "data/processed/val.parquet"

    if Path(model_path).exists() and Path(val_path).exists():
        model = xgb.Booster()
        model.load_model(model_path)

        # Detect feature columns
        df = pl.read_parquet(val_path)
        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
        ]

        explainer = ModelExplainer(model, feature_cols)
        explainer.generate_summary(val_path)
        explainer.set_importance_baseline()
        print("SHAP analysis complete.")
    else:
        print("Model or validation data not found.")
