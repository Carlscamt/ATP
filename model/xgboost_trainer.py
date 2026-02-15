"""
XGBoost GPU Trainer with MLflow

GPU-accelerated training using tree_method='hist' + device='cuda'.
Polars-native Parquet input → numpy → DMatrix conversion.
Auto-logs params, metrics, and model artifacts to MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import polars as pl
import xgboost as xgb
import yaml

logger = logging.getLogger("model")

# Columns to exclude from features
EXCLUDE_COLS = {
    "match_id", "date", "label", "winner", "player1_id", "player2_id",
    "player1_name", "player2_name", "player1_seed", "player2_seed",
    "score", "tournament_id", "tournament_name", "tournament_slug",
    "surface", "tier", "round", "location", "start_date", "year",
    "winner_flag", "scraped_at",
}


def _load_config() -> dict:
    cfg_path = Path("config/model_config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


class XGBoostTrainer:
    """Train XGBoost model on GPU with MLflow tracking."""

    def __init__(self, config: dict | None = None):
        self.config = config or _load_config()
        xgb_cfg = self.config.get("xgboost", {})
        mlflow_cfg = self.config.get("mlflow", {})

        self.params: dict[str, Any] = {
            "tree_method": xgb_cfg.get("tree_method", "hist"),
            "device": xgb_cfg.get("device", "cuda"),
            "objective": xgb_cfg.get("objective", "binary:logistic"),
            "max_delta_step": xgb_cfg.get("max_delta_step", 5),
            "learning_rate": xgb_cfg.get("learning_rate", 0.01),
            "max_depth": xgb_cfg.get("max_depth", 6),
            "min_child_weight": xgb_cfg.get("min_child_weight", 1),
            "subsample": xgb_cfg.get("subsample", 0.8),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
            "eval_metric": xgb_cfg.get("eval_metric", ["aucpr", "logloss"]),
        }
        self.num_boost_round: int = xgb_cfg.get("num_boost_round", 1000)
        self.early_stopping: int = xgb_cfg.get("early_stopping_rounds", 50)

        self.experiment_name: str = mlflow_cfg.get("experiment_name", "atp_betting_model")
        self.tracking_uri: str = mlflow_cfg.get("tracking_uri", "mlruns")
        self.model_name: str = mlflow_cfg.get("model_name", "ATP_XGBoost")

        self.feature_cols: list[str] = []
        self.model: xgb.Booster | None = None

    # ---------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------
    def _load_data(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load Parquet → extract features → numpy arrays."""
        df = pl.read_parquet(path)
        return self._df_to_arrays(df)

    def _df_to_arrays(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Convert Polars DataFrame to numpy X, y arrays."""
        # Auto-detect feature columns
        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
        ]

        if not self.feature_cols:
            self.feature_cols = feature_cols
        else:
            # Use the same features as training
            feature_cols = [c for c in self.feature_cols if c in df.columns]

        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df["label"].to_numpy().astype(np.float32) if "label" in df.columns else np.zeros(len(df))

        return X, y

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    def train(
        self, train_path: str, val_path: str, *, log_to_mlflow: bool = True
    ) -> xgb.Booster:
        """
        Train XGBoost on GPU with optional MLflow logging.

        Auto-calculates scale_pos_weight from training label distribution.
        """
        X_train, y_train = self._load_data(train_path)
        X_val, y_val = self._load_data(val_path)

        # Calculate class imbalance weight
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            self.params["scale_pos_weight"] = n_neg / n_pos
            logger.info(
                "Class balance: %d positive, %d negative, scale_pos_weight=%.2f",
                n_pos, n_neg, self.params["scale_pos_weight"],
            )

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)

        logger.info(
            "Training XGBoost: %d train, %d val, %d features, device=%s",
            len(y_train), len(y_val), len(self.feature_cols), self.params["device"],
        )

        if log_to_mlflow:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

        context = mlflow.start_run() if log_to_mlflow else _nullcontext()

        with context as run:
            # Log params
            if log_to_mlflow:
                mlflow.log_params({
                    k: str(v) for k, v in self.params.items()
                })
                mlflow.log_param("num_features", len(self.feature_cols))
                mlflow.log_param("train_size", len(y_train))
                mlflow.log_param("val_size", len(y_val))

            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=self.early_stopping,
                verbose_eval=50,
            )

            # Log model & metrics
            if log_to_mlflow:
                mlflow.xgboost.log_model(self.model, "model")

                # Log best metrics
                best_iter = self.model.best_iteration
                mlflow.log_metric("best_iteration", best_iter)

                # Log feature importance
                importance = self.model.get_score(importance_type="gain")
                for feat, score in sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )[:20]:
                    mlflow.log_metric(f"importance_{feat}", score)

                logger.info(
                    "MLflow run: %s, best_iteration=%d",
                    run.info.run_id if run else "N/A", best_iter,
                )

        return self.model

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Predict probabilities for a DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X, _ = self._df_to_arrays(df)
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_cols)
        return self.model.predict(dmatrix)

    # ---------------------------------------------------------
    # Threshold tuning
    # ---------------------------------------------------------
    def tune_threshold(
        self, val_path: str, min_precision: float = 0.55
    ) -> float:
        """
        Find optimal decision threshold that maximizes recall
        while maintaining precision ≥ min_precision.
        """
        from sklearn.metrics import precision_recall_curve

        X_val, y_val = self._load_data(val_path)
        dval = xgb.DMatrix(X_val, feature_names=self.feature_cols)
        probs = self.model.predict(dval)

        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

        # Find best threshold: max recall where precision >= target
        best_threshold = 0.5
        best_recall = 0.0

        for prec, rec, thr in zip(precisions, recalls, thresholds):
            if prec >= min_precision and rec > best_recall:
                best_recall = rec
                best_threshold = thr

        logger.info(
            "Optimal threshold: %.4f (recall=%.4f at precision≥%.2f)",
            best_threshold, best_recall, min_precision,
        )
        return best_threshold

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------
    def save(self, path: str = "model/xgboost_model.json"):
        """Save model to JSON."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info("Model saved → %s", path)

    def load(self, path: str = "model/xgboost_model.json"):
        """Load model from JSON."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        logger.info("Model loaded ← %s", path)


class _nullcontext:
    """Minimal context manager for Python < 3.10 compatibility."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    trainer = XGBoostTrainer()
    train_path = "data/processed/train.parquet"
    val_path = "data/processed/val.parquet"

    if Path(train_path).exists() and Path(val_path).exists():
        model = trainer.train(train_path, val_path)
        threshold = trainer.tune_threshold(val_path)
        trainer.save()
        print(f"\nTraining complete. Optimal threshold: {threshold:.4f}")
    else:
        print("No train/val data found. Run processor + splitter first.")
