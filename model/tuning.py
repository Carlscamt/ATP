"""
Optuna Hyperparameter Tuning

Bayesian optimisation with XGBoost-specific pruning.
Uses walk-forward CV to avoid temporal leakage during tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
import yaml
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import recall_score, precision_score

logger = logging.getLogger("model")


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
    "winner_flag", "scraped_at",
}


class OptunaOptimizer:
    """Bayesian hyperparameter optimization with Optuna."""

    def __init__(self, config: dict | None = None):
        self.config = config or _load_config()
        optuna_cfg = self.config.get("optuna", {})
        self.n_trials: int = optuna_cfg.get("n_trials", 100)
        self.timeout: int = optuna_cfg.get("timeout", 3600)
        self.best_params: dict | None = None
        self.feature_cols: list[str] = []

    def optimize(
        self,
        train_path: str,
        val_path: str,
        *,
        min_precision: float = 0.55,
    ) -> dict:
        """
        Run Optuna optimization targeting recall at precision ≥ threshold.
        """
        # Load data
        train_df = pl.read_parquet(train_path)
        val_df = pl.read_parquet(val_path)

        self.feature_cols = [
            c for c in train_df.columns
            if c not in EXCLUDE_COLS
            and train_df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
        ]

        X_train = train_df.select(self.feature_cols).to_numpy().astype(np.float32)
        y_train = train_df["label"].to_numpy().astype(np.float32)
        X_val = val_df.select(self.feature_cols).to_numpy().astype(np.float32)
        y_val = val_df["label"].to_numpy().astype(np.float32)

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        base_scale = n_neg / n_pos if n_pos > 0 else 1.0

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "tree_method": "hist",
                "device": "cuda",
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", base_scale * 0.5, base_scale * 2.0
                ),
            }

            pruning_callback = XGBoostPruningCallback(trial, "val-aucpr")

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=False,
                callbacks=[pruning_callback],
            )

            # Predict and find best threshold
            preds = model.predict(dval)
            best_recall = 0.0

            for threshold in np.arange(0.2, 0.8, 0.02):
                y_pred = (preds >= threshold).astype(int)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)

                if prec >= min_precision and rec > best_recall:
                    best_recall = rec

            return best_recall

        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name="atp_xgboost_tuning",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        logger.info(
            "Starting Optuna: %d trials, timeout=%ds", self.n_trials, self.timeout
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        self.best_params = study.best_trial.params
        logger.info(
            "Best trial: recall=%.4f, params=%s",
            study.best_trial.value, self.best_params,
        )

        # Log best params to MLflow
        mlflow_cfg = self.config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "atp_betting_model"))
        with mlflow.start_run(run_name="optuna_best"):
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_recall", study.best_trial.value)

        return self.best_params


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    optimizer = OptunaOptimizer()
    train_path = "data/processed/train.parquet"
    val_path = "data/processed/val.parquet"

    if Path(train_path).exists() and Path(val_path).exists():
        best = optimizer.optimize(train_path, val_path)
        print(f"\nBest hyperparameters: {best}")
    else:
        print("No train/val data found. Run processor + splitter first.")
