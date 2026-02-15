"""
FastAPI Inference Server

Endpoints:
  POST /predict  — Get prediction for a match
  GET  /health   — Health check
  GET  /stats    — Betting performance
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Global state (loaded on startup)
# ---------------------------------------------------------------------------
model: xgb.Booster | None = None
feature_cols: list[str] = []
threshold: float = 0.5

EXCLUDE_COLS = {
    "match_id", "date", "label", "winner", "player1_id", "player2_id",
    "player1_name", "player2_name", "player1_seed", "player2_seed",
    "score", "tournament_id", "tournament_name", "tournament_slug",
    "surface", "tier", "round", "location", "start_date", "year",
    "winner_flag", "scraped_at",
}


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and feature columns on startup."""
    global model, feature_cols, threshold

    model_path = "model/xgboost_model.json"
    if Path(model_path).exists():
        model = xgb.Booster()
        model.load_model(model_path)
        logger.info("Model loaded from %s", model_path)

    # Load feature columns
    feat_path = "data/processed/features.parquet"
    if Path(feat_path).exists():
        df = pl.read_parquet(feat_path, n_rows=1)
        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
        ]

    logger.info("API ready: %d features, threshold=%.4f", len(feature_cols), threshold)
    yield


app = FastAPI(title="ATP Betting Model", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class MatchInput(BaseModel):
    match_id: str
    player1_name: str
    player2_name: str
    player1_odds: float = 0.0
    player2_odds: float = 0.0
    features: dict[str, float] = {}


class PredictionOutput(BaseModel):
    match_id: str
    player1_name: str
    player2_name: str
    player1_win_prob: float
    player2_win_prob: float
    recommended_bet: str | None = None
    expected_value: float = 0.0
    kelly_stake_pct: float = 0.0


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    threshold: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(match: MatchInput):
    """Generate prediction for a match."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build feature vector
    feature_values = []
    for col in feature_cols:
        feature_values.append(match.features.get(col, 0.0))

    X = np.array([feature_values], dtype=np.float32)
    dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
    prob = float(model.predict(dmatrix)[0])

    # EV calculation
    from betting.strategy import calculate_ev, fractional_kelly

    recommended = None
    ev = 0.0
    kelly = 0.0

    if match.player1_odds > 0:
        ev_p1 = calculate_ev(prob, match.player1_odds)
        ev_p2 = calculate_ev(1 - prob, match.player2_odds) if match.player2_odds > 0 else 0

        if ev_p1 > ev_p2 and ev_p1 > 0:
            recommended = match.player1_name
            ev = ev_p1
            kelly = fractional_kelly(prob, match.player1_odds)
        elif ev_p2 > 0:
            recommended = match.player2_name
            ev = ev_p2
            kelly = fractional_kelly(1 - prob, match.player2_odds)

    return PredictionOutput(
        match_id=match.match_id,
        player1_name=match.player1_name,
        player2_name=match.player2_name,
        player1_win_prob=round(prob, 4),
        player2_win_prob=round(1 - prob, 4),
        recommended_bet=recommended,
        expected_value=round(ev, 4),
        kelly_stake_pct=round(kelly, 4),
    )


@app.get("/health", response_model=HealthOutput)
def health():
    """Health check endpoint."""
    return HealthOutput(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        n_features=len(feature_cols),
        threshold=threshold,
    )


@app.get("/stats")
def stats():
    """Get recent betting performance."""
    from database.bet_logger import BetLogger

    bet_logger = BetLogger()
    perf = bet_logger.get_performance()
    if len(perf) > 0:
        return perf.to_dicts()
    return {"message": "No betting history available"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
