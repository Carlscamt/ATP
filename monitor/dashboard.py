"""
Streamlit Monitoring Dashboard

Real-time tracking of betting performance, model metrics,
bankroll evolution, and feature importance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import streamlit as st

logger = logging.getLogger("monitor")

st.set_page_config(page_title="ATP Betting Monitor", layout="wide")
st.title("🎾 ATP Betting Model Dashboard")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # Refresh every 5 minutes
def load_predictions() -> pl.DataFrame:
    """Load all daily prediction files."""
    pred_dir = Path("data/predictions")
    if not pred_dir.exists():
        return pl.DataFrame()

    parquet_files = list(pred_dir.glob("*.parquet"))
    if not parquet_files:
        return pl.DataFrame()

    dfs = [pl.read_parquet(f) for f in parquet_files]
    return pl.concat(dfs, how="diagonal")


@st.cache_data(ttl=300)
def load_bet_log() -> pl.DataFrame:
    """Load bet history."""
    path = Path("data/predictions/bet_log.parquet")
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
time_range = st.sidebar.selectbox(
    "Time Range", ["Last 7 days", "Last 30 days", "Last 90 days", "All Time"]
)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
st.header("📊 Performance KPIs")

predictions = load_predictions()
bet_log = load_bet_log()

if len(predictions) > 0:
    col1, col2, col3, col4 = st.columns(4)

    total_bets = len(predictions)
    avg_ev = predictions["expected_value"].mean() if "expected_value" in predictions.columns else 0

    col1.metric("Total Predictions", total_bets)
    col2.metric("Avg Expected Value", f"{avg_ev:.2%}" if avg_ev else "N/A")

    if "stake" in predictions.columns:
        total_stake = predictions["stake"].sum()
        col3.metric("Total Staked", f"${total_stake:,.2f}")

    if len(bet_log) > 0 and "profit" in bet_log.columns:
        total_profit = bet_log["profit"].sum()
        col4.metric("Total P&L", f"${total_profit:,.2f}",
                     delta=f"{total_profit:+,.2f}")
else:
    st.info("No prediction data available yet. Run the daily pipeline first.")

# ---------------------------------------------------------------------------
# Bankroll Chart
# ---------------------------------------------------------------------------
st.header("💰 Bankroll Evolution")

if len(bet_log) > 0 and "bankroll_after" in bet_log.columns:
    bankroll_data = bet_log.select(["created_at", "bankroll_before"]).to_pandas()
    st.line_chart(bankroll_data.set_index("created_at")["bankroll_before"])
else:
    st.info("No bankroll history available.")

# ---------------------------------------------------------------------------
# Recent Predictions
# ---------------------------------------------------------------------------
st.header("🔮 Recent Predictions")

if len(predictions) > 0:
    display_cols = [c for c in [
        "date", "player_name", "opponent", "tournament",
        "model_prob", "decimal_odds", "expected_value", "stake"
    ] if c in predictions.columns]

    if display_cols:
        recent = predictions.select(display_cols).tail(20)
        st.dataframe(recent.to_pandas(), use_container_width=True)

# ---------------------------------------------------------------------------
# Model Metrics
# ---------------------------------------------------------------------------
st.header("🧪 Model Metrics")

model_path = Path("model/xgboost_model.json")
if model_path.exists():
    st.success("✅ Model loaded")

    shap_bar = Path("model/shap/shap_importance_bar.png")
    shap_bee = Path("model/shap/shap_beeswarm.png")

    if shap_bar.exists():
        st.image(str(shap_bar), caption="Feature Importance (SHAP)")
    if shap_bee.exists():
        st.image(str(shap_bee), caption="SHAP Beeswarm")
else:
    st.warning("⚠️ No trained model found")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("ATP Betting Model v1.0 | XGBoost GPU + Polars + MLflow")
