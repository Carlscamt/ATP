# ATP Betting Model

Production-ready ATP tennis betting model built with XGBoost GPU, Polars, and MLflow.

## Architecture

```
ATP/
├── config/           # YAML configuration (model params, logging)
├── scraper/          # Data acquisition (3 scrapers)
│   ├── atp_scraper.py        # Historical matches (Infosys API + BS4)
│   ├── schedule_scraper.py   # Upcoming matches + player ID resolution
│   └── odds_scraper.py       # Bookmaker odds (The Odds API / Oddsportal)
├── data/             # Feature engineering (Polars, leakage-safe)
│   ├── processor.py          # SafeFeatureEngineer (Elo, EMA, Rolling, H2H)
│   ├── leakage_prevention.py # TimeSeriesFeatureTransformer
│   ├── splitter.py           # Temporal splits + walk-forward CV
│   └── quality_checks.py     # Schema & value validation
├── model/            # Training & explainability
│   ├── xgboost_trainer.py    # GPU training + MLflow
│   ├── tuning.py             # Optuna hyperparameter search
│   └── explainability.py     # SHAP + feature drift detection
├── betting/          # Strategy & backtesting
│   ├── strategy.py           # Fractional Kelly + EV
│   └── backtest.py           # Walk-forward backtester
├── deploy/           # Production inference
│   └── inference.py          # Daily prediction pipeline
├── database/         # PostgreSQL bet tracking
│   ├── schema.sql
│   └── bet_logger.py
├── api/              # FastAPI server
│   └── server.py
├── monitor/          # Observability
│   ├── dashboard.py          # Streamlit dashboard
│   └── alerts.py             # Health checks
├── scripts/          # Automation
│   └── daily_predictions.py  # 6 AM scheduler
├── tests/            # 20+ tests (pytest)
│   └── test_pipeline.py
└── utils/            # Shared utilities
    └── error_handler.py      # Retry, rate limiting
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scrape historical data (2020-2024)
python -m scraper.atp_scraper 2020 2024

# 3. Engineer features
python -m data.processor

# 4. Split data (chronological)
python -m data.splitter

# 5. Quality check
python -m data.quality_checks data/processed/features.parquet

# 6. Train model (GPU)
python -m model.xgboost_trainer

# 7. Run daily predictions
python scripts/daily_predictions.py

# 8. Start API server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# 9. Launch monitoring dashboard
streamlit run monitor/dashboard.py
```

## Key Design Decisions

### Leakage Prevention
- All rolling/EMA features use `shift(1)` — current match excluded
- Elo computed in chronological order (eager loop)
- `TimeSeriesFeatureTransformer` fits ONLY on training data
- Temporal splits with walk-forward CV — no shuffling

### Technology Stack
| Component | Technology | Reason |
|-----------|-----------|--------|
| Data Processing | **Polars** | 5-10x faster than Pandas, lazy evaluation |
| Model | **XGBoost (GPU)** | `tree_method='hist'`, `device='cuda'` |
| Experiment Tracking | **MLflow** | Model versioning, param logging |
| Hyperparameter Tuning | **Optuna** | Bayesian + XGBoost pruning |
| Explainability | **SHAP** | TreeExplainer + drift detection |
| API | **FastAPI** | Async, Pydantic validation |
| Dashboard | **Streamlit** | Real-time KPI monitoring |
| Database | **PostgreSQL** | Bet tracking + Parquet fallback |

### Betting Strategy
- **Fractional Kelly (25%)** with 5% bankroll cap per bet
- EV-positive only — no negative expected value bets
- Odds filter: 1.5 to 10.0 decimal odds
- High-EV (>20%) flagged for manual review
- Backtest ROI >15% flags potential data leakage

## Tests

```bash
pytest tests/ -v
```

6 test categories: data quality, leakage prevention, betting math, feature engineering, API endpoints, E2E pipeline.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.2+ (for XGBoost GPU training)
- PostgreSQL (optional — Parquet fallback available)
- The Odds API key (optional — free tier: 500 req/month)
