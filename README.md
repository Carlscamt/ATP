# ATP Tennis Betting Model 🎾

A production-grade machine learning pipeline for predicting ATP tennis match outcomes and identifying value bets.

## Features

- **Advanced Scraping**: Bypasses Cloudflare bot detection using `curl_cffi` with Chrome TLS impersonation. Extracts detailed match statistics (aces, break points, serve %) via the hidden Hawkeye JSON API.
- **Machine Learning**: XGBoost model with GPU acceleration, hyperparameter tuning via Optuna, and SHAP explainability.
- **Feature Engineering**: Rolling averages, ELO ratings (surface-specific), fatigue metrics, and head-to-head stats.
- **Betting Strategy**: Fractional Kelly Criterion sizing with backtesting engine to validate profitability.
- **Operations**: FastAPI inference server, PostgreSQL bet logging, and Streamlit monitoring dashboard.
- **CI/CD**: GitHub Actions workflow for automated testing.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Carlscamt/ATP.git
   cd ATP
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires Python 3.10+ and CUDA toolkit for GPU training (optional).*

3. **Configuration**:
   - Check `config/model_config.yaml` for model parameters.
   - Set up environment variables (optional) in `.env` for database URLs or API keys.

## Data Pipeline

### 1. Scrape Data
Scrape historical match data (results + stats) for a range of years:
```bash
# Scrape data from 2020 to 2024
python -m scraper.atp_scraper 2020 2024
```
*Output: `data/raw/matches.parquet`*

### 2. Feature Engineering
Process raw data into ML-ready features (rolling stats, ELO, etc.):
```bash
python -m data.processor
```
*Output: `data/processed/features.parquet`*

### 3. Train Model
Train the XGBoost model with Optuna hyperparameter optimization:
```bash
python -m model.xgboost_trainer --tune
```
*Output: Saves model to `data/models/xgb_model.json` and metrics to MLflow.*

## Inference & Betting

### Run API Server
Start the FastAPI prediction server:
```bash
python -m api.server
```
*Docs available at `http://localhost:8000/docs`*

### Daily Predictions
Generate predictions for upcoming matches:
```bash
python -m scripts.daily_predictions
```

### Dashboard
Monitor model performance and recent bets:
```bash
streamlit run monitor/dashboard.py
```

## Project Structure

```
├── api/                # FastAPI application
├── betting/            # Betting strategies & backtesting
├── config/             # Configuration files (model_config.yaml)
├── data/               # Data storage (raw, processed, models)
├── database/           # Database schema & logging logic
├── deploy/             # Deployment scripts
├── model/              # XGBoost training & tuning logic
├── monitor/            # Streamlit dashboard & alerts
├── scraper/            # data collection (atp_scraper.py)
├── scripts/            # Utility scripts (daily predictions, testing)
├── tests/              # Pytest suite
└── utils/              # Helper functions
```

## Strategy Details

The model uses a **Fractional Kelly Criterion** (default 0.25) to size bets based on the predicted edge (Expected Value). 
- **Min Odds**: 1.50
- **Max Odds**: 10.00
- **Min Edge**: 2%
- **Model Calibration**: Probability calibration is applied to ensure model confidence matches real-world win rates.

## License

MIT License. See `LICENSE` for details.
