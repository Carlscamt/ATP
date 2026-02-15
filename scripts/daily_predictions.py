"""
Daily Predictions — Scheduled Job

Runs at 6 AM daily: scrape → predict → recommend → save.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scheduler")


def run_daily_job():
    """Execute the daily prediction pipeline."""
    try:
        from deploy.inference import ProductionInference

        logger.info("=" * 60)
        logger.info("DAILY PREDICTIONS — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
        logger.info("=" * 60)

        inference = ProductionInference()
        bets = inference.run_daily_predictions(days_ahead=2)

        if bets:
            logger.info("Found %d betting opportunities:", len(bets))
            for b in bets:
                logger.info(
                    "  %s @ %.2f (EV=%.2f%%, Stake=$%.2f)",
                    b["player_name"], b["decimal_odds"],
                    b["expected_value"] * 100, b["stake"],
                )
        else:
            logger.info("No positive EV bets found today")

    except Exception as e:
        logger.error("Daily prediction failed: %s", e, exc_info=True)


if __name__ == "__main__":
    logger.info("Starting daily prediction scheduler (6:00 AM)")

    # Schedule at 6 AM
    schedule.every().day.at("06:00").do(run_daily_job)

    # Also run immediately on first start
    run_daily_job()

    while True:
        schedule.run_pending()
        time.sleep(60)
