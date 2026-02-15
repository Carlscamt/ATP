#!/usr/bin/env python3
"""
5-Year ATP Data Scrape with Hawkeye Stats (2020-2024)
Run overnight on Linux: nohup python3 scrape_all.py &

Estimated time: ~3-4 hours
Output: data/raw/matches.parquet
"""
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape_log.txt", mode="w"),
    ],
)
logger = logging.getLogger("scrape_all")

from scraper.atp_scraper import ATPScraper

START_YEAR = 2020
END_YEAR = 2024

def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("  ATP 5-Year Scrape with Hawkeye Stats (%d-%d)", START_YEAR, END_YEAR)
    logger.info("  Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # Delete old data to start fresh
    parquet_path = Path("data/raw/matches.parquet")
    if parquet_path.exists():
        parquet_path.unlink()
        logger.info("Deleted old matches.parquet")

    scraper = ATPScraper({"rate_limit_seconds": 1.5})

    total_matches = 0
    total_with_stats = 0
    year_summaries = []

    for year in range(START_YEAR, END_YEAR + 1):
        year_start = time.time()
        logger.info("")
        logger.info("=" * 70)
        logger.info("  YEAR %d", year)
        logger.info("=" * 70)

        try:
            df = scraper.scrape_year(year, include_stats=True)
            year_matches = len(df)
            year_stats = 0

            if year_matches > 0 and "p1_aces" in df.columns:
                import polars as pl
                year_stats = len(df.filter(pl.col("p1_aces").is_not_null()))

            elapsed = time.time() - year_start
            total_matches += year_matches
            total_with_stats += year_stats

            summary = f"{year}: {year_matches} matches, {year_stats} with stats ({elapsed/60:.1f} min)"
            year_summaries.append(summary)
            logger.info(summary)

        except Exception as e:
            logger.error("FAILED on year %d: %s", year, e)
            year_summaries.append(f"{year}: FAILED - {e}")

    # Final summary
    total_time = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SCRAPE COMPLETE")
    logger.info("=" * 70)
    logger.info("  Total time: %.1f minutes (%.1f hours)", total_time / 60, total_time / 3600)
    logger.info("  Total matches: %d", total_matches)
    logger.info("  With Hawkeye stats: %d", total_with_stats)
    logger.info("")
    for s in year_summaries:
        logger.info("  %s", s)

    # Verify parquet
    if parquet_path.exists():
        import polars as pl
        df = pl.read_parquet(parquet_path)
        logger.info("")
        logger.info("  Parquet file: %s", parquet_path)
        logger.info("  Total rows: %d", len(df))
        logger.info("  Total columns: %d", len(df.columns))
        logger.info("  File size: %.1f MB", parquet_path.stat().st_size / 1024 / 1024)

if __name__ == "__main__":
    main()
