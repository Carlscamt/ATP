"""
Bookmaker Odds Scraper

Fetches current betting odds for upcoming ATP matches.
Supports The Odds API (recommended) and Oddsportal (free fallback).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import requests

from utils.error_handler import fetch_with_retry

logger = logging.getLogger("scraper")


class OddsScraper:
    """Scrape bookmaker odds for upcoming ATP matches."""

    def __init__(self, config: dict | None = None, api_key: str = ""):
        self.config = config or {}
        self.api_key = api_key or self.config.get("odds_api_key", "")

    # =================================================================
    # 1.  THE ODDS API (recommended — $50-200/mo)
    # =================================================================
    def scrape_odds_api(self) -> pl.DataFrame:
        """
        Fetch ATP odds from The Odds API.

        Docs: https://the-odds-api.com/liveapi/guides/v4/
        Free tier: 500 requests/month.
        """
        if not self.api_key:
            logger.error("No API key set for The Odds API")
            return pl.DataFrame()

        url = "https://api.the-odds-api.com/v4/sports/tennis_atp/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": "us,uk,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }

        try:
            resp = fetch_with_retry(url, params=params)
            data = resp.json()
        except Exception:
            logger.error("Failed to fetch odds from The Odds API")
            return pl.DataFrame()

        odds_records: list[dict[str, Any]] = []
        for event in data:
            record = self._parse_odds_api_event(event)
            if record:
                odds_records.append(record)

        if not odds_records:
            logger.info("No odds data returned from The Odds API")
            return pl.DataFrame()

        df = pl.DataFrame(odds_records)
        self._save(df)
        return df

    def _parse_odds_api_event(self, event: dict) -> dict[str, Any] | None:
        """Parse a single event from The Odds API response."""
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")

        if not home or not away:
            return None

        # Find best odds across bookmakers
        best_home_odds = 1.0
        best_away_odds = 1.0
        best_bookmaker = ""

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = {
                        o["name"]: o["price"] for o in market.get("outcomes", [])
                    }
                    home_odds = outcomes.get(home, 1.0)
                    away_odds = outcomes.get(away, 1.0)

                    if home_odds > best_home_odds:
                        best_home_odds = home_odds
                        best_bookmaker = bookmaker.get("title", "")
                    if away_odds > best_away_odds:
                        best_away_odds = away_odds

        return {
            "event_id": event.get("id", ""),
            "player1_name": home,
            "player2_name": away,
            "player1_odds": best_home_odds,
            "player2_odds": best_away_odds,
            "bookmaker": best_bookmaker,
            "commence_time": commence,
            "scraped_at": datetime.now().isoformat(),
        }

    # =================================================================
    # 2.  MANUAL / SCRAPING FALLBACK (Oddsportal — free)
    # =================================================================
    def scrape_oddsportal(self) -> pl.DataFrame:
        """
        Scrape odds from Oddsportal.com (requires Playwright for JS).
        This is a free alternative but less reliable.

        NOTE: Oddsportal heavily relies on JavaScript rendering.
        This method requires Playwright to be installed.
        """
        logger.info("Oddsportal scraping — requires Playwright")

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.error("Playwright not installed. Run: playwright install chromium")
            return pl.DataFrame()

        odds_records: list[dict[str, Any]] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                page.goto(
                    "https://www.oddsportal.com/tennis/",
                    wait_until="networkidle",
                    timeout=30000,
                )
                page.wait_for_timeout(3000)

                # Find match rows
                matches = page.query_selector_all(
                    "div.eventRow, tr.deactivate"
                )

                for match in matches:
                    try:
                        record = self._parse_oddsportal_match(match)
                        if record:
                            odds_records.append(record)
                    except Exception as e:
                        logger.warning("Skipping Oddsportal match: %s", e)

            except Exception as e:
                logger.error("Oddsportal scraping failed: %s", e)
            finally:
                browser.close()

        if not odds_records:
            return pl.DataFrame()

        df = pl.DataFrame(odds_records)
        self._save(df)
        return df

    def _parse_oddsportal_match(self, match_el) -> dict[str, Any] | None:
        """Parse a single match element from Oddsportal."""
        players = match_el.query_selector_all("a.participant-name, span.name")
        if len(players) < 2:
            return None

        p1_name = players[0].inner_text().strip()
        p2_name = players[1].inner_text().strip()

        odds_els = match_el.query_selector_all("span.odds-value, td.odds-val")
        p1_odds = float(odds_els[0].inner_text()) if len(odds_els) > 0 else 1.0
        p2_odds = float(odds_els[1].inner_text()) if len(odds_els) > 1 else 1.0

        return {
            "event_id": f"oddsportal_{p1_name}_{p2_name}",
            "player1_name": p1_name,
            "player2_name": p2_name,
            "player1_odds": p1_odds,
            "player2_odds": p2_odds,
            "bookmaker": "Oddsportal (best)",
            "commence_time": "",
            "scraped_at": datetime.now().isoformat(),
        }

    # =================================================================
    # 3.  UNIFIED INTERFACE
    # =================================================================
    def scrape_odds(self) -> pl.DataFrame:
        """
        Scrape odds using the best available method.
        Priority: The Odds API > Oddsportal.
        """
        if self.api_key:
            logger.info("Using The Odds API")
            df = self.scrape_odds_api()
            if len(df) > 0:
                return df

        logger.info("Falling back to Oddsportal scraping")
        return self.scrape_oddsportal()

    def _save(self, df: pl.DataFrame):
        """Save odds to Parquet."""
        out_path = self.config.get(
            "upcoming_odds", "data/upcoming/odds.parquet"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        logger.info("Saved %d odds records → %s", len(df), out_path)


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    config_path = Path("config/model_config.yaml")
    cfg = {}
    api_key = ""
    if config_path.exists():
        with open(config_path) as f:
            full_cfg = yaml.safe_load(f)
        cfg = full_cfg.get("paths", {})
        api_key = full_cfg.get("scraping", {}).get("odds_api_key", "")

    scraper = OddsScraper(config=cfg, api_key=api_key)
    df = scraper.scrape_odds()
    print(f"\nFetched {len(df)} odds records")
    if len(df) > 0:
        print(df.head(10))
