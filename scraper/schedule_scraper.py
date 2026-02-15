"""
ATP Schedule Scraper — Upcoming Matches

Scrapes upcoming match schedules from atptour.com for inference.
Resolves player names to historical IDs for feature engineering.

Uses Playwright for JavaScript rendering to bypass bot detection.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import requests
from bs4 import BeautifulSoup

from utils.error_handler import fetch_with_retry
from utils.playwright_client import PlaywrightClient

logger = logging.getLogger("scraper")

BASE_URL = "https://www.atptour.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.atptour.com/",
}


class ScheduleScraper:
    """Scrape upcoming ATP matches for the next N days."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.player_mapping: pl.DataFrame | None = None
        self.rate_seconds: float = self.config.get("rate_limit_seconds", 2.0)
        self._playwright_client: PlaywrightClient | None = None
        self._last_request: float = 0.0

    def _get_client(self) -> PlaywrightClient:
        """Get or create Playwright client."""
        if self._playwright_client is None:
            self._playwright_client = PlaywrightClient(
                rate_limit_seconds=self.rate_seconds,
                headless=True,
                timeout=30000
            )
        return self._playwright_client

    def _close_client(self):
        """Close Playwright client."""
        if self._playwright_client:
            self._playwright_client.close()
            self._playwright_client = None

    def _throttle(self):
        """Apply rate limiting."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.rate_seconds:
            time.sleep(self.rate_seconds - elapsed)
        self._last_request = time.monotonic()

    def _fetch_url(self, url: str) -> requests.Response:
        """Fetch URL using Playwright with fallback."""
        self._throttle()
        
        try:
            client = self._get_client()
            return client.get(url)
        except Exception as e:
            logger.warning(f"Playwright failed for {url}: {e}")
            logger.info("Falling back to requests library")
            return fetch_with_retry(url, headers=HEADERS)

    def _load_player_mapping(self):
        """Load player ID → name mapping from historical database."""
        path = self.config.get(
            "player_mapping", "data/processed/player_mapping.parquet"
        )
        if Path(path).exists():
            self.player_mapping = pl.read_parquet(path)
            logger.info("Loaded %d players from mapping", len(self.player_mapping))
        else:
            logger.warning("Player mapping not found at %s", path)
            self.player_mapping = pl.DataFrame(
                schema={"player_id": pl.Utf8, "name": pl.Utf8}
            )

    def scrape_upcoming_matches(self, days_ahead: int = 7) -> pl.DataFrame:
        """
        Scrape upcoming matches from ATP current scores page.

        Returns DataFrame with:
            match_id, date, tournament, round, surface, tier,
            player1_name, player1_id, player2_name, player2_id,
            scraped_at
        """
        if self.player_mapping is None:
            self._load_player_mapping()

        url = f"{BASE_URL}/en/scores/current"
        logger.info("Scraping upcoming matches: %s", url)

        try:
            resp = self._fetch_url(url)
        except Exception:
            logger.error("Failed to fetch upcoming schedule")
            return pl.DataFrame()

        soup = BeautifulSoup(resp.text, "lxml")
        upcoming: list[dict[str, Any]] = []

        # Parse tournament sections on the scores page
        tournament_sections = soup.select(
            "div.tournament-section, div.scores-section"
        )

        for section in tournament_sections:
            tournament_info = self._parse_tournament_info(section)
            match_elements = section.select(
                "div.match-group div.match, tr.match-row"
            )

            for match_el in match_elements:
                try:
                    match = self._parse_upcoming_match(match_el, tournament_info)
                    if match:
                        upcoming.append(match)
                except Exception as e:
                    logger.warning("Skipping upcoming match: %s", e)

        if not upcoming:
            logger.info("No upcoming matches found on ATP schedule page")
            return pl.DataFrame()

        df = pl.DataFrame(upcoming)

        # Filter to requested time window
        cutoff = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        df = df.filter(pl.col("date") <= cutoff)

        # Save
        out_path = self.config.get(
            "upcoming_matches", "data/upcoming/matches.parquet"
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        logger.info("Saved %d upcoming matches → %s", len(df), out_path)

        return df

    def _parse_tournament_info(self, section: BeautifulSoup) -> dict[str, str]:
        """Extract tournament metadata from a section header."""
        name_tag = section.select_one("h3, .tournament-name, .event-name")
        name = name_tag.get_text(strip=True) if name_tag else "Unknown"

        surface_tag = section.select_one(".surface, .event-surface")
        surface = surface_tag.get_text(strip=True) if surface_tag else ""

        tier_tag = section.select_one(".category, .event-category")
        tier = tier_tag.get_text(strip=True) if tier_tag else ""

        return {"tournament": name, "surface": surface, "tier": tier}

    def _parse_upcoming_match(
        self, match_el: BeautifulSoup, tournament_info: dict
    ) -> dict[str, Any] | None:
        """Parse a single upcoming match element."""

        # Player names
        player_tags = match_el.select(
            "a.player-name, span.player-name, td.player-name a"
        )
        if len(player_tags) < 2:
            return None

        p1_name = player_tags[0].get_text(strip=True)
        p2_name = player_tags[1].get_text(strip=True)

        # Resolve player IDs
        p1_id = self.resolve_player_id(p1_name)
        p2_id = self.resolve_player_id(p2_name)

        # Round
        round_tag = match_el.select_one(".round, .match-round")
        round_name = round_tag.get_text(strip=True) if round_tag else ""

        # Scheduled date/time
        date_tag = match_el.select_one(".date, .match-date, time")
        date_str = ""
        if date_tag:
            date_str = date_tag.get("datetime", "") or date_tag.get_text(strip=True)

        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        return {
            "match_id": f"upcoming_{p1_id}_{p2_id}_{date_str}",
            "date": date_str[:10],  # YYYY-MM-DD
            "tournament": tournament_info.get("tournament", ""),
            "round": round_name,
            "surface": tournament_info.get("surface", ""),
            "tier": tournament_info.get("tier", ""),
            "player1_name": p1_name,
            "player1_id": p1_id,
            "player2_name": p2_name,
            "player2_id": p2_id,
            "scraped_at": datetime.now().isoformat(),
        }

    def resolve_player_id(self, player_name: str) -> str:
        """
        Map player name to ID from historical database.
        CRITICAL: Must match IDs used in training data.

        Returns empty string if player is unknown (flag for manual review).
        """
        if self.player_mapping is None or len(self.player_mapping) == 0:
            return ""

        # Exact match first
        exact = self.player_mapping.filter(
            pl.col("name").str.to_lowercase() == player_name.lower()
        )
        if len(exact) > 0:
            return exact["player_id"][0]

        # Fuzzy: last name match
        last_name = player_name.split()[-1].lower() if player_name else ""
        if last_name:
            fuzzy = self.player_mapping.filter(
                pl.col("name").str.to_lowercase().str.contains(last_name)
            )
            if len(fuzzy) == 1:
                return fuzzy["player_id"][0]

        logger.warning("Unknown player: '%s' — will use default Elo", player_name)
        return ""


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    config_path = Path("config/model_config.yaml")
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            full_cfg = yaml.safe_load(f)
        cfg = full_cfg.get("paths", {})

    scraper = ScheduleScraper(config=cfg)
    df = scraper.scrape_upcoming_matches(days_ahead=7)
    print(f"\nFound {len(df)} upcoming matches")
    if len(df) > 0:
        print(df.head(10))
