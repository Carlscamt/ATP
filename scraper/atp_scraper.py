"""
ATP Tour Historical Data Scraper (v2 — curl_cffi + Hawkeye JSON API)

Bypasses Cloudflare/Akamai bot detection with TLS fingerprint impersonation.
Uses the real ATP site HTML structure (verified Feb 2026).

Data sources:
  - Calendar:  /en/scores/results-archive?year={year}
    Selectors: div.tournament-info → a.tournament__profile
               span.name, span.venue, span.Date
               img.events_banner (src → tier)

  - Results:   /en/scores/archive/{slug}/{tid}/{year}/results
    Selectors: div.match-group (round), div.match (match),
               div.stats-item > div.name > a (player),
               div.winner (winner flag), div.score-item > span (scores)
               div.match-cta a[href*=match-stats] (stats link)

  - Stats:     /-/Hawkeye/MatchStats/{year}/{tid}/{match_id}  (JSON API)
    Returns:   Aces, DoubleFaults, ServeRating, FirstServe%,
               FirstServePointsWon%, SecondServePointsWon%,
               BreakPointsSaved%, BreakPointsConverted%,
               ServiceGamesPlayed, ReturnRating, TotalPointsWon%
               (with raw Dividend/Divisor per stat, per set + match total)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from bs4 import BeautifulSoup, Tag

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    raise ImportError("curl_cffi required — pip install curl_cffi")

logger = logging.getLogger("scraper.atp")

BASE_URL = "https://www.atptour.com"

# Tournament tier mapping from badge image filename
TIER_MAP: dict[str, str] = {
    "grandslam": "Grand Slam",
    "1000": "Masters 1000",
    "500": "ATP 500",
    "250": "ATP 250",
    "finals": "ATP Finals",
    "nextgen": "Next Gen Finals",
    "unitedcup": "United Cup",
    "olympics": "Olympics",
    "laver": "Laver Cup",
}

SURFACE_MAP: dict[str, str] = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
}


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------
class ATPScraper:
    """Scrapes historical ATP Tour data and writes to Parquet."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.rate_seconds: float = self.config.get("rate_limit_seconds", 2.0)
        self.output_dir = Path(self.config.get("output_dir", "data/raw"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time: float = 0

    # -----------------------------------------------------------------------
    # HTTP helpers
    # -----------------------------------------------------------------------
    def _throttle(self):
        """Rate limiting — be polite to ATP servers."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_seconds:
            time.sleep(self.rate_seconds - elapsed)

    def _get(self, url: str, **kwargs) -> cffi_requests.Response:
        """GET with curl_cffi Chrome impersonation + throttle + retry."""
        self._throttle()
        for attempt in range(3):
            try:
                resp = cffi_requests.get(
                    url,
                    impersonate="chrome",
                    timeout=30,
                    **kwargs,
                )
                self._last_request_time = time.time()

                if resp.status_code == 403:
                    logger.warning("403 on %s — attempt %d/3", url, attempt + 1)
                    time.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code == 429:
                    logger.warning("429 rate limited — sleeping 30s")
                    time.sleep(30)
                    continue

                resp.raise_for_status()
                return resp

            except Exception as e:
                logger.warning("Request error on %s: %s (attempt %d)", url, e, attempt + 1)
                time.sleep(3 * (attempt + 1))

        raise RuntimeError(f"Failed to fetch {url} after 3 attempts")

    # -----------------------------------------------------------------------
    # Calendar scraping
    # -----------------------------------------------------------------------
    def scrape_calendar(self, year: int) -> list[dict[str, Any]]:
        """
        Scrape the list of tournaments for a given year.

        Returns list of dicts with keys:
            tournament_name, tournament_id, tournament_slug,
            location, tier, start_date, year
        """
        url = f"{BASE_URL}/en/scores/results-archive?year={year}"
        logger.info("Scraping calendar for %d: %s", year, url)
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        tournaments: list[dict[str, Any]] = []

        # Each tournament is a div.tournament-info
        for ti in soup.select("div.tournament-info"):
            parsed = self._parse_tournament_info(ti, year)
            if parsed:
                tournaments.append(parsed)

        logger.info("Found %d tournaments for %d", len(tournaments), year)
        return tournaments

    def _parse_tournament_info(
        self, el: Tag, year: int
    ) -> dict[str, Any] | None:
        """Parse a single div.tournament-info element."""
        # --- Tournament link, name, slug, ID ---
        profile_link = el.select_one("a.tournament__profile")
        if not profile_link:
            return None

        href = profile_link.get("href", "")
        # href: /en/tournaments/brisbane/339/overview
        parts = href.strip("/").split("/")
        slug = parts[2] if len(parts) > 2 else ""
        tid = parts[3] if len(parts) > 3 else ""

        name_el = el.select_one("span.name")
        name = name_el.get_text(strip=True) if name_el else slug

        # --- Location ---
        venue_el = el.select_one("span.venue")
        location = venue_el.get_text(strip=True).rstrip(" |") if venue_el else ""

        # --- Date ---
        date_el = el.select_one("span.Date")
        date_str = date_el.get_text(strip=True) if date_el else ""
        start_date = self._parse_date(date_str, year)

        # --- Tier (from badge image) ---
        badge_img = el.select_one("img.events_banner")
        tier = self._extract_tier(badge_img)

        # --- Archive link (results URL) ---
        parent_li = el.parent
        results_link = parent_li.select_one("a[href*='scores/archive']") if parent_li else None
        results_href = results_link.get("href", "") if results_link else ""

        return {
            "tournament_name": name,
            "tournament_id": tid,
            "tournament_slug": slug,
            "location": location,
            "tier": tier,
            "start_date": start_date,
            "year": year,
            "results_url": results_href,
        }

    def _extract_tier(self, badge_img: Tag | None) -> str:
        """Extract tier from badge image src."""
        if not badge_img:
            return "Unknown"
        src = badge_img.get("src", "").lower()
        for key, tier in TIER_MAP.items():
            if key in src:
                return tier
        return "ATP 250"  # Default

    # -----------------------------------------------------------------------
    # Tournament results scraping
    # -----------------------------------------------------------------------
    def scrape_tournament_results(
        self, slug: str, tid: str, year: int
    ) -> list[dict[str, Any]]:
        """
        Scrape all match results for a single tournament.

        Returns list of dicts with keys:
            match_id, round, player1_name, player1_id, player2_name,
            player2_id, score_p1, score_p2, winner, winner_id,
            stats_url, tournament_id, year
        """
        url = f"{BASE_URL}/en/scores/archive/{slug}/{tid}/{year}/results"
        logger.info("Scraping results: %s %d", slug, year)
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        matches: list[dict[str, Any]] = []

        # Match groups contain a round header + matches
        for group in soup.select("div.match-group"):
            # Extract round name from match-header
            header = group.select_one("div.match-header")
            round_name = ""
            if header:
                round_text = header.get_text(strip=True)
                # Remove duration suffix like " -03:44"
                round_name = re.sub(r"\s*-\d{2}:\d{2}$", "", round_text).strip()

            # Parse each match in this round
            for match_el in group.select("div.match"):
                parsed = self._parse_match(match_el, round_name, tid, year)
                if parsed:
                    matches.append(parsed)

        logger.info("  %s %d: %d matches", slug, year, len(matches))
        return matches

    def _parse_match(
        self, el: Tag, round_name: str, tid: str, year: int
    ) -> dict[str, Any] | None:
        """Parse a single div.match element."""
        stats_items = el.select("div.stats-item")
        if len(stats_items) < 2:
            return None

        players = []
        for si in stats_items[:2]:
            name_el = si.select_one("div.name a")
            if not name_el:
                continue

            name = name_el.get_text(strip=True)
            href = name_el.get("href", "")
            pid = self._extract_player_id(href)

            # Seed
            seed_el = si.select_one("div.name span")
            seed = seed_el.get_text(strip=True).strip("()") if seed_el else ""

            # Winner marker
            is_winner = si.select_one("div.winner") is not None

            # Scores
            scores = [s.get_text(strip=True) for s in si.select("div.score-item span")]

            players.append({
                "name": name,
                "id": pid,
                "seed": seed,
                "is_winner": is_winner,
                "scores": scores,
            })

        if len(players) < 2:
            return None

        # Determine winner
        winner_name = ""
        winner_id = ""
        for p in players:
            if p["is_winner"]:
                winner_name = p["name"]
                winner_id = p["id"]
                break

        # Match stats link
        stats_link = el.select_one("div.match-cta a[href*='match-stats']")
        stats_url = stats_link.get("href", "") if stats_link else ""

        # Extract match_id from stats URL
        raw_match_id = ""
        if stats_url:
            # /en/scores/match-stats/archive/2024/580/ms001
            parts = stats_url.strip("/").split("/")
            if parts:
                raw_match_id = parts[-1]

        # Build globally unique match_id (prefix with tid to avoid cross-tournament collisions)
        if raw_match_id:
            match_id = f"{tid}_{raw_match_id}"
        else:
            match_id = f"{tid}_{players[0]['id']}_{players[1]['id']}"

        # Build score string
        score_str = self._build_score_string(players[0]["scores"], players[1]["scores"])

        return {
            "match_id": match_id,
            "raw_match_id": raw_match_id,  # For Hawkeye API calls
            "round": round_name,
            "player1_name": players[0]["name"],
            "player1_id": players[0]["id"],
            "player1_seed": players[0]["seed"],
            "player2_name": players[1]["name"],
            "player2_id": players[1]["id"],
            "player2_seed": players[1]["seed"],
            "score": score_str,
            "winner": winner_name,
            "winner_id": winner_id,
            "stats_url": stats_url,
            "tournament_id": tid,
            "year": year,
        }

    def _build_score_string(self, scores1: list[str], scores2: list[str]) -> str:
        """Build a human-readable score from per-set scores."""
        sets = []
        for s1, s2 in zip(scores1, scores2):
            if s1 or s2:
                sets.append(f"{s1}-{s2}")
        return " ".join(sets)

    # -----------------------------------------------------------------------
    # Match stats via Hawkeye JSON API
    # -----------------------------------------------------------------------
    def scrape_match_stats(
        self, year: int, tid: str, raw_match_id: str
    ) -> dict[str, Any] | None:
        """
        Fetch detailed match stats via the Hawkeye JSON API.

        GET /-/Hawkeye/MatchStats/{year}/{tid}/{raw_match_id}
        Returns JSON with per-set and match-total stats for both players.

        Stats extracted (per player):
            aces, double_faults, serve_rating, first_serve_pct,
            first_serve_points_won_pct, second_serve_points_won_pct,
            break_points_saved_pct, break_points_faced,
            break_points_converted_pct, break_points_opportunities,
            service_games_played, return_rating,
            total_service_points_won_pct, total_return_points_won_pct,
            total_points_won_pct, match_duration
        """
        url = f"{BASE_URL}/-/Hawkeye/MatchStats/{year}/{tid}/{raw_match_id}"
        logger.debug("Fetching stats: %s", url)

        try:
            resp = self._get(url)
            data = resp.json()
        except Exception as e:
            logger.debug("Stats API failed for %s/%s: %s", tid, raw_match_id, e)
            return None

        return self._parse_hawkeye_json(data, year, tid, raw_match_id)

    def _parse_hawkeye_json(
        self, data: dict, year: int, tid: str, raw_match_id: str
    ) -> dict[str, Any] | None:
        """Parse the Hawkeye JSON response into flat stat columns."""
        match = data.get("Match", {})
        if not match:
            return None

        p1_data = match.get("PlayerTeam1", {})
        p2_data = match.get("PlayerTeam2", {})
        if not p1_data or not p2_data:
            return None

        stats: dict[str, Any] = {
            "match_id": f"{tid}_{raw_match_id}",
            "tournament_id": tid,
            "year": year,
        }

        # Extract whole-match stats (set 0 = match totals)
        for prefix, team in [("p1", p1_data), ("p2", p2_data)]:
            sets = team.get("Sets", [])
            if not sets:
                continue

            # Set 0 = match totals
            match_set = sets[0] if sets else {}
            set_stats = match_set.get("Stats", {})
            if not set_stats:
                continue

            # Match duration
            time_str = set_stats.get("Time", "")
            if time_str and prefix == "p1":  # Only need duration once
                stats["match_duration"] = time_str

            # Service stats
            svc = set_stats.get("ServiceStats", {})
            if svc:
                stats[f"{prefix}_aces"] = self._safe_num(svc, "Aces", "Number")
                stats[f"{prefix}_double_faults"] = self._safe_num(svc, "DoubleFaults", "Number")
                stats[f"{prefix}_serve_rating"] = self._safe_num(svc, "ServeRating", "Number")
                stats[f"{prefix}_first_serve_pct"] = self._safe_num(svc, "FirstServe", "Percent")
                stats[f"{prefix}_first_serve_won_pct"] = self._safe_num(svc, "FirstServePointsWon", "Percent")
                stats[f"{prefix}_second_serve_won_pct"] = self._safe_num(svc, "SecondServePointsWon", "Percent")
                stats[f"{prefix}_break_points_saved_pct"] = self._safe_num(svc, "BreakPointsSaved", "Percent")
                stats[f"{prefix}_service_games"] = self._safe_num(svc, "ServiceGamesPlayed", "Number")

                # Raw serve numbers (dividend/divisor)
                stats[f"{prefix}_first_serve_in"] = self._safe_num(svc, "FirstServe", "Dividend")
                stats[f"{prefix}_first_serve_total"] = self._safe_num(svc, "FirstServe", "Divisor")
                stats[f"{prefix}_first_serve_won"] = self._safe_num(svc, "FirstServePointsWon", "Dividend")
                stats[f"{prefix}_second_serve_won"] = self._safe_num(svc, "SecondServePointsWon", "Dividend")
                stats[f"{prefix}_second_serve_total"] = self._safe_num(svc, "SecondServePointsWon", "Divisor")
                stats[f"{prefix}_bp_saved"] = self._safe_num(svc, "BreakPointsSaved", "Dividend")
                stats[f"{prefix}_bp_faced"] = self._safe_num(svc, "BreakPointsSaved", "Divisor")

            # Return stats
            ret = set_stats.get("ReturnStats", {})
            if ret:
                stats[f"{prefix}_return_rating"] = self._safe_num(ret, "ReturnRating", "Number")
                stats[f"{prefix}_first_return_won_pct"] = self._safe_num(ret, "FirstServeReturnPointsWon", "Percent")
                stats[f"{prefix}_second_return_won_pct"] = self._safe_num(ret, "SecondServeReturnPointsWon", "Percent")
                stats[f"{prefix}_bp_converted_pct"] = self._safe_num(ret, "BreakPointsConverted", "Percent")
                stats[f"{prefix}_bp_opportunities"] = self._safe_num(ret, "BreakPointsConverted", "Divisor")
                stats[f"{prefix}_bp_converted"] = self._safe_num(ret, "BreakPointsConverted", "Dividend")
                stats[f"{prefix}_return_games"] = self._safe_num(ret, "ReturnGamesPlayed", "Number")

            # Point stats
            pts = set_stats.get("PointStats", {})
            if pts:
                stats[f"{prefix}_service_points_won_pct"] = self._safe_num(pts, "TotalServicePointsWon", "Percent")
                stats[f"{prefix}_return_points_won_pct"] = self._safe_num(pts, "TotalReturnPointsWon", "Percent")
                stats[f"{prefix}_total_points_won_pct"] = self._safe_num(pts, "TotalPointsWon", "Percent")
                stats[f"{prefix}_total_points_won"] = self._safe_num(pts, "TotalPointsWon", "Dividend")
                stats[f"{prefix}_total_points"] = self._safe_num(pts, "TotalPointsWon", "Divisor")

        # Only return if we got actual stats
        stat_keys = [k for k in stats.keys() if k.startswith("p1_") or k.startswith("p2_")]
        if len(stat_keys) >= 10:
            return stats
        return None

    @staticmethod
    def _safe_num(parent: dict, stat_key: str, field: str) -> float | None:
        """Safely extract a numeric value from nested Hawkeye JSON."""
        stat = parent.get(stat_key, {})
        if not isinstance(stat, dict):
            return None
        val = stat.get(field)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # -----------------------------------------------------------------------
    # Full year scraping
    # -----------------------------------------------------------------------
    def scrape_year(
        self, year: int, *, include_stats: bool = False
    ) -> pl.DataFrame:
        """
        Scrape all matches for a given year.

        1. Get tournament calendar
        2. For each tournament, get match results
        3. Optionally fetch detailed stats per match
        4. Return combined Polars DataFrame
        """
        tournaments = self.scrape_calendar(year)
        all_matches: list[dict[str, Any]] = []

        for tourn in tournaments:
            slug = tourn["tournament_slug"]
            tid = tourn["tournament_id"]
            tier = tourn["tier"]
            start_date = tourn["start_date"]
            name = tourn["tournament_name"]
            location = tourn["location"]

            # Skip team events (United Cup, Laver Cup)
            if tier in ("United Cup", "Laver Cup"):
                logger.info("  Skipping team event: %s", name)
                continue

            try:
                matches = self.scrape_tournament_results(slug, tid, year)
            except Exception as e:
                logger.error("  Failed to scrape %s: %s", slug, e)
                continue

            for m in matches:
                m["tournament_name"] = name
                m["tier"] = tier
                m["date"] = start_date
                m["location"] = location

                # Fetch stats if requested and raw_match_id exists
                if include_stats and m.get("raw_match_id"):
                    try:
                        stats = self.scrape_match_stats(year, tid, m["raw_match_id"])
                        if stats:
                            m.update(stats)
                    except Exception as e:
                        logger.debug("Stats failed for %s: %s", m["raw_match_id"], e)

                all_matches.append(m)

            # Batch save every 10 tournaments
            if len(all_matches) > 0 and len(all_matches) % 500 == 0:
                logger.info("  Progress: %d matches scraped so far", len(all_matches))

        if not all_matches:
            logger.warning("No matches found for %d", year)
            return pl.DataFrame()

        df = pl.DataFrame(all_matches)

        # Save to Parquet
        self._append_to_parquet(all_matches)
        logger.info("Scraped %d matches for %d", len(df), year)
        return df

    def scrape_years(
        self, start_year: int, end_year: int, *, include_stats: bool = False
    ) -> pl.DataFrame:
        """Scrape multiple years of ATP data."""
        all_dfs = []
        for year in range(start_year, end_year + 1):
            df = self.scrape_year(year, include_stats=include_stats)
            if len(df) > 0:
                all_dfs.append(df)
        return pl.concat(all_dfs, how="diagonal") if all_dfs else pl.DataFrame()

    # -----------------------------------------------------------------------
    # Player mapping
    # -----------------------------------------------------------------------
    def build_player_mapping(self, matches_path: str) -> pl.DataFrame:
        """
        Build a player_id → name mapping table from scraped matches.
        Critical for resolving upcoming match player IDs.
        """
        if not Path(matches_path).exists():
            logger.warning("No matches file at %s", matches_path)
            return pl.DataFrame()

        df = pl.read_parquet(matches_path)

        # Combine player1 and player2 columns
        p1 = df.select(
            pl.col("player1_id").alias("player_id"),
            pl.col("player1_name").alias("player_name"),
        )
        p2 = df.select(
            pl.col("player2_id").alias("player_id"),
            pl.col("player2_name").alias("player_name"),
        )

        mapping = pl.concat([p1, p2]).unique(subset=["player_id"])

        out_path = "data/processed/player_mapping.parquet"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        mapping.write_parquet(out_path)
        logger.info("Player mapping: %d unique players → %s", len(mapping), out_path)
        return mapping

    # -----------------------------------------------------------------------
    # Parquet I/O
    # -----------------------------------------------------------------------
    def _append_to_parquet(self, records: list[dict[str, Any]]):
        """Append records to the raw matches Parquet file."""
        path = self.output_dir / "matches.parquet"
        new_df = pl.DataFrame(records)

        if path.exists():
            existing = pl.read_parquet(path)
            combined = pl.concat([existing, new_df], how="diagonal")
            # Deduplicate by composite key (match_id already prefixed with tid)
            if "match_id" in combined.columns:
                combined = combined.unique(subset=["match_id", "tournament_id"], keep="last")
            combined.write_parquet(path)
            logger.info("Appended %d records → %s (total: %d)", len(new_df), path, len(combined))
        else:
            new_df.write_parquet(path)
            logger.info("Created %s with %d records", path, len(new_df))

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_player_id(href: str) -> str:
        """Extract player ID from profile URL path.
        /en/players/jannik-sinner/s0ag/overview → s0ag
        """
        parts = href.strip("/").split("/")
        if len(parts) >= 4 and parts[1] == "players":
            return parts[3]
        return ""

    @staticmethod
    def _parse_date(date_str: str, year: int) -> str:
        """Parse ATP date formats → YYYY-MM-DD.

        Common formats:
            '29 December, 2023 - 7 January, 2024'
            '1 - 7 January, 2024'
            '31 December, 2023 - 7 January, 2024'
        """
        if not date_str:
            return f"{year}-01-01"

        # Take the first date part (before the dash for ranges)
        parts = date_str.split(" - ")
        first = parts[0].strip()

        # Try various formats
        for fmt in [
            "%d %B, %Y",    # 29 December, 2023
            "%d %B %Y",     # 29 December 2023
            "%B %d, %Y",    # December 29, 2023
        ]:
            try:
                return datetime.strptime(first, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue

        # If it's just a day number (from a range like "1 - 7 January, 2024")
        if first.isdigit() and len(parts) > 1:
            # Parse the end part for month/year
            end = parts[1].strip()
            for fmt in ["%d %B, %Y", "%d %B %Y"]:
                try:
                    end_date = datetime.strptime(end, fmt)
                    return end_date.replace(day=int(first)).strftime("%Y-%m-%d")
                except ValueError:
                    continue

        return f"{year}-01-01"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = {}
    config_path = Path("config/model_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f).get("scraping", {})

    scraper = ATPScraper(config)

    # CLI usage: python -m scraper.atp_scraper [year]
    #        or: python -m scraper.atp_scraper [start_year] [end_year]
    args = sys.argv[1:]
    if len(args) >= 2:
        start = int(args[0])
        end = int(args[1])
        df = scraper.scrape_years(start, end, include_stats=False)
        print(f"\nScraped {len(df)} match records for {start}-{end}")
    else:
        year = int(args[0]) if args else 2024
        df = scraper.scrape_year(year, include_stats=False)
        print(f"\nScraped {len(df)} match records for {year}")

    if len(df) > 0:
        print(df.head(5))
