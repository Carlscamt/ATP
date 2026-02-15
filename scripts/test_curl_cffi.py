"""
Test ATP scraping with curl_cffi (TLS fingerprint impersonation).
curl_cffi bypasses many bot detection systems by mimicking real browser TLS.
"""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_curl_cffi")

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    logger.error("curl_cffi not installed. Run: pip install curl_cffi")
    exit(1)

from bs4 import BeautifulSoup

Path("data/debug").mkdir(parents=True, exist_ok=True)


def test_endpoint(name, url, save_as):
    logger.info("=" * 60)
    logger.info("TEST: %s", name)
    logger.info("URL:  %s", url)

    try:
        resp = cffi_requests.get(url, impersonate="chrome")
        logger.info("Status: %d", resp.status_code)
        logger.info("Length: %d bytes", len(resp.text))

        # Save
        save_path = f"data/debug/{save_as}"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(resp.text)

        if resp.status_code == 403:
            logger.error("  -> BLOCKED (403)")
            return False
        if resp.status_code != 200:
            logger.error("  -> Unexpected status: %d", resp.status_code)
            return False

        # Check content
        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else "no title"
        logger.info("Title: %s", title)

        # Look for tournament data
        tourney_rows = soup.select("tr.tourney-result")
        tourney_titles = soup.select("a.tourney-title")
        player_links = soup.select("a[href*='players']")
        archive_links = soup.select("a[href*='scores/archive']")

        logger.info("  tourney-result rows: %d", len(tourney_rows))
        logger.info("  tourney-title links: %d", len(tourney_titles))
        logger.info("  player links: %d", len(player_links))
        logger.info("  archive links: %d", len(archive_links))

        for t in tourney_titles[:5]:
            logger.info("    -> %s", t.get_text(strip=True))

        for p in player_links[:5]:
            logger.info("    -> Player: %s | %s", p.get_text(strip=True), p.get("href", "")[:60])

        has_data = len(tourney_rows) > 0 or len(tourney_titles) > 0 or len(player_links) > 0
        if has_data:
            logger.info("  -> SUCCESS — data found")
        else:
            logger.warning("  -> Page loaded but no target data found (may be JS-rendered)")

        logger.info("  Saved to %s", save_path)
        return has_data

    except Exception as e:
        logger.error("  -> Error: %s", e)
        return False


if __name__ == "__main__":
    logger.info("ATP SCRAPER — curl_cffi TLS Fingerprint Test")
    logger.info("")

    results = {}

    # Test 1: Calendar
    results["calendar"] = test_endpoint(
        "Calendar 2024",
        "https://www.atptour.com/en/scores/results-archive?year=2024",
        "cffi_calendar.html",
    )
    time.sleep(2)

    # Test 2: Tournament results
    results["results"] = test_endpoint(
        "Australian Open 2024 Results",
        "https://www.atptour.com/en/scores/archive/australian-open/580/2024/results",
        "cffi_results.html",
    )
    time.sleep(2)

    # Test 3: Current scores
    results["current"] = test_endpoint(
        "Current Scores",
        "https://www.atptour.com/en/scores/current",
        "cffi_current.html",
    )
    time.sleep(2)

    # Test 4: Rankings (another useful endpoint)
    results["rankings"] = test_endpoint(
        "ATP Rankings",
        "https://www.atptour.com/en/rankings/singles",
        "cffi_rankings.html",
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s", icon, name)

    if all(results.values()):
        logger.info("\nAll endpoints accessible with curl_cffi!")
    elif any(results.values()):
        logger.info("\nPartial success — some endpoints work, some may need browser rendering")
    else:
        logger.info("\nAll blocked — will need Playwright browser automation")
