"""
Diagnostic Test Script for ATP Scrapers
Tests each scraper component systematically to identify issues.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test 1: Verify all required imports are available."""
    logger.info("=" * 60)
    logger.info("TEST 1: Checking imports...")
    logger.info("=" * 60)
    
    issues = []
    
    try:
        import requests
        logger.info("✓ requests imported successfully")
    except ImportError as e:
        issues.append(f"✗ requests: {e}")
    
    try:
        import polars as pl
        logger.info("✓ polars imported successfully")
    except ImportError as e:
        issues.append(f"✗ polars: {e}")
    
    try:
        from bs4 import BeautifulSoup
        logger.info("✓ BeautifulSoup imported successfully")
    except ImportError as e:
        issues.append(f"✗ BeautifulSoup: {e}")
    
    try:
        from tenacity import retry
        logger.info("✓ tenacity imported successfully")
    except ImportError as e:
        issues.append(f"✗ tenacity: {e}")
    
    try:
        from playwright.sync_api import sync_playwright
        logger.info("✓ playwright imported successfully")
    except ImportError as e:
        issues.append(f"✗ playwright (optional): {e}")
        logger.warning("Playwright not available - Oddsportal scraping will fail")
    
    try:
        import yaml
        logger.info("✓ yaml imported successfully")
    except ImportError as e:
        issues.append(f"✗ yaml: {e}")
    
    return issues


def test_module_imports():
    """Test 2: Verify project modules can be imported."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Checking project module imports...")
    logger.info("=" * 60)
    
    issues = []
    
    try:
        from utils.error_handler import fetch_with_retry, rate_limit
        logger.info("✓ utils.error_handler imported successfully")
    except ImportError as e:
        issues.append(f"✗ utils.error_handler: {e}")
    
    try:
        from scraper.atp_scraper import ATPScraper
        logger.info("✓ scraper.atp_scraper imported successfully")
    except ImportError as e:
        issues.append(f"✗ scraper.atp_scraper: {e}")
    
    try:
        from scraper.odds_scraper import OddsScraper
        logger.info("✓ scraper.odds_scraper imported successfully")
    except ImportError as e:
        issues.append(f"✗ scraper.odds_scraper: {e}")
    
    try:
        from scraper.schedule_scraper import ScheduleScraper
        logger.info("✓ scraper.schedule_scraper imported successfully")
    except ImportError as e:
        issues.append(f"✗ scraper.schedule_scraper: {e}")
    
    return issues


def test_directory_structure():
    """Test 3: Verify required directories exist."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Checking directory structure...")
    logger.info("=" * 60)
    
    issues = []
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/upcoming",
        "data/elo",
        "data/predictions",
        "data/folds",
        "config",
        "logs"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"✓ {dir_path} exists")
        else:
            logger.warning(f"✗ {dir_path} does not exist - will be created on first run")
            issues.append(f"Missing directory: {dir_path}")
    
    return issues


def test_config_files():
    """Test 4: Verify configuration files exist and are valid."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Checking configuration files...")
    logger.info("=" * 60)
    
    issues = []
    
    config_path = Path("config/model_config.yaml")
    if config_path.exists():
        logger.info(f"✓ {config_path} exists")
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ {config_path} is valid YAML")
            
            # Check for API key
            api_key = config.get("scraping", {}).get("odds_api_key", "")
            if api_key:
                logger.info("✓ Odds API key is configured")
            else:
                logger.warning("✗ Odds API key not configured - odds scraping will fail")
                issues.append("Missing odds_api_key in config")
                
        except Exception as e:
            issues.append(f"Error reading config: {e}")
    else:
        issues.append(f"Config file not found: {config_path}")
    
    return issues


def test_network_connectivity():
    """Test 5: Verify network connectivity to ATP website."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Checking network connectivity...")
    logger.info("=" * 60)
    
    issues = []
    
    try:
        import requests
        
        # Test ATP Tour website
        logger.info("Testing connection to atptour.com...")
        response = requests.get(
            "https://www.atptour.com",
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        if response.status_code == 200:
            logger.info(f"✓ ATP Tour website accessible (status: {response.status_code})")
        else:
            logger.warning(f"⚠ ATP Tour returned status: {response.status_code}")
            issues.append(f"ATP website returned status {response.status_code}")
            
    except requests.exceptions.Timeout:
        issues.append("Network timeout connecting to ATP website")
    except requests.exceptions.ConnectionError as e:
        issues.append(f"Connection error: {e}")
    except Exception as e:
        issues.append(f"Network test failed: {e}")
    
    return issues


def test_atp_scraper_basic():
    """Test 6: Test basic ATP scraper functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Testing ATP scraper basic functionality...")
    logger.info("=" * 60)
    
    issues = []
    
    try:
        from scraper.atp_scraper import ATPScraper
        import yaml
        
        # Load config
        config_path = Path("config/model_config.yaml")
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("paths", {})
            config.update(full_config.get("scraping", {}))
        
        scraper = ATPScraper(config=config)
        logger.info("✓ ATPScraper instantiated successfully")
        
        # Test calendar scraping for current year
        logger.info("Testing calendar scraping for 2024...")
        tournaments = scraper.scrape_calendar(2024)
        
        if tournaments:
            logger.info(f"✓ Successfully scraped {len(tournaments)} tournaments")
            logger.info(f"  Sample tournament: {tournaments[0].get('tournament_name', 'N/A')}")
        else:
            logger.warning("✗ No tournaments found - website structure may have changed")
            issues.append("Calendar scraping returned no results")
            
    except Exception as e:
        logger.error(f"✗ ATP scraper test failed: {e}", exc_info=True)
        issues.append(f"ATP scraper error: {e}")
    
    return issues


def test_schedule_scraper_basic():
    """Test 7: Test schedule scraper functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Testing schedule scraper...")
    logger.info("=" * 60)
    
    issues = []
    
    try:
        from scraper.schedule_scraper import ScheduleScraper
        import yaml
        
        # Load config
        config_path = Path("config/model_config.yaml")
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("paths", {})
        
        scraper = ScheduleScraper(config=config)
        logger.info("✓ ScheduleScraper instantiated successfully")
        
        # Check for player mapping
        mapping_path = Path(config.get("player_mapping", "data/processed/player_mapping.parquet"))
        if mapping_path.exists():
            logger.info(f"✓ Player mapping exists at {mapping_path}")
        else:
            logger.warning(f"✗ Player mapping not found at {mapping_path}")
            logger.warning("  Schedule scraper will not be able to resolve player IDs")
            issues.append("Missing player_mapping.parquet")
        
        # Test upcoming matches scraping
        logger.info("Testing upcoming matches scraping...")
        df = scraper.scrape_upcoming_matches(days_ahead=7)
        
        if len(df) > 0:
            logger.info(f"✓ Successfully scraped {len(df)} upcoming matches")
        else:
            logger.warning("✗ No upcoming matches found")
            issues.append("Schedule scraping returned no results")
            
    except Exception as e:
        logger.error(f"✗ Schedule scraper test failed: {e}", exc_info=True)
        issues.append(f"Schedule scraper error: {e}")
    
    return issues


def test_odds_scraper_basic():
    """Test 8: Test odds scraper functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: Testing odds scraper...")
    logger.info("=" * 60)
    
    issues = []
    
    try:
        from scraper.odds_scraper import OddsScraper
        import yaml
        
        # Load config
        config_path = Path("config/model_config.yaml")
        config = {}
        api_key = ""
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("paths", {})
            api_key = full_config.get("scraping", {}).get("odds_api_key", "")
        
        scraper = OddsScraper(config=config, api_key=api_key)
        logger.info("✓ OddsScraper instantiated successfully")
        
        if api_key:
            logger.info("✓ API key configured - will test The Odds API")
            # Note: We won't actually call the API to avoid using quota
            logger.info("  (Skipping actual API call to preserve quota)")
        else:
            logger.warning("✗ No API key configured")
            logger.warning("  Odds scraping will fall back to Oddsportal (requires Playwright)")
            issues.append("No odds API key configured")
            
    except Exception as e:
        logger.error(f"✗ Odds scraper test failed: {e}", exc_info=True)
        issues.append(f"Odds scraper error: {e}")
    
    return issues


def main():
    """Run all diagnostic tests."""
    logger.info("\n" + "=" * 60)
    logger.info("ATP SCRAPER DIAGNOSTIC TEST SUITE")
    logger.info("=" * 60)
    
    all_issues = []
    
    # Run all tests
    all_issues.extend(test_imports())
    all_issues.extend(test_module_imports())
    all_issues.extend(test_directory_structure())
    all_issues.extend(test_config_files())
    all_issues.extend(test_network_connectivity())
    all_issues.extend(test_atp_scraper_basic())
    all_issues.extend(test_schedule_scraper_basic())
    all_issues.extend(test_odds_scraper_basic())
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    if not all_issues:
        logger.info("✓ All tests passed! Scrapers should be working correctly.")
        return 0
    else:
        logger.warning(f"\n⚠ Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            logger.warning(f"  {i}. {issue}")
        
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDATIONS:")
        logger.info("=" * 60)
        
        # Provide specific recommendations
        if any("playwright" in str(issue).lower() for issue in all_issues):
            logger.info("• Install Playwright: pip install playwright && playwright install chromium")
        
        if any("api_key" in str(issue).lower() or "odds" in str(issue).lower() for issue in all_issues):
            logger.info("• Configure Odds API key in config/model_config.yaml")
            logger.info("  Get a free key at: https://the-odds-api.com/")
        
        if any("player_mapping" in str(issue).lower() for issue in all_issues):
            logger.info("• Run ATP scraper first to build player mapping:")
            logger.info("  python -c 'from scraper.atp_scraper import ATPScraper; s=ATPScraper(); s.scrape_year(2024)'")
        
        if any("directory" in str(issue).lower() for issue in all_issues):
            logger.info("• Create missing directories - they will be auto-created on first run")
        
        if any("import" in str(issue).lower() or "module" in str(issue).lower() for issue in all_issues):
            logger.info("• Python path issue detected - add project root to PYTHONPATH")
            logger.info("  Or run from project root with: python -m tests.test_scrapers_diagnostic")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
