"""
Playwright HTTP Client for ATP Scraping

Provides a Playwright-based HTTP client that:
- Renders JavaScript (bypasses bot detection)
- Handles cookies and sessions
- Includes realistic browser fingerprinting
- Implements rate limiting
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger("playwright_client")

# Try to import Playwright
try:
    from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - falling back to requests")


class PlaywrightClient:
    """
    HTTP client using Playwright for JavaScript rendering.
    Bypasses bot detection by rendering pages like a real browser.
    """
    
    def __init__(
        self,
        rate_limit_seconds: float = 2.0,
        headless: bool = True,
        timeout: int = 30000,
    ):
        """
        Initialize Playwright client.
        
        Args:
            rate_limit_seconds: Minimum delay between requests
            headless: Run browser in headless mode
            timeout: Page load timeout in milliseconds
        """
        self.rate_limit_seconds = rate_limit_seconds
        self.headless = headless
        self.timeout = timeout
        
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._last_request: float = 0.0
        
        if PLAYWRIGHT_AVAILABLE:
            self._init_browser()
        else:
            raise ImportError("Playwright is not installed. Run: pip install playwright")
    
    def _init_browser(self):
        """Initialize Playwright browser with realistic settings."""
        try:
            from playwright_stealth.stealth import stealth_sync

            self._playwright = sync_playwright().start()
            
            # Launch browser with anti-detection measures
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                ]
            )
            
            # Create context with realistic browser fingerprint
            self._context = self._browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/122.0.0.0 Safari/537.36'
                ),
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                ignore_https_errors=True,
            )
            
            self._page = self._context.new_page()

            # Apply stealth
            stealth_sync(self._page)
            
            # Set default timeout
            self._page.set_default_timeout(self.timeout)
            
            logger.info("Playwright browser initialized successfully with stealth")
            
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            raise
    
    def _throttle(self):
        """Apply rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request = time.monotonic()
    
    def get(self, url: str, wait_for_selector: str = None, wait_timeout: int = 10000, **kwargs) -> requests.Response:
        """
        Fetch a URL using Playwright and return a requests.Response-like object.
        
        Args:
            url: URL to fetch
            wait_for_selector: Optional CSS selector to wait for before returning
            wait_timeout: Timeout for waiting (ms)
        
        This allows Playwright to be used as a drop-in replacement for requests.
        """
        self._throttle()
        
        try:
            # Add random mouse movements and delays to appear more human-like
            self._page.mouse.move(100, 100)
            
            # Navigate to page
            response = self._page.goto(url, wait_until='networkidle', timeout=wait_timeout)
            
            # Wait for specific selector if provided
            if wait_for_selector:
                try:
                    self._page.wait_for_selector(wait_for_selector, timeout=wait_timeout)
                except Exception as e:
                    logger.warning(f"Selector {wait_for_selector} not found: {e}")
            else:
                # Default wait for dynamic content
                self._page.wait_for_timeout(5000)
            
            # Get final URL (in case of redirects)
            final_url = self._page.url
            
            # Create a mock response object
            content = self._page.content()
            status = response.status if response else 200
            
            # Get headers from context instead of page
            headers = {}
            try:
                # Try to get any available headers
                headers = {"content-type": "text/html"}
            except:
                pass
            
            # Create response-like object
            mock_response = MockResponse(
                url=final_url,
                status_code=status,
                text=content,
                headers=headers,
            )
            
            logger.debug(f"Fetched {url} - Status: {status}")
            return mock_response
            
        except Exception as e:
            logger.error(f"Playwright failed to fetch {url}: {e}")
            # Fall back to requests if Playwright fails
            logger.info("Falling back to requests library")
            return self._fallback_get(url)
    
    def _fallback_get(self, url: str) -> requests.Response:
        """Fallback to requests library if Playwright fails."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        return response
    
    def close(self):
        """Clean up Playwright resources."""
        try:
            if self._page:
                self._page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
            logger.info("Playwright browser closed")
        except Exception as e:
            logger.warning(f"Error closing Playwright: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MockResponse:
    """
    Mock response object that mimics requests.Response
    for compatibility with existing code.
    """
    
    def __init__(
        self,
        url: str,
        status_code: int,
        text: str,
        headers: dict[str, str],
    ):
        self.url = url
        self.status_code = status_code
        self.text = text
        self.headers = headers
    
    def json(self) -> dict[str, Any]:
        """Parse response as JSON."""
        import json
        return json.loads(self.text)
    
    def raise_for_status(self):
        """Raise exception for error status codes."""
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error",
                response=self
            )


# Decorator for using Playwright client with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def fetch_with_playwright(url: str, **kwargs) -> requests.Response:
    """
    Fetch a URL using Playwright with retry logic.
    
    Usage:
        response = fetch_with_playwright("https://example.com")
        soup = BeautifulSoup(response.text, "lxml")
    """
    with PlaywrightClient() as client:
        return client.get(url, **kwargs)


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    test_url = "https://www.atptour.com/en/scores/results-archive?year=2024"
    
    print(f"Testing Playwright client with: {test_url}")
    
    with PlaywrightClient(rate_limit_seconds=3.0) as client:
        response = client.get(test_url)
        print(f"Status: {response.status_code}")
        print(f"URL: {response.url}")
        print(f"Content length: {len(response.text)} characters")
        
        if response.status_code == 200:
            print("✓ Successfully fetched page with Playwright!")
        else:
            print(f"✗ Failed with status {response.status_code}")
