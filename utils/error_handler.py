"""ATP Betting Model — Utility Functions"""
import functools
import logging
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import requests

logger = logging.getLogger(__name__)


def log_and_continue(func):
    """Decorator: catch errors, log them, return None instead of crashing."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                "Error in %s: %s", func.__name__, str(e), exc_info=True
            )
            return None
    return wrapper


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )),
    before_sleep=lambda retry_state: logger.warning(
        "Retry %d for %s after %s",
        retry_state.attempt_number,
        retry_state.fn.__name__,
        retry_state.outcome.exception(),
    ),
)
def fetch_with_retry(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: int = 30,
) -> requests.Response:
    """GET request with exponential-backoff retry."""
    response = requests.get(url, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def rate_limit(seconds: float = 1.0):
    """Decorator: enforce minimum delay between successive calls."""
    def decorator(func):
        last_called: list[float] = [0.0]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.monotonic() - last_called[0]
            if elapsed < seconds:
                time.sleep(seconds - elapsed)
            last_called[0] = time.monotonic()
            return func(*args, **kwargs)

        return wrapper
    return decorator
