"""Shared retry policy for Gemini API calls.

Both halves of the pipeline talk to the same API and hit the same transient failures:
429 (rate limit) and 5xx (server busy). The embedding step has had careful retry handling
since step 3; the generation step had none -- and a real `docker compose` run promptly
answered with a 503 "high demand" and a stack trace instead of a recommendation.

Rather than copying the logic, it lives here once:

- **Retry 429 and 5xx, nothing else.** A 401 (bad key) or 400 (bad request) is our own
  mistake; retrying it just delays a failure we need to see immediately.
- **Prefer the server's own retry delay.** The API often replies with `retryDelay: '42s'`.
  Guessing a backoff is a fallback; when the server tells us how long to wait, that number
  is always better than ours.
- **`sleep` is injectable** so the tests exercise the real policy in milliseconds instead
  of actually waiting out a backoff.
"""

import time
from collections.abc import Callable

from google.genai import errors

MAX_RETRIES = 6


def server_retry_delay(error: errors.APIError) -> float | None:
    """Read the retry delay the API sends us, e.g. {'retryDelay': '42s'}.

    The field is nested and optional, so we dig for it defensively and return None if it
    is not there.
    """
    details = getattr(error, "details", None) or {}
    for detail in details.get("error", {}).get("details", []):
        delay = detail.get("retryDelay")
        if delay:
            return float(delay.rstrip("s"))
    return None


def call_with_retry[T](
    operation: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `operation`, retrying transient API failures with the server's own backoff."""
    last_error: errors.APIError | None = None

    for attempt in range(max_retries):
        try:
            return operation()

        except errors.APIError as error:
            # 429 = rate limit, 5xx = server hiccup. Both are worth retrying. Anything
            # else is our fault: fail immediately and loudly instead of hiding it behind
            # six slow retries.
            if error.code != 429 and error.code < 500:
                raise

            last_error = error
            wait = server_retry_delay(error) or 2**attempt
            print(f"    API error {error.code}, waiting {wait:.0f}s...")
            sleep(wait + 1)  # +1s safety margin

    raise last_error  # all retries exhausted
