"""Unit tests for the shared Gemini retry policy.

Retry code is notoriously untested -- it only runs when something is already going wrong,
which is exactly when you cannot afford it to be broken. These tests exercise the real
policy with a fake `sleep`, so the full backoff behaviour is verified in milliseconds.

The behaviour that matters:
- transient failures (429, 5xx) are retried, permanent ones (4xx) are not;
- the server's own `retryDelay` wins over our guess;
- giving up re-raises the real API error, not a generic one.
"""

import pytest
from google.genai import errors

from vibewatch.gemini import call_with_retry, server_retry_delay


def _api_error(code: int, retry_delay: str | None = None) -> errors.APIError:
    """An APIError shaped like the ones the SDK raises."""
    response_json = {"error": {"code": code, "message": "boom", "status": "ERROR"}}
    if retry_delay:
        response_json["error"]["details"] = [{"retryDelay": retry_delay}]
    return errors.APIError(code, response_json)


def test_succeeds_without_sleeping_when_nothing_goes_wrong():
    slept = []
    assert call_with_retry(lambda: "ok", sleep=slept.append) == "ok"
    assert slept == []


def test_retries_a_rate_limit_and_returns_the_eventual_result():
    attempts = []
    slept = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise _api_error(429)
        return "ok"

    assert call_with_retry(flaky, sleep=slept.append) == "ok"
    assert len(attempts) == 3
    assert len(slept) == 2  # one wait per failure, none after success


def test_retries_a_busy_server():
    # The real case that motivated this: a 503 "model is experiencing high demand" during
    # generation used to surface as a stack trace instead of a recommendation.
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise _api_error(503)
        return "recommendation"

    assert call_with_retry(flaky, sleep=lambda _seconds: None) == "recommendation"


def test_does_not_retry_a_bad_request():
    # A 400/401 is our own mistake. Retrying it six times only delays the error we need.
    attempts = []

    def broken():
        attempts.append(1)
        raise _api_error(400)

    with pytest.raises(errors.APIError):
        call_with_retry(broken, sleep=lambda _seconds: None)

    assert len(attempts) == 1


def test_prefers_the_delay_the_server_asked_for():
    slept = []

    def flaky():
        if not slept:
            raise _api_error(429, retry_delay="42s")
        return "ok"

    call_with_retry(flaky, sleep=slept.append)

    # 42s from the server (+1s margin), not our 2**0 = 1s guess.
    assert slept == [43]


def test_falls_back_to_exponential_backoff_without_a_server_delay():
    slept = []

    def always_busy():
        raise _api_error(503)

    with pytest.raises(errors.APIError):
        call_with_retry(always_busy, max_retries=4, sleep=slept.append)

    # 2**0, 2**1, 2**2, 2**3 -- each with the +1s safety margin.
    assert slept == [2, 3, 5, 9]


def test_gives_up_with_the_real_error_after_the_last_attempt():
    def always_busy():
        raise _api_error(503)

    with pytest.raises(errors.APIError) as error:
        call_with_retry(always_busy, max_retries=2, sleep=lambda _seconds: None)

    assert error.value.code == 503


def test_server_retry_delay_is_none_when_the_error_carries_no_hint():
    assert server_retry_delay(_api_error(429)) is None
    assert server_retry_delay(_api_error(429, retry_delay="7s")) == 7.0
