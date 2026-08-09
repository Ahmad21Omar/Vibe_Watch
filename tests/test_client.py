"""Tests for the API client -- no server, httpx transport faked.

The client is thin, so what matters is not the happy path but the ERROR path: this is the
only place that turns an HTTP failure into something a person can read. If it leaks an
httpx traceback into the UI, or worse swallows a failure, the operator learns nothing.
"""

import httpx
import pytest

from vibewatch import client
from vibewatch.client import ApiError


def _transport(handler):
    """Install a fake httpx transport for the duration of a test."""
    return httpx.MockTransport(handler)


@pytest.fixture
def mock_http(monkeypatch):
    """Route httpx.post/get through a handler the test provides."""

    def install(handler):
        transport = _transport(handler)

        def fake_post(url, **kwargs):
            with httpx.Client(transport=transport) as http:
                return http.post(url, **kwargs)

        def fake_get(url, **kwargs):
            with httpx.Client(transport=transport) as http:
                return http.get(url, **kwargs)

        monkeypatch.setattr(httpx, "post", fake_post)
        monkeypatch.setattr(httpx, "get", fake_get)

    return install


def test_recommend_posts_the_query_and_returns_the_body(mock_http):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answer": "Watch it.", "hits": []})

    mock_http(handler)

    result = client.recommend("dark survival", limit=3, media_type="movie")

    assert seen["url"].endswith("/recommend")
    assert seen["body"] == {"query": "dark survival", "limit": 3, "media_type": "movie"}
    assert result["answer"] == "Watch it."


def test_an_unreachable_api_says_so_in_words(mock_http):
    # The most common failure in practice: the service is not running. The message must
    # name the address, or the user has no idea what to start.
    def handler(request):
        raise httpx.ConnectError("connection refused")

    mock_http(handler)

    with pytest.raises(ApiError) as error:
        client.recommend("dark survival")

    assert "Cannot reach the API" in str(error.value)
    assert "http" in str(error.value)  # the configured URL is included


def test_an_error_response_surfaces_the_servers_reason(mock_http):
    # FastAPI explains itself in `detail`. Dropping it would turn "index is empty" into
    # a bare 503 and cost the operator the one clue that matters.
    def handler(request):
        return httpx.Response(503, json={"detail": "index is empty"})

    mock_http(handler)

    with pytest.raises(ApiError) as error:
        client.recommend("dark survival")

    assert "503" in str(error.value)
    assert "index is empty" in str(error.value)


def test_a_non_json_error_body_is_still_reported(mock_http):
    # Not every error comes from FastAPI -- a proxy or gateway answers in plain text.
    def handler(request):
        return httpx.Response(502, text="Bad Gateway")

    mock_http(handler)

    with pytest.raises(ApiError) as error:
        client.recommend("dark survival")

    assert "Bad Gateway" in str(error.value)


def test_genres_unwraps_the_list(mock_http):
    mock_http(lambda request: httpx.Response(200, json={"genres": ["Drama", "Comedy"]}))

    assert client.genres() == ["Drama", "Comedy"]
