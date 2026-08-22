"""Tests for the embedded API -- the single-process deployment shape.

This is the piece that lets a free host (one process, one port) run the same architecture
as everything else: the API on loopback, the UI still speaking HTTP to it. Two properties
are worth pinning, and both would fail confusingly in production if they broke:

- it must return only once the server actually ANSWERS. Returning early means the first
  user request hits a socket nobody is listening on, and the app reports "cannot reach the
  API" while being perfectly healthy.
- it must stay opt-in. An automatic embedded server would mean a UI that cannot reach its
  real API silently starts serving from a hidden second copy of the pipeline.
"""

import httpx

from vibewatch.config import Settings
from vibewatch.embedded import serve_api_in_background


def test_it_is_opt_in_and_off_by_default():
    # The default deployment is separate services. Embedding must be a decision.
    assert Settings(_env_file=None).embedded_api is False


def test_the_server_answers_by_the_time_it_returns():
    # A free port, so the test cannot collide with a locally running API.
    base_url = serve_api_in_background(port=8765)

    # No sleep, no retry: if the function returned, the server owes us an answer NOW.
    response = httpx.get(f"{base_url}/openapi.json", timeout=5)

    assert response.status_code == 200
    assert "/recommend" in response.json()["paths"]


def test_the_embedded_api_serves_the_real_application():
    # Not a stub: the same routes, the same validation. An empty query is rejected by the
    # schema here exactly as it would be by the standalone service.
    base_url = serve_api_in_background(port=8766)

    assert httpx.post(f"{base_url}/recommend", json={"query": ""}, timeout=5).status_code == 422


def test_it_binds_loopback_only():
    # Binding 0.0.0.0 would publish an unauthenticated pipeline that spends LLM quota on
    # every request -- on a host whose whole point is being publicly reachable.
    from vibewatch import embedded

    assert embedded.HOST == "127.0.0.1"
