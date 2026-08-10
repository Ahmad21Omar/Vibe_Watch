"""Tests for the HTTP API -- FastAPI's TestClient, no server and no live services.

The API is a thin layer, so these tests deliberately do NOT re-check recommendation
quality (that is measured a layer down). What they check is what a layer like this is
actually for:

- the CONTRACT: does a request shape map to the right pipeline call, does the response
  carry the answer *and* its sources?
- the STATUS CODES: a malformed request must be 422 and a broken dependency 503. Getting
  these wrong is how a client ends up retrying a request that can never succeed -- or
  giving up on one that would work a second later.
"""

import pytest
from fastapi.testclient import TestClient

from vibewatch.api import app


def _state(**overrides):
    return {
        "answer": "Watch The Road (2009).",
        "hits": [
            {
                "score": 0.68,
                "tmdb_id": 20766,
                "media_type": "movie",
                "title": "The Road",
                "overview": "A father and his son walk through burned America.",
                "original_language": "en",
                "genres": ["Drama"],
                "release_year": 2009,
                "popularity": 20.0,
                "vote_average": 7.0,
                "vote_count": 4000,
                "poster_path": "/poster.jpg",
            }
        ],
        "inferred_filters": {},
        **overrides,
    }


@pytest.fixture
def client(monkeypatch):
    """A TestClient plus the log of pipeline calls the API made."""
    calls: list[dict] = []

    def fake_recommend(query, *, limit=5, **filters):
        calls.append({"query": query, "limit": limit, **filters})
        return _state()

    monkeypatch.setattr("vibewatch.api.recommend", fake_recommend)
    test_client = TestClient(app)
    test_client.calls = calls
    return test_client


# --- the contract -----------------------------------------------------------------------


def test_a_query_returns_the_answer_and_its_sources(client):
    response = client.post("/recommend", json={"query": "dark survival"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Watch The Road (2009)."
    # Sources always travel with the answer -- that is the API-level form of grounding.
    assert body["hits"][0]["title"] == "The Road"
    assert body["hits"][0]["score"] == 0.68


def test_only_the_query_is_required(client):
    client.post("/recommend", json={"query": "dark survival"})

    # No filters given must mean "search everything", not a stray constraint on every call.
    # `history` is always passed (empty for a first turn) -- the graph turns [] into None.
    assert client.calls == [{"query": "dark survival", "limit": 5, "history": []}]


def test_filters_reach_the_pipeline(client):
    client.post(
        "/recommend",
        json={
            "query": "dark survival",
            "limit": 3,
            "media_type": "movie",
            "genres": ["Drama"],
            "release_year_min": 2000,
        },
    )

    assert client.calls[-1] == {
        "query": "dark survival",
        "limit": 3,
        "history": [],
        "media_type": "movie",
        "genres": ["Drama"],
        "release_year_min": 2000,
    }


def test_inferred_filters_and_relaxation_are_reported(client, monkeypatch):
    # A client that cannot see these leaves the user guessing why the results look the way
    # they do -- the same reason the UI shows them.
    monkeypatch.setattr(
        "vibewatch.api.recommend",
        lambda query, **kwargs: _state(
            inferred_filters={"media_type": "movie"}, relaxed=True
        ),
    )

    body = client.post("/recommend", json={"query": "funny movies"}).json()

    assert body["inferred_filters"] == {"media_type": "movie"}
    assert body["relaxed"] is True


# --- status codes -----------------------------------------------------------------------


def test_an_empty_query_is_rejected_before_any_work_happens(client):
    response = client.post("/recommend", json={"query": ""})

    assert response.status_code == 422
    assert client.calls == [], "a malformed request must not reach the pipeline"


def test_an_invalid_media_type_is_rejected(client):
    # Catching this at the edge beats passing "documentary" down to Qdrant, where it
    # would silently match nothing and look like an empty catalogue.
    assert client.post(
        "/recommend", json={"query": "x", "media_type": "documentary"}
    ).status_code == 422


def test_an_absurd_limit_is_rejected(client):
    assert client.post("/recommend", json={"query": "x", "limit": 500}).status_code == 422
    assert client.post("/recommend", json={"query": "x", "limit": 0}).status_code == 422


def test_a_broken_dependency_is_503_not_500(client, monkeypatch):
    # 503 says "try again"; a 500 says "your request is broken" and invites the client to
    # give up on a request that would succeed a second later.
    def boom(query, **kwargs):
        raise ConnectionError("Qdrant unreachable")

    monkeypatch.setattr("vibewatch.api.recommend", boom)

    response = client.post("/recommend", json={"query": "dark survival"})

    assert response.status_code == 503
    assert "Qdrant unreachable" in response.json()["detail"]


# --- health -----------------------------------------------------------------------------


def test_health_is_ok_when_the_index_has_titles(client, monkeypatch):
    monkeypatch.setattr(
        "vibewatch.api.get_client",
        lambda: type("C", (), {"count": lambda self, name: type("R", (), {"count": 912})()})(),
    )

    body = client.get("/health").json()
    assert body == {"status": "ok", "indexed_titles": 912}


def test_health_reports_an_empty_index_as_unavailable(client, monkeypatch):
    # Readiness, not liveness: a running process with an empty index answers nothing
    # useful, and a 200 here would tell a load balancer to send traffic into a hole.
    monkeypatch.setattr(
        "vibewatch.api.get_client",
        lambda: type("C", (), {"count": lambda self, name: type("R", (), {"count": 0})()})(),
    )

    response = client.get("/health")
    assert response.status_code == 503
    assert "empty" in response.json()["detail"]


def test_health_reports_an_unreachable_index(client, monkeypatch):
    def boom():
        raise ConnectionError("connection refused")

    monkeypatch.setattr("vibewatch.api.get_client", boom)

    assert client.get("/health").status_code == 503


# --- conversations ----------------------------------------------------------------------


def test_history_is_forwarded_to_the_pipeline(client):
    # The client owns the conversation, so it must arrive intact -- a dropped history
    # turns "but shorter" into a search for the words "but shorter".
    client.post(
        "/recommend",
        json={"query": "but shorter", "history": ["dark survival movies"]},
    )

    assert client.calls[-1]["history"] == ["dark survival movies"]


def test_history_is_not_mistaken_for_a_filter(client):
    # `filters()` builds the Qdrant kwargs from the request fields. If `history` leaked in
    # there it would become a filter on a payload field that does not exist -- matching
    # nothing, and looking like an empty catalogue.
    client.post("/recommend", json={"query": "x", "history": ["a", "b"]})

    call = client.calls[-1]
    assert call["history"] == ["a", "b"]          # passed as its own argument...
    assert set(call) == {"query", "limit", "history"}  # ...and not as a filter


def test_an_unbounded_history_is_rejected(client):
    # An unbounded history is an unbounded prompt, and someone else's bill.
    response = client.post(
        "/recommend", json={"query": "x", "history": [f"turn {i}" for i in range(50)]}
    )

    assert response.status_code == 422
