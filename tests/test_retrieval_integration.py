"""End-to-end retrieval test against LIVE services (Qdrant + Gemini).

This is the smoke test that proves the whole retrieval half actually works: embed a real
mood query with Gemini, search the populated Qdrant index, get grounded hits back. Unlike
the rest of the suite it needs a running Qdrant with the `titles` collection populated AND a
Gemini API key -- so it is marked `integration` and DESELECTED by default (see pyproject).

Run it explicitly, after `docker compose up -d` and `python -m scripts.index_titles`:

    pytest -m integration

It skips itself cleanly (never fails) when the services or key are absent, so a plain
`pytest` on a fresh checkout stays green.

We assert STRUCTURE, not specific titles: which movies rank top shifts as the TMDb catalogue
changes, so pinning "Lost must be #1" would make the test flaky. What must always hold: we
get results, best-first, with self-contained payloads and cosine scores in range.
"""

import pytest

from vibewatch.config import settings
from vibewatch.vector_store import COLLECTION_NAME, get_client, search

pytestmark = pytest.mark.integration


def _index_ready() -> bool:
    """True only if Qdrant is reachable AND the collection actually has points."""
    try:
        client = get_client()
        return (
            client.collection_exists(COLLECTION_NAME)
            and client.count(COLLECTION_NAME).count > 0
        )
    except Exception:
        return False


requires_services = pytest.mark.skipif(
    not settings.gemini_api_key or not _index_ready(),
    reason="needs GEMINI_API_KEY and a populated Qdrant 'titles' collection",
)


@requires_services
def test_mood_query_returns_grounded_ranked_hits():
    from vibewatch.embeddings import embed_query

    query_vector = embed_query("survival, people fighting to stay alive in a hostile world")
    hits = search(get_client(), query_vector, limit=5)

    # We asked for 5 and the index holds ~900 titles, so we must get exactly 5 back.
    assert len(hits) == 5

    # Every hit is self-contained -- the payload we need to filter on and to show the user
    # travelled with the vector, so no second DB lookup is needed.
    for hit in hits:
        assert hit["media_type"] in {"movie", "tv"}
        assert isinstance(hit["title"], str) and hit["title"]
        assert isinstance(hit["genres"], list)
        # Cosine similarity lives in [-1, 1]; anything outside means a wiring/metric bug.
        assert -1.0 <= hit["score"] <= 1.0

    # Qdrant returns best-first; the ranking is the whole point of retrieval.
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


@requires_services
def test_media_type_filter_returns_only_movies():
    # A hard filter must actually constrain the result set, not just re-rank it.
    from vibewatch.embeddings import embed_query

    query_vector = embed_query("survival, people fighting to stay alive in a hostile world")
    hits = search(get_client(), query_vector, limit=5, media_type="movie")

    assert hits, "expected at least one movie to match"
    assert all(hit["media_type"] == "movie" for hit in hits)
