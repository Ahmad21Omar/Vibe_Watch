"""Unit tests for vector_store.search() -- WITHOUT a real Qdrant.

search() is deliberately a thin adapter: it hands the query to Qdrant and reshapes each
hit into a flat dict of `{"score": ..., **payload}`. That reshaping is exactly where a
silent bug would hurt -- a dropped score, a forgotten `with_payload`, or the wrong result
order would corrupt every recommendation downstream. So we test the adapter logic against a
fake client that records how it was called and returns canned points. No network, no Docker,
runs in milliseconds -- the same "fast, pure" philosophy as the rest of the suite.
"""

from types import SimpleNamespace

from vibewatch.vector_store import COLLECTION_NAME, search


class _RecordingClient:
    """A stand-in for QdrantClient that records the query and returns canned points.

    Qdrant's real `query_points` returns an object with a `.points` list, each point having
    `.score` and `.payload`. We reproduce just that shape -- nothing more is needed.
    """

    def __init__(self, points):
        self._points = points
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self._points)


def _point(score, payload):
    return SimpleNamespace(score=score, payload=payload)


def test_search_flattens_score_and_payload_in_order():
    # The result must be score + the whole payload, one flat dict per hit, in the order
    # Qdrant returned them (already sorted best-first). If search() ever reorders or drops
    # a field, this breaks.
    client = _RecordingClient(
        [
            _point(0.91, {"title": "Lost", "media_type": "tv"}),
            _point(0.88, {"title": "Mad Max", "media_type": "movie"}),
        ]
    )
    hits = search(client, [0.1] * 3072, limit=2)
    assert hits == [
        {"score": 0.91, "title": "Lost", "media_type": "tv"},
        {"score": 0.88, "title": "Mad Max", "media_type": "movie"},
    ]


def test_search_forwards_query_parameters():
    # The angle-based search only works if we ask Qdrant the right question: the right
    # collection, the caller's vector and limit, and with_payload so the hit is
    # self-contained (no second lookup). Pin all four so a refactor can't quietly drop one.
    client = _RecordingClient([])
    search(client, [0.1, 0.2, 0.3], limit=7)

    (call,) = client.calls
    assert call["collection_name"] == COLLECTION_NAME
    assert call["query"] == [0.1, 0.2, 0.3]
    assert call["limit"] == 7
    assert call["with_payload"] is True


def test_search_default_limit_is_five():
    client = _RecordingClient([])
    search(client, [0.0] * 3072)
    assert client.calls[0]["limit"] == 5
