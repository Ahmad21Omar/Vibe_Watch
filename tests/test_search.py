"""Unit tests for vector_store.search() -- WITHOUT a real Qdrant.

search() is deliberately a thin adapter: it hands the query to Qdrant and reshapes each
hit into a flat dict of `{"score": ..., **payload}`. That reshaping is exactly where a
silent bug would hurt -- a dropped score, a forgotten `with_payload`, or the wrong result
order would corrupt every recommendation downstream. So we test the adapter logic against a
fake client that records how it was called and returns canned points. No network, no Docker,
runs in milliseconds -- the same "fast, pure" philosophy as the rest of the suite.
"""

from types import SimpleNamespace

from vibewatch.vector_store import COLLECTION_NAME, _build_filter, search


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


# --- metadata filters (Step 4) ---------------------------------------------------------


def test_no_filters_build_no_filter():
    # A pure mood query must NOT smuggle an empty filter into Qdrant -- None means
    # "search everything", which is the intended default.
    assert _build_filter() is None


def test_media_type_filter_matches_exact_value():
    flt = _build_filter(media_type="movie")
    (cond,) = flt.must
    assert cond.key == "media_type"
    assert cond.match.value == "movie"


def test_genres_filter_is_any_of():
    # Overlap semantics: a title kept if it has ANY requested genre, not all of them.
    flt = _build_filter(genres=["Drama", "Thriller"])
    (cond,) = flt.must
    assert cond.key == "genres"
    assert cond.match.any == ["Drama", "Thriller"]


def test_release_year_bounds_are_inclusive_range():
    flt = _build_filter(release_year_min=2010, release_year_max=2020)
    (cond,) = flt.must
    assert cond.key == "release_year"
    assert cond.range.gte == 2010
    assert cond.range.lte == 2020


def test_open_ended_year_leaves_other_bound_none():
    # Only a lower bound: "since 2015" must not accidentally cap the upper end.
    flt = _build_filter(release_year_min=2015)
    (cond,) = flt.must
    assert cond.range.gte == 2015
    assert cond.range.lte is None


def test_multiple_filters_are_anded_together():
    flt = _build_filter(media_type="tv", genres=["Drama"], original_language="en")
    assert {cond.key for cond in flt.must} == {"media_type", "genres", "original_language"}


def test_search_forwards_filter_to_qdrant():
    client = _RecordingClient([])
    search(client, [0.1] * 3072, media_type="movie", release_year_min=2015)
    query_filter = client.calls[0]["query_filter"]
    assert query_filter is not None
    assert {cond.key for cond in query_filter.must} == {"media_type", "release_year"}


def test_search_without_filters_passes_none():
    client = _RecordingClient([])
    search(client, [0.1] * 3072)
    assert client.calls[0]["query_filter"] is None
