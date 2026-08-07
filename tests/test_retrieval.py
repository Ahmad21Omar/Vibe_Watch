"""Unit tests for retrieve() -- the query -> hits seam, without real services.

retrieve() has one job: embed the query, then hand the vector (plus any filters) to the
search layer. Two modes exist -- dense (the measured-better default) and hybrid -- and both
are tested, because the hybrid path is kept as a runnable comparison, not dead code.

For hybrid the extra thing to verify is that the RAW QUERY TEXT reaches the sparse side.
The dense side needs the embedding, the sparse side needs the words; if the text stopped
arriving, the sparse vector would be empty and hybrid would silently collapse back to
semantic-only, with no error anywhere.

Verified with a fake embed function and a recording fake client: no Gemini, no Qdrant.
"""

from types import SimpleNamespace

from vibewatch.retrieval import retrieve
from vibewatch.vector_store import DENSE_VECTOR, SPARSE_VECTOR


class _RecordingClient:
    def __init__(self, points):
        self._points = points
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self._points)


def _fake_embed(vector=(0.5, 0.5, 0.5), seen=None):
    def _embed(text: str) -> list[float]:
        if seen is not None:
            seen["text"] = text
        return list(vector)

    return _embed


# --- dense, the default (it measured better than hybrid -- see retrieval.py) -------------


def test_dense_is_the_default():
    # Pinning the default is pinning a MEASUREMENT: hybrid scored recall@5 0.557 against
    # dense's 0.832 on the gold set. If someone flips this back, the suite says so.
    client = _RecordingClient([])
    retrieve("dark survival", client=client, embed=_fake_embed())

    assert "prefetch" not in client.calls[0], "default must be dense, not hybrid"
    assert client.calls[0]["using"] == DENSE_VECTOR


def test_dense_mode_embeds_query_and_returns_mapped_hits():
    client = _RecordingClient([SimpleNamespace(score=0.9, payload={"title": "Lost"})])
    seen = {}

    hits = retrieve(
        "dark survival", mode="dense", client=client, embed=_fake_embed(seen=seen), limit=3
    )

    # The user's words reached the embedder unchanged...
    assert seen["text"] == "dark survival"
    # ...its vector reached Qdrant unchanged, against the dense vector, with the limit...
    assert client.calls[0]["query"] == [0.5, 0.5, 0.5]
    assert client.calls[0]["using"] == DENSE_VECTOR
    assert client.calls[0]["limit"] == 3
    # ...and the hit came back flattened, exactly as search() produced it.
    assert hits == [{"score": 0.9, "title": "Lost"}]


def test_dense_mode_forwards_filters():
    client = _RecordingClient([])

    retrieve(
        "dark survival",
        mode="dense",
        client=client,
        embed=_fake_embed(),
        media_type="tv",
        genres=["Drama"],
    )

    query_filter = client.calls[0]["query_filter"]
    assert {cond.key for cond in query_filter.must} == {"media_type", "genres"}


def test_dense_mode_without_filters_searches_everything():
    client = _RecordingClient([])
    retrieve("dark survival", mode="dense", client=client, embed=_fake_embed())
    assert client.calls[0]["query_filter"] is None


# --- hybrid: opt-in, kept as a runnable comparison ------------------------------------


def test_hybrid_mode_queries_both_vectors():
    client = _RecordingClient([SimpleNamespace(score=0.03, payload={"title": "Lost"})])

    hits = retrieve("dark survival", mode="hybrid", client=client, embed=_fake_embed(), limit=5)

    call = client.calls[0]
    dense_branch, sparse_branch = call["prefetch"]
    assert dense_branch.using == DENSE_VECTOR
    assert dense_branch.query == [0.5, 0.5, 0.5]
    assert sparse_branch.using == SPARSE_VECTOR
    assert hits == [{"score": 0.03, "title": "Lost"}]


def test_hybrid_sends_the_query_words_to_the_sparse_side():
    # The whole point of the keyword half. If the text stopped arriving, the sparse vector
    # would be empty, RRF would rank on the dense list alone, and nothing would fail.
    from vibewatch.bm25 import token_id

    client = _RecordingClient([])
    retrieve("inception dreams", mode="hybrid", client=client, embed=_fake_embed())

    _, sparse_branch = client.calls[0]["prefetch"]
    assert set(sparse_branch.query.indices) == {token_id("inception"), token_id("dreams")}


def test_hybrid_prefetches_more_candidates_than_it_returns():
    # Fusion can only reorder what it is given: prefetching exactly `limit` per branch
    # would discard the consensus candidates fusion exists to surface.
    client = _RecordingClient([])
    retrieve("dark survival", mode="hybrid", client=client, embed=_fake_embed(), limit=5)

    for branch in client.calls[0]["prefetch"]:
        assert branch.limit > 5


def test_hybrid_applies_filters_to_both_branches():
    # A hard constraint enforced on only one branch would leak excluded titles back in
    # through the other -- the subtlest way to break a filter.
    client = _RecordingClient([])
    retrieve("dark survival", mode="hybrid", client=client, embed=_fake_embed(), media_type="movie")

    for branch in client.calls[0]["prefetch"]:
        assert branch.filter is not None
        assert {cond.key for cond in branch.filter.must} == {"media_type"}


def test_hybrid_without_filters_constrains_neither_branch():
    client = _RecordingClient([])
    retrieve("dark survival", mode="hybrid", client=client, embed=_fake_embed())

    for branch in client.calls[0]["prefetch"]:
        assert branch.filter is None
