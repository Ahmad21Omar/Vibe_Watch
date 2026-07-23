"""Unit tests for retrieve() -- the query -> hits seam, without real services.

retrieve() has one job: embed the query, then hand the vector (plus any filters) to
search(). We verify that wiring with a fake embed function and a recording fake client, so
the test proves the seam is connected correctly without ever hitting Gemini or Qdrant.
"""

from types import SimpleNamespace

from vibewatch.retrieval import retrieve


class _RecordingClient:
    def __init__(self, points):
        self._points = points
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self._points)


def test_retrieve_embeds_query_and_returns_mapped_hits():
    client = _RecordingClient([SimpleNamespace(score=0.9, payload={"title": "Lost"})])
    seen = {}

    def fake_embed(text: str) -> list[float]:
        seen["text"] = text
        return [0.5, 0.5, 0.5]

    hits = retrieve("dark survival", client=client, embed=fake_embed, limit=3)

    # The user's words reached the embedder unchanged...
    assert seen["text"] == "dark survival"
    # ...its vector reached Qdrant unchanged, with the requested limit...
    assert client.calls[0]["query"] == [0.5, 0.5, 0.5]
    assert client.calls[0]["limit"] == 3
    # ...and the hit came back flattened, exactly as search() produced it.
    assert hits == [{"score": 0.9, "title": "Lost"}]


def test_retrieve_forwards_filters_to_search():
    client = _RecordingClient([])

    retrieve(
        "dark survival",
        client=client,
        embed=lambda _text: [0.0, 0.0, 0.0],
        media_type="tv",
        genres=["Drama"],
    )

    query_filter = client.calls[0]["query_filter"]
    assert query_filter is not None
    assert {cond.key for cond in query_filter.must} == {"media_type", "genres"}


def test_retrieve_without_filters_searches_everything():
    client = _RecordingClient([])
    retrieve("dark survival", client=client, embed=lambda _text: [0.0])
    assert client.calls[0]["query_filter"] is None
