"""Tests for the resumable embedding logic (_embed_with_cache).

We inject a fake embed function, so these tests exercise the caching/resume/order
behaviour end-to-end WITHOUT calling the real API. This is exactly why the real embed
function is passed in as a parameter -- dependency injection makes it testable.
"""

from vibewatch import embeddings
from vibewatch.embedding_cache import EmbeddingCache

TASK = "RETRIEVAL_DOCUMENT"


def make_fake_batch(calls: list[str]):
    """A stand-in for the API: records what it was asked to embed and returns a
    trivial 1-D 'vector' derived from the text, so results are checkable."""

    def fake_batch(texts, task_type):
        calls.extend(texts)
        return [[float(len(t))] for t in texts]

    return fake_batch


def _run(texts, cache, calls):
    return embeddings._embed_with_cache(
        texts, task_type=TASK, cache=cache,
        embed_batch=make_fake_batch(calls), batch_size=10, pause_seconds=0.0,
    )


def test_returns_vectors_in_input_order(tmp_path):
    calls: list[str] = []
    out = _run(["a", "bbb", "cc"], EmbeddingCache(tmp_path / "c.json"), calls)
    assert out == [[1.0], [3.0], [2.0]]


def test_second_run_uses_cache_and_calls_nothing(tmp_path):
    path = tmp_path / "c.json"
    first_calls: list[str] = []
    _run(["a", "bb"], EmbeddingCache(path), first_calls)
    assert first_calls == ["a", "bb"]  # first run embeds everything

    # Simulate a restart: a fresh cache loaded from the same file.
    second_calls: list[str] = []
    out = _run(["a", "bb"], EmbeddingCache(path), second_calls)
    assert out == [[1.0], [2.0]]
    assert second_calls == []  # nothing re-embedded -> fully resumed from disk


def test_only_missing_texts_are_embedded(tmp_path):
    path = tmp_path / "c.json"
    _run(["a"], EmbeddingCache(path), [])  # cache now holds "a"

    calls: list[str] = []
    _run(["a", "new"], EmbeddingCache(path), calls)
    assert calls == ["new"]  # "a" comes from cache, only "new" hits the API


def test_duplicate_texts_are_embedded_once(tmp_path):
    calls: list[str] = []
    out = _run(["x", "x", "x"], EmbeddingCache(tmp_path / "c.json"), calls)
    assert calls == ["x"]  # deduplicated
    assert out == [[1.0], [1.0], [1.0]]  # but every position still gets its vector
