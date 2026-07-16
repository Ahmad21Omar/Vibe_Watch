"""Tests for the on-disk embedding cache.

The cache is what makes an expensive embedding run resumable, so its guarantees are
worth pinning down: keys are content-addressed, vectors survive a restart, and a save
is atomic (a crash cannot corrupt the file).
"""

from vibewatch.embedding_cache import EmbeddingCache, text_hash

MODEL = "gemini-embedding-001"


def test_text_hash_is_deterministic():
    assert text_hash(MODEL, "RETRIEVAL_DOCUMENT", "hello") == text_hash(
        MODEL, "RETRIEVAL_DOCUMENT", "hello"
    )


def test_text_hash_depends_on_text_model_and_task_type():
    base = text_hash(MODEL, "RETRIEVAL_DOCUMENT", "hello")
    assert base != text_hash(MODEL, "RETRIEVAL_DOCUMENT", "world")  # text
    assert base != text_hash(MODEL, "RETRIEVAL_QUERY", "hello")  # task_type
    assert base != text_hash("other-model", "RETRIEVAL_DOCUMENT", "hello")  # model


def test_hash_has_no_boundary_collision():
    # The \x00 separator must stop "a"+"bc" from hashing the same as "ab"+"c".
    assert text_hash(MODEL, "RETRIEVAL_DOCUMENT", "abc") != text_hash(
        MODEL, "RETRIEVAL_DOCUMENT\x00a", "bc"
    )


def test_put_then_get_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.json")
    cache.put("k", [0.1, 0.2, 0.3])
    assert cache.get("k") == [0.1, 0.2, 0.3]
    assert cache.get("missing") is None


def test_survives_a_restart(tmp_path):
    # The whole point: write, "restart" (new instance), and the data is still there.
    path = tmp_path / "cache.json"
    first = EmbeddingCache(path)
    first.put("k", [1.0, 2.0])
    first.save()

    reloaded = EmbeddingCache(path)
    assert reloaded.get("k") == [1.0, 2.0]
    assert len(reloaded) == 1


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "cache.json"
    cache = EmbeddingCache(path)
    cache.put("k", [1.0])
    cache.save()

    assert path.exists()
    # The temporary file used during the atomic write must be gone afterwards.
    assert not (tmp_path / "cache.json.tmp").exists()


def test_missing_file_starts_empty(tmp_path):
    assert len(EmbeddingCache(tmp_path / "does_not_exist.json")) == 0
