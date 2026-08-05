"""Turn text into vectors using the Gemini embedding API.

Three things in here matter a lot and are easy to get wrong:

1. TASK TYPE (asymmetric embeddings)
   A search query ("survival, dark") and a document (a 40-word plot) are very
   different kinds of text. The model is trained to project both into one shared
   space -- but only if we tell it which role the text plays. So documents are
   embedded as RETRIEVAL_DOCUMENT and queries as RETRIEVAL_QUERY. Using the same
   type for both measurably degrades results.

2. BATCHING
   One API call per title would mean ~900 round trips. The API accepts a list of
   texts and returns a list of vectors, so we send them in batches instead.

3. RATE LIMITING
   The free tier allows 100 EMBEDDINGS per minute -- counted per text, not per
   API call. Batching therefore saves round trips but not quota. We do two things:
   - throttle proactively, so we stay under the limit instead of crashing into it;
   - if we still get a 429, use the retry delay the SERVER tells us, rather than
     guessing with a blind backoff.
"""

import time
from pathlib import Path

from google import genai
from google.genai import types

from vibewatch.config import settings
from vibewatch.embedding_cache import EmbeddingCache, text_hash
from vibewatch.gemini import call_with_retry

MODEL = "gemini-embedding-001"

# Where computed document vectors are checkpointed so a run is resumable.
CACHE_PATH = Path("data/embedding_cache.json")

# The vector length this model produces. Qdrant needs to know it up front.
VECTOR_SIZE = 3072

# Free-tier quota is 100 embedded texts per minute. We aim for 90 instead of 100:
# running exactly at the limit reliably produces occasional 429s (clock skew, the
# server counts slightly differently than we do). A safety margin is cheaper than
# a retry storm.
EMBEDDINGS_PER_MINUTE = 90

BATCH_SIZE = 50

# Created on first use and then reused -- the client holds a connection pool. Lazy rather
# than at import time so that importing this module never needs an API key: the pure unit
# tests import it constantly and must run on a fresh clone without any secrets.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.require("gemini_api_key"))
    return _client


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed one batch, retrying on transient errors (429 rate limit, 5xx).

    The retry policy itself lives in `vibewatch.gemini` -- the generation step needs
    exactly the same behaviour, and one shared implementation cannot drift.
    """

    def request() -> list[list[float]]:
        response = _get_client().models.embed_content(
            model=MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [embedding.values for embedding in response.embeddings]

    return call_with_retry(request)


def _embed_with_cache(
    texts: list[str],
    task_type: str,
    cache: EmbeddingCache,
    embed_batch,
    batch_size: int = BATCH_SIZE,
    pause_seconds: float = 0.0,
) -> list[list[float]]:
    """Embed `texts`, but only the ones not already in `cache`.

    `embed_batch` is passed in rather than hard-wired so this logic can be tested with
    a stub -- no real API call needed. After each batch we persist the cache, so a
    crash loses at most one batch instead of the whole run (that is the resumability
    lesson made concrete).
    """
    # Which texts still need embedding? Deduplicate by cache key so a text repeated in
    # the input is embedded only once; keep first-seen order for readable progress.
    missing: list[str] = []
    seen: set[str] = set()
    for text in texts:
        key = text_hash(MODEL, task_type, text)
        if key not in cache and key not in seen:
            seen.add(key)
            missing.append(text)

    if missing:
        print(f"  {len(missing)} new to embed, {len(texts) - len(missing)} served from cache")

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = embed_batch(batch, task_type=task_type)
        for text, vector in zip(batch, vectors, strict=True):
            cache.put(text_hash(MODEL, task_type, text), vector)
        cache.save()  # checkpoint after every batch
        print(f"  embedded {min(start + batch_size, len(missing))}/{len(missing)} (new)")

        # Pace ourselves to stay under the quota -- but not after the final batch.
        if start + batch_size < len(missing):
            time.sleep(pause_seconds)

    # Return a vector for every requested text, in the original order.
    return [cache.get(text_hash(MODEL, task_type, text)) for text in texts]


def embed_documents(texts: list[str], cache_path: Path = CACHE_PATH) -> list[list[float]]:
    """Embed many documents (the movies/TV shows we want to index).

    Resumable: vectors are cached on disk, so re-running only embeds what is new.
    Throttled: embedding N texts costs N units of quota, so we pace ourselves to stay
    under the per-minute limit instead of hammering the API and handling 429s.
    """
    cache = EmbeddingCache(cache_path)
    seconds_per_batch = BATCH_SIZE / EMBEDDINGS_PER_MINUTE * 60
    return _embed_with_cache(
        texts,
        task_type="RETRIEVAL_DOCUMENT",
        cache=cache,
        embed_batch=_embed_batch,
        pause_seconds=seconds_per_batch,
    )


def embed_query(text: str) -> list[float]:
    """Embed a single user query. Note the different task type."""
    return _embed_batch([text], task_type="RETRIEVAL_QUERY")[0]
