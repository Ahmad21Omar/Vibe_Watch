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

from google import genai
from google.genai import errors, types

from vibewatch.config import settings

MODEL = "gemini-embedding-001"

# The vector length this model produces. Qdrant needs to know it up front.
VECTOR_SIZE = 3072

# Free-tier quota is 100 embedded texts per minute. We aim for 90 instead of 100:
# running exactly at the limit reliably produces occasional 429s (clock skew, the
# server counts slightly differently than we do). A safety margin is cheaper than
# a retry storm.
EMBEDDINGS_PER_MINUTE = 90

BATCH_SIZE = 50
MAX_RETRIES = 6

# Created once at import time and reused -- the client holds a connection pool.
_client = genai.Client(api_key=settings.gemini_api_key)


def _server_retry_delay(error: errors.APIError) -> float | None:
    """Read the retry delay the API sends us, e.g. {'retryDelay': '42s'}.

    Guessing a backoff is a fallback; if the server tells us how long to wait,
    that is always the better number. The field is nested and optional, so we
    dig for it defensively and return None if it is not there.
    """
    details = getattr(error, "details", None) or {}
    for detail in details.get("error", {}).get("details", []):
        delay = detail.get("retryDelay")
        if delay:
            return float(delay.rstrip("s"))
    return None


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed one batch, retrying on transient errors (429 rate limit, 5xx)."""
    last_error: errors.APIError | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = _client.models.embed_content(
                model=MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [embedding.values for embedding in response.embeddings]

        except errors.APIError as error:
            # 429 = rate limit, 5xx = server hiccup. Both are worth retrying.
            # Anything else (401 bad key, 400 bad request) is our fault: retrying
            # would not help, so fail immediately and loudly.
            if error.code != 429 and error.code < 500:
                raise

            last_error = error
            # Prefer the server's own number; fall back to exponential backoff.
            wait = _server_retry_delay(error) or 2**attempt
            print(f"    API error {error.code}, waiting {wait:.0f}s...")
            time.sleep(wait + 1)  # +1s safety margin

    raise last_error  # all retries exhausted


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed many documents (the movies/TV shows we want to index).

    Throttled to stay inside the free-tier quota. Embedding N texts costs N units of
    quota, so after each batch we wait long enough that our average rate stays below
    the limit -- much better than hammering the API and handling the inevitable 429.
    """
    vectors: list[list[float]] = []
    seconds_per_batch = BATCH_SIZE / EMBEDDINGS_PER_MINUTE * 60  # 50/100*60 = 30s

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(_embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
        print(f"  embedded {len(vectors)}/{len(texts)}")

        # Pace ourselves -- but do not sleep after the final batch.
        if start + BATCH_SIZE < len(texts):
            time.sleep(seconds_per_batch)

    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single user query. Note the different task type."""
    return _embed_batch([text], task_type="RETRIEVAL_QUERY")[0]
