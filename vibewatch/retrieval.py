"""The retrieval seam: turn a natural-language query into ranked, grounded titles.

This is the ONE function the generation step (Step 5) will call. It hides the two moving
parts behind a single name: embed the user's words as a RETRIEVAL_QUERY, then search the
Qdrant index -- optionally narrowed by hard metadata filters. Keeping this seam small means
the LangGraph flow in Step 5 never has to know Gemini or Qdrant exist; it just asks for
titles that match a mood.

`client` and `embed` are injectable (the same pattern embeddings.py uses for `embed_batch`)
so this can be unit-tested with fakes -- no Docker, no API, no quota.
"""

from collections.abc import Callable

from qdrant_client import QdrantClient

from vibewatch.embeddings import embed_query
from vibewatch.vector_store import get_client, search


def retrieve(
    query: str,
    *,
    limit: int = 5,
    client: QdrantClient | None = None,
    embed: Callable[[str], list[float]] = embed_query,
    **filters,
) -> list[dict]:
    """Embed `query` and return the `limit` best-matching titles as flat dicts.

    `**filters` is forwarded verbatim to search() -- `media_type`, `genres`,
    `release_year_min` / `release_year_max`, `original_language` -- so a caller writes
    `retrieve("dark survival", media_type="movie", release_year_min=2015)` and never
    touches the vector layer.
    """
    client = client or get_client()
    query_vector = embed(query)
    return search(client, query_vector, limit=limit, **filters)
