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
from typing import Literal

from qdrant_client import QdrantClient

from vibewatch.embeddings import embed_query
from vibewatch.vector_store import get_client, hybrid_search, search

# "dense" is semantic only; "hybrid" fuses it with BM25 keyword search via RRF.
#
# DEFAULT IS DENSE, AND THAT IS A MEASURED RESULT, NOT AN OVERSIGHT.
# Hybrid search is the standard answer to "how do I improve retrieval", so it was built,
# tested, and then evaluated against the gold set on an identical index:
#
#     dense   recall@5 0.832   MRR 0.917
#     hybrid  recall@5 0.557   MRR 0.715      <- clearly worse
#
# Two reasons, both specific to this project:
#
# 1. Our queries are moods ("epic fantasy quest with swords and magic"), which contain no
#    rare terms for BM25 to latch onto. It matches generic words instead -- "story" pulls
#    up *Crazy Story* -- and RRF fuses by POSITION, so the top of a useless keyword list
#    carries the same weight as the top of a good semantic one and displaces it.
# 2. The lexical case BM25 is supposed to rescue is already covered: `embedding_text()`
#    embeds the TITLE along with the plot, so the dense vectors handle proper nouns like
#    "Frieren" or "Tarantino" on their own (verified query by query).
#
# The hybrid path stays in the codebase because the comparison is the point -- and because
# a larger corpus, or plot-only embeddings, would change the answer. Run it yourself:
# `python -m scripts.evaluate_retrieval --mode hybrid`.
Mode = Literal["hybrid", "dense"]


def retrieve(
    query: str,
    *,
    limit: int = 5,
    mode: Mode = "dense",
    client: QdrantClient | None = None,
    embed: Callable[[str], list[float]] = embed_query,
    **filters,
) -> list[dict]:
    """Embed `query` and return the `limit` best-matching titles as flat dicts.

    `**filters` is forwarded verbatim to the search -- `media_type`, `genres`,
    `release_year_min` / `release_year_max`, `original_language` -- so a caller writes
    `retrieve("dark survival", media_type="movie", release_year_min=2015)` and never
    touches the vector layer.
    """
    client = client or get_client()
    query_vector = embed(query)

    if mode == "dense":
        return search(client, query_vector, limit=limit, **filters)
    return hybrid_search(client, query_vector, query, limit=limit, **filters)
