"""Retrieval metrics: how good are the titles we retrieve, not just how correct is the code?

Everything the test suite has measured so far is *correctness* -- the filter is built, the
vector reaches Qdrant, the graph is wired. None of that says whether the results are any
GOOD. That question needs a labelled set of queries and numbers that can be compared
across changes to `embedding_text()`, the model, or the chunking.

Two metrics, because they answer different questions:

- **Recall@k** -- of the titles we consider relevant, how many made it into the top k?
  This is the one that matters for RAG: the generator can only recommend what retrieval
  handed it, so a title missing here is unrecoverable downstream.
- **MRR** (mean reciprocal rank) -- how high up is the FIRST relevant hit? 1.0 means it
  ranked first, 0.5 means second. Recall ignores position; MRR is where re-ranking work
  shows up.

Deliberately pure functions over lists of titles: no Qdrant, no API, instant to test. The
live part (embedding queries and searching) lives in `scripts/evaluate_retrieval.py`.
"""

from statistics import mean


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Share of the relevant titles that appear in the top `k` retrieved.

    Returns 0.0 for an empty label set rather than raising: a query nobody labelled
    contributes nothing, it does not crash the report.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return sum(title in top_k for title in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / (rank of the first relevant hit); 0.0 if none was retrieved at all.

    Rank is 1-based, so a relevant title in first place scores 1.0, second place 0.5.
    """
    relevant_set = set(relevant)
    for rank, title in enumerate(retrieved, start=1):
        if title in relevant_set:
            return 1 / rank
    return 0.0


def evaluate(results: list[tuple[list[str], list[str]]], k: int = 5) -> dict[str, float]:
    """Aggregate per-query (retrieved, relevant) pairs into one report.

    The mean over queries -- NOT over hits -- so every query counts equally regardless of
    how many titles were labelled for it. Otherwise a single heavily-labelled query would
    dominate the score and mask regressions everywhere else.
    """
    if not results:
        return {"queries": 0, f"recall@{k}": 0.0, "mrr": 0.0}

    return {
        "queries": len(results),
        f"recall@{k}": mean(recall_at_k(r, g, k) for r, g in results),
        "mrr": mean(reciprocal_rank(r, g) for r, g in results),
    }
