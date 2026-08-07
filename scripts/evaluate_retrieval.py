"""Measure retrieval quality against the hand-labelled set in eval/gold_queries.json.

Run:  python -m scripts.evaluate_retrieval
      python -m scripts.evaluate_retrieval --k 10

Needs a running Qdrant with the index populated, and one Gemini embedding per query
(12 queries = 12 of the 1000 daily embeddings -- cheap enough to run often).

Why this exists: every test so far proves the pipeline is CORRECT. None of them proves it
is GOOD. This script produces two numbers -- recall@k and MRR -- that can be compared
before and after a change to `embedding_text()`, the embedding model, or the ranking.
A number you can move is worth more than an opinion about whether results "feel better".

The absolute values are not the point (the labels only mark titles a human is sure about,
while many unlabelled titles are also fine answers). The DELTA between two runs is.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from vibewatch.evaluation import evaluate, recall_at_k, reciprocal_rank
from vibewatch.retrieval import retrieve
from vibewatch.vector_store import get_client, indexed_titles

GOLD_PATH = Path("eval/gold_queries.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against gold labels.")
    parser.add_argument("--k", type=int, default=5, help="how many titles to retrieve")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense"],
        default="hybrid",
        help="hybrid = semantic + BM25 fused with RRF; dense = semantic only (the baseline)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print the retrieved titles per query"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["queries"]

    # Check the labels against the catalogue BEFORE spending any quota: a label that is
    # not in the index can never be retrieved, so it would depress recall forever and look
    # like a retrieval problem when it is really a labelling one.
    catalogue = indexed_titles(get_client())
    missing = sorted(
        {title for case in gold for title in case["relevant"]} - catalogue
    )
    if missing:
        print("WARNING: these gold labels are not in the index and can never be found:")
        for title in missing:
            print(f"  - {title}")
        print()

    print(
        f"Evaluating {len(gold)} queries at k={args.k}, mode={args.mode} "
        "(one embedding call each)...\n"
    )

    results: list[tuple[list[str], list[str]]] = []
    for case in gold:
        hits = retrieve(case["query"], limit=args.k, mode=args.mode)
        retrieved = [hit["title"] for hit in hits]
        relevant = case["relevant"]
        results.append((retrieved, relevant))

        recall = recall_at_k(retrieved, relevant, args.k)
        rank = reciprocal_rank(retrieved, relevant)
        # Per-query lines make a bad average diagnosable: one broken query looks very
        # different from uniformly mediocre retrieval, and the mean hides which it is.
        print(f"  recall {recall:.2f}  rr {rank:.2f}  {case['query'][:52]}")
        if args.verbose:
            for title in retrieved:
                mark = "*" if title in relevant else " "
                print(f"        {mark} {title}")

    report = evaluate(results, k=args.k)

    # Recall@k cannot reach 1.0 when a query has more than k labels -- with 6 relevant
    # titles and k=5 the best possible score is 5/6. Printing that ceiling stops the
    # headline number from being read as "17% of results are bad" when part of the gap
    # is pure arithmetic.
    ceiling = mean(
        min(args.k, len(case["relevant"])) / len(case["relevant"]) for case in gold
    )

    print(f"\n  mode         {args.mode}")
    print(f"  queries      {report['queries']}")
    print(f"  recall@{args.k}     {report[f'recall@{args.k}']:.3f}  (max possible {ceiling:.3f})")
    print(f"  MRR          {report['mrr']:.3f}")


if __name__ == "__main__":
    main()
