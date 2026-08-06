"""Measure whether the generated recommendations stay faithful to their sources.

Run:  python -m scripts.evaluate_generation            # 4 queries (default, cheap)
      python -m scripts.evaluate_generation --n 8      # more queries
      python -m scripts.evaluate_generation --json     # full verdicts, claim by claim

Needs a running Qdrant with the index populated, and a Gemini key.

COST, and why the default is small: each query costs one embedding, one generation call
AND one judging call. Retrieval evaluation is cheap enough to run constantly; this is not.
Four queries is enough to notice a regression in the prompt; run more before a release.

The queries are reused from eval/gold_queries.json -- the same set retrieval is scored on,
so a drop can be traced: bad answers on good retrieval mean the prompt regressed, while
both dropping at once points at the index or the embedding model.
"""

import argparse
import json
from pathlib import Path

from vibewatch.graph import recommend
from vibewatch.judge import as_json, format_report, score_faithfulness

GOLD_PATH = Path("eval/gold_queries.json")

# Default number of queries. Deliberately small -- see the cost note above.
DEFAULT_N = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge generated answers for faithfulness.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="how many queries to judge")
    parser.add_argument("--limit", type=int, default=5, help="titles retrieved per query")
    parser.add_argument("--json", action="store_true", help="print every claim and verdict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["queries"]
    queries = [case["query"] for case in gold][: args.n]

    print(f"Judging {len(queries)} answers (each costs 1 embedding + 2 LLM calls)...\n")

    results = []
    for query in queries:
        state = recommend(query, limit=args.limit)
        result = score_faithfulness(state["hits"], state["answer"])
        results.append(result)
        print(f"  {result['faithfulness']:.2f}  {query[:56]}")

    print()
    print(format_report(results))

    if args.json:
        print("\n--- full verdicts ---")
        print(as_json(results))


if __name__ == "__main__":
    main()
