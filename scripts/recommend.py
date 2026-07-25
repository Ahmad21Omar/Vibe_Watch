"""Ask Vibewatch for a recommendation from the command line.

Run:  python -m scripts.recommend "survival, dark, hopeless"
      python -m scripts.recommend "something funny" --type movie --since 2015
      python -m scripts.recommend "epic space opera" --genre "Science Fiction" --limit 8

This is the whole RAG pipeline in one command -- the online counterpart to the offline
ingestion scripts. It needs a running Qdrant with the `titles` collection populated
(`docker compose up -d` + `python -m scripts.index_titles`) and a Gemini API key.

Besides being convenient, printing the retrieved titles NEXT TO the generated answer is a
debugging tool: when a recommendation looks wrong, you can see immediately whether the
retrieval was bad or the generation ignored good context.
"""

import argparse

from vibewatch.graph import recommend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get a mood-based movie/TV recommendation.")
    parser.add_argument("query", help="what you are in the mood for, in plain English")
    parser.add_argument("--type", choices=["movie", "tv"], help="restrict to movies or TV")
    parser.add_argument(
        "--genre", action="append", help="restrict to a genre (repeatable, matches ANY)"
    )
    parser.add_argument("--since", type=int, help="earliest release year (inclusive)")
    parser.add_argument("--until", type=int, help="latest release year (inclusive)")
    parser.add_argument("--limit", type=int, default=5, help="how many titles to retrieve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Only pass filters the user actually set -- a None would be a filter on None.
    filters = {
        key: value
        for key, value in {
            "media_type": args.type,
            "genres": args.genre,
            "release_year_min": args.since,
            "release_year_max": args.until,
        }.items()
        if value is not None
    }

    state = recommend(args.query, limit=args.limit, **filters)

    if state.get("relaxed"):
        print("(No title matched your filters, so they were dropped for this search.)\n")

    print(state["answer"])

    print("\n--- retrieved ---")
    for hit in state["hits"]:
        year = hit.get("release_year") or "----"
        print(f"  {hit['score']:.4f}  {hit['title']} ({year}) [{hit['media_type']}]")


if __name__ == "__main__":
    main()
