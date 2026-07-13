"""Ingestion: fetch movies & TV shows from TMDb and cache them as JSON.

Run:  python -m scripts.fetch_titles

This runs OFFLINE and only once (or whenever we want to refresh the catalogue).
Separating ingestion from the live query path is the standard RAG architecture:
no API limits at query time, and reproducible data to build embeddings from.
"""

import json
from pathlib import Path

from vibewatch.models import Title
from vibewatch.tmdb import RESULTS_PER_PAGE, TMDbClient, to_title

# 25 pages x 20 results = 500 titles per media type -> ~1000 in total.
# Enough for a meaningful demo, small enough to embed quickly.
PAGES_PER_TYPE = 25

OUTPUT_PATH = Path("data/titles.json")


def fetch_media_type(client: TMDbClient, media_type: str) -> list[Title]:
    """Fetch all pages for one media type ("movie" or "tv")."""
    genre_names = client.genre_map(media_type)
    titles: list[Title] = []

    for page in range(1, PAGES_PER_TYPE + 1):
        for raw in client.discover(media_type, page=page):
            title = to_title(raw, media_type=media_type, genre_names=genre_names)
            # Data quality: a title without a plot has nothing to embed, so it could
            # never be retrieved meaningfully. We drop it.
            if not title.overview.strip():
                continue
            titles.append(title)

        print(f"  {media_type}: page {page}/{PAGES_PER_TYPE} -> {len(titles)} usable titles")

    return titles


def main() -> None:
    client = TMDbClient()
    try:
        all_titles: list[Title] = []
        for media_type in ("movie", "tv"):
            print(f"Fetching {media_type}s ({PAGES_PER_TYPE} pages x {RESULTS_PER_PAGE})...")
            all_titles.extend(fetch_media_type(client, media_type))
    finally:
        client.close()

    # The same title can appear on two pages while popularity shifts during paging.
    unique = {(t.media_type, t.tmdb_id): t for t in all_titles}
    titles = list(unique.values())
    duplicates = len(all_titles) - len(titles)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps([t.model_dump() for t in titles], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    movies = sum(1 for t in titles if t.media_type == "movie")
    shows = len(titles) - movies
    print(f"\nSaved {len(titles)} titles ({movies} movies, {shows} TV shows) -> {OUTPUT_PATH}")
    print(f"Removed {duplicates} duplicates.")


if __name__ == "__main__":
    main()
