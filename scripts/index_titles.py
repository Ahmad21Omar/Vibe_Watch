"""Indexing: embed the cached titles and load them into Qdrant.

Run:  python -m scripts.index_titles   (after scripts/fetch_titles.py)

This is the second half of the offline ingestion. Afterwards Qdrant holds one point
per title -- vector + metadata -- and the app can answer queries without touching
TMDb or re-embedding anything.
"""

import json
from pathlib import Path

from vibewatch.embeddings import embed_documents
from vibewatch.models import Title
from vibewatch.vector_store import create_collection, get_client, index_titles

INPUT_PATH = Path("data/titles.json")


def main() -> None:
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    # Pydantic re-validates every record on the way in. If fetch_titles.py ever wrote
    # a broken row, we find out here -- not during a user query.
    titles = [Title(**item) for item in raw]
    print(f"Loaded {len(titles)} titles from {INPUT_PATH}")

    print("Embedding (this calls the Gemini API)...")
    vectors = embed_documents([title.embedding_text() for title in titles])

    client = get_client()
    print(f"\nCreating collection (vector size {len(vectors[0])}, cosine distance)...")
    create_collection(client)

    print("Uploading to Qdrant...")
    index_titles(client, titles, vectors)

    count = client.count(collection_name="titles").count
    print(f"\nDone. Qdrant now holds {count} points.")


if __name__ == "__main__":
    main()
