"""Qdrant vector store: create the collection, index titles, search.

A "collection" in Qdrant is roughly a table. Each row is a "point" and consists of:
- an id
- a vector      (the 3072 numbers -> used for similarity search)
- a payload     (the metadata -> used for hard filters and for showing results)

Design decision -- distance metric: COSINE.
Cosine similarity measures the ANGLE between two vectors, ignoring their length.
That is what we want: a long plot summary and a three-word query should be
comparable by their *direction* (meaning), not by how much text there was.
Euclidean distance would let text length distort the result.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vibewatch.config import settings
from vibewatch.embeddings import VECTOR_SIZE
from vibewatch.models import Title

COLLECTION_NAME = "titles"

# How many points we send to Qdrant per request.
UPSERT_BATCH_SIZE = 100


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def point_id(title: Title) -> str:
    """A stable, unique id for a title.

    We cannot use tmdb_id alone: a movie and a TV show may share the same number.
    We also do not want a running counter, because re-running the indexing would
    then assign different ids and create duplicates.

    uuid5 derives a UUID deterministically from a string: same input -> same id,
    always. That makes indexing IDEMPOTENT -- running it twice updates the existing
    points instead of duplicating them.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vibewatch:{title.media_type}:{title.tmdb_id}"))


def create_collection(client: QdrantClient) -> None:
    """Create the collection from scratch (dropping it if it already exists)."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Payload indexes make metadata filters fast. Without them Qdrant would have to
    # scan every point's payload; with them it can narrow the candidates first.
    # These are exactly the fields we want to filter on in step 4.
    for field, schema in [
        ("media_type", "keyword"),
        ("genres", "keyword"),
        ("original_language", "keyword"),
        ("release_year", "integer"),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )


def index_titles(client: QdrantClient, titles: list[Title], vectors: list[list[float]]) -> None:
    """Write titles + their vectors into Qdrant."""
    points = [
        PointStruct(
            id=point_id(title),
            vector=vector,
            # The payload travels with the vector. We store everything we need to
            # filter on AND everything we want to show the user, so a search hit is
            # self-contained -- no second lookup in another database.
            payload=title.model_dump(),
        )
        # zip() walks both lists in lockstep: (title_0, vector_0), (title_1, vector_1)...
        for title, vector in zip(titles, vectors, strict=True)
    ]

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + UPSERT_BATCH_SIZE],
        )
        print(f"  upserted {min(start + UPSERT_BATCH_SIZE, len(points))}/{len(points)}")


def search(client: QdrantClient, query_vector: list[float], limit: int = 5) -> list[dict]:
    """Find the `limit` nearest titles to a query vector.

    Step 4 will extend this with metadata filters.
    """
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return [{"score": point.score, **point.payload} for point in response.points]
