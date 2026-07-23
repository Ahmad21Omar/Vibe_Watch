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
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

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


def _build_filter(
    *,
    media_type: str | None = None,
    genres: list[str] | None = None,
    release_year_min: int | None = None,
    release_year_max: int | None = None,
    original_language: str | None = None,
) -> Filter | None:
    """Translate simple keyword args into a Qdrant filter -- or None if nothing was asked.

    Qdrant applies the filter BEFORE the vector search, using the payload indexes we built
    in create_collection(). So "sci-fi movies since 2015, ranked by mood" stays fast: it
    first narrows to the points that match the hard constraints, then compares vectors only
    among those. Filtering after the search would instead throw away good matches and leave
    fewer than `limit` results.
    """
    conditions: list[FieldCondition] = []

    if media_type is not None:
        conditions.append(FieldCondition(key="media_type", match=MatchValue(value=media_type)))
    if original_language is not None:
        conditions.append(
            FieldCondition(key="original_language", match=MatchValue(value=original_language))
        )
    if genres:
        # MatchAny = keep a title if its genre list overlaps the requested genres (OR within
        # genres): asking for ["Drama", "Thriller"] matches a title tagged with either.
        conditions.append(FieldCondition(key="genres", match=MatchAny(any=list(genres))))
    if release_year_min is not None or release_year_max is not None:
        # gte/lte are inclusive; passing only one bound leaves the other side open.
        conditions.append(
            FieldCondition(
                key="release_year", range=Range(gte=release_year_min, lte=release_year_max)
            )
        )

    if not conditions:
        return None
    # `must` = AND across the different fields (media_type AND year AND ...).
    return Filter(must=conditions)


def search(
    client: QdrantClient,
    query_vector: list[float],
    limit: int = 5,
    *,
    media_type: str | None = None,
    genres: list[str] | None = None,
    release_year_min: int | None = None,
    release_year_max: int | None = None,
    original_language: str | None = None,
) -> list[dict]:
    """Find the `limit` nearest titles to a query vector, optionally within hard filters.

    The vector part answers "what feels like this?"; the optional filters answer "and only
    among movies / this genre / since this year". Keyword-only so a call reads self-
    documentingly: `search(client, vec, media_type="movie", release_year_min=2015)`.
    """
    query_filter = _build_filter(
        media_type=media_type,
        genres=genres,
        release_year_min=release_year_min,
        release_year_max=release_year_max,
        original_language=original_language,
    )
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )
    return [{"score": point.score, **point.payload} for point in response.points]
