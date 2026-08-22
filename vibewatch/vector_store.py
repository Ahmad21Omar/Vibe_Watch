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
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PointStruct,
    Prefetch,
    Range,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from vibewatch.bm25 import query_vector as bm25_query_vector
from vibewatch.config import settings
from vibewatch.embeddings import VECTOR_SIZE
from vibewatch.models import Title

COLLECTION_NAME = "titles"

# Each point carries TWO vectors, so they need names. Dense = meaning (Gemini embedding),
# sparse = words (BM25). Naming them is what lets one query ask both and fuse the answers.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# How many points we send to Qdrant per request.
UPSERT_BATCH_SIZE = 100

# Upper bound for the genre facet. TMDb defines roughly 20 genres, so this is generous
# on purpose -- silently truncating the list would hide filter options from the user.
MAX_FACET_VALUES = 100

# Page size when walking the whole collection.
SCROLL_BATCH_SIZE = 256


def get_client() -> QdrantClient:
    """Connect to Qdrant -- the local container, or a managed cluster if a key is set.

    The api_key is passed only when configured. Handing `api_key=""` to a local instance
    would make the client send an empty Authorization header, which some setups reject;
    and requiring a key would break every local run and the whole test suite.
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


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
        # Named configs, one per vector kind. The dense side keeps cosine distance; the
        # sparse side needs no metric at all -- sparse scoring IS the dot product, which
        # is exactly what makes our BM25 weights come out as BM25 scores.
        vectors_config={DENSE_VECTOR: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(index=SparseIndexParams())},
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


def index_titles(
    client: QdrantClient,
    titles: list[Title],
    vectors: list[list[float]],
    sparse_vectors: list[dict[int, float]],
) -> None:
    """Write titles plus both of their vectors into Qdrant."""
    points = [
        PointStruct(
            id=point_id(title),
            vector={
                DENSE_VECTOR: vector,
                # Qdrant wants the sparse vector split into parallel index/value lists.
                SPARSE_VECTOR: SparseVector(
                    indices=list(sparse.keys()), values=list(sparse.values())
                ),
            },
            # The payload travels with the vector. We store everything we need to
            # filter on AND everything we want to show the user, so a search hit is
            # self-contained -- no second lookup in another database.
            payload=title.model_dump(),
        )
        # zip() walks the lists in lockstep: (title_0, vector_0, sparse_0), ...
        for title, vector, sparse in zip(titles, vectors, sparse_vectors, strict=True)
    ]

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + UPSERT_BATCH_SIZE],
        )
        print(f"  upserted {min(start + UPSERT_BATCH_SIZE, len(points))}/{len(points)}")


def available_genres(client: QdrantClient) -> list[str]:
    """The genres that actually occur in the index, alphabetically.

    A UI must offer the genres the catalogue really contains -- a hardcoded list would
    drift the moment TMDb adds one, and offering a genre that matches nothing sends the
    user straight into an empty result.

    `facet` aggregates on the server using the payload index we built for `genres`: one
    small request instead of scrolling all 900 payloads across the wire just to collect
    distinct values.
    """
    response = client.facet(
        collection_name=COLLECTION_NAME, key="genres", limit=MAX_FACET_VALUES
    )
    return sorted(hit.value for hit in response.hits)


def indexed_titles(client: QdrantClient) -> set[str]:
    """Every title name currently in the index.

    Used by the evaluation to spot gold labels that are not in the catalogue at all: such
    a label can never be retrieved, so it would depress the score forever and look like a
    retrieval problem when it is really a DATA problem.

    Scrolls the payloads (no vectors, one field) rather than embedding anything -- the
    obvious alternative, searching for each title by name, would burn the daily embedding
    quota just to answer a question about metadata.
    """
    titles: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_BATCH_SIZE,
            with_payload=["title"],
            with_vectors=False,
            offset=offset,
        )
        titles.update(point.payload["title"] for point in points)
        if offset is None:  # Qdrant returns None once the last page is done
            return titles


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
        using=DENSE_VECTOR,  # required now that a point carries more than one vector
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )
    return [{"score": point.score, **point.payload} for point in response.points]


# How many candidates each half retrieves before fusion. Fusion can only reorder what it
# is given, so this must be comfortably larger than `limit` -- otherwise a title ranked
# 6th by both halves (a strong consensus candidate) never reaches the fusion step at all.
PREFETCH_MULTIPLIER = 4
MIN_PREFETCH = 20


def hybrid_search(
    client: QdrantClient,
    query_vector: list[float],
    query_text: str,
    limit: int = 5,
    **filters,
) -> list[dict]:
    """Search semantically AND lexically, then fuse the two rankings with RRF.

    Dense finds what the query *means*, sparse finds what it *says*. Fusing them fixes the
    blind spot each one has alone: a query naming a specific title ("something like
    Inception") is a lexical question that a 3072-dimensional average of meaning answers
    only vaguely, while a pure mood ("dark, hopeless") has no keywords to match at all.

    RECIPROCAL RANK FUSION, and why it is the right fuser here:
    each result contributes 1/(k + rank) from each list it appears in. It uses only
    POSITIONS, never the raw scores -- which matters because our two scores are not
    comparable in any way: cosine similarity lives in [-1, 1] and clusters tightly, BM25
    is unbounded and depends on corpus statistics. Any weighted sum of the two would need
    a normalisation constant that is really just a hyperparameter in disguise. RRF needs
    none, and a title that both halves rank highly wins on agreement.

    The filters apply to BOTH branches: a hard constraint that only narrowed one half
    would leak excluded titles back in through the other.
    """
    query_filter = _build_filter(**filters)
    prefetch_limit = max(limit * PREFETCH_MULTIPLIER, MIN_PREFETCH)
    sparse = bm25_query_vector(query_text)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=query_vector,
                using=DENSE_VECTOR,
                limit=prefetch_limit,
                filter=query_filter,
            ),
            Prefetch(
                query=SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
                using=SPARSE_VECTOR,
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    # NOTE: `score` here is the RRF score (small, e.g. ~0.03), not cosine similarity.
    # It ranks correctly but is not a similarity -- do not present it as one.
    return [{"score": point.score, **point.payload} for point in response.points]
