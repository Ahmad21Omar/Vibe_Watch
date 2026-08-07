"""Tests for point_id(): the idempotency guarantee of our indexing.

The whole "re-running the indexer updates instead of duplicating" property rests on
this one function being deterministic AND collision-free between movies and TV shows.
If either breaks, we silently get duplicate or overwritten titles in Qdrant.
"""

import uuid

from vibewatch.vector_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    index_titles,
    point_id,
)


def test_point_id_is_deterministic(make_title):
    # Same title -> same id, every time. This is what makes re-indexing idempotent.
    assert point_id(make_title()) == point_id(make_title())


def test_movie_and_tv_with_same_tmdb_id_differ(make_title):
    # TMDb ids are only unique WITHIN a media type; a movie and a show can share 550.
    # The id must include media_type, or one would overwrite the other in Qdrant.
    assert point_id(make_title(media_type="movie")) != point_id(make_title(media_type="tv"))


def test_different_tmdb_ids_differ(make_title):
    assert point_id(make_title(tmdb_id=550)) != point_id(make_title(tmdb_id=551))


def test_point_id_is_a_valid_uuid(make_title):
    # Qdrant accepts a UUID string as a point id; make sure that's what we produce.
    value = point_id(make_title())
    assert str(uuid.UUID(value)) == value


def test_point_id_ignores_non_identifying_fields(make_title):
    # Editing a plot or rating must NOT change the id -- otherwise a re-index would
    # create a duplicate instead of updating the existing point.
    a = point_id(make_title(overview="old", vote_average=1.0))
    b = point_id(make_title(overview="new", vote_average=9.0))
    assert a == b


# --- writing points: both vectors must arrive ------------------------------------------


class _RecordingUpsertClient:
    def __init__(self):
        self.points = []

    def upsert(self, collection_name, points):
        self.points.extend(points)


def test_index_titles_writes_both_vectors_and_the_payload(make_title):
    # Hybrid search only works if BOTH vectors are stored under the names the query uses.
    # Writing only the dense one would leave sparse search matching nothing -- and the
    # hybrid query would still succeed, just silently ranked by the dense half alone.
    client = _RecordingUpsertClient()
    title = make_title()

    index_titles(client, [title], [[0.1, 0.2, 0.3]], [{7: 1.5, 9: 0.5}])

    (point,) = client.points
    assert point.vector[DENSE_VECTOR] == [0.1, 0.2, 0.3]
    assert point.vector[SPARSE_VECTOR].indices == [7, 9]
    assert point.vector[SPARSE_VECTOR].values == [1.5, 0.5]
    # The payload is what makes a hit self-contained -- no second lookup to show a result.
    assert point.payload["title"] == title.title
    assert point.id == point_id(title)
