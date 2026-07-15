"""Tests for point_id(): the idempotency guarantee of our indexing.

The whole "re-running the indexer updates instead of duplicating" property rests on
this one function being deterministic AND collision-free between movies and TV shows.
If either breaks, we silently get duplicate or overwritten titles in Qdrant.
"""

import uuid

from vibewatch.models import Title
from vibewatch.vector_store import point_id


def make_title(**overrides) -> Title:
    defaults = dict(
        tmdb_id=550,
        media_type="movie",
        title="Fight Club",
        overview="x",
        original_language="en",
        genres=["Drama"],
        release_year=1999,
        popularity=1.0,
        vote_average=8.0,
        vote_count=10,
        poster_path=None,
    )
    return Title(**{**defaults, **overrides})


def test_point_id_is_deterministic():
    # Same title -> same id, every time. This is what makes re-indexing idempotent.
    assert point_id(make_title()) == point_id(make_title())


def test_movie_and_tv_with_same_tmdb_id_differ():
    # TMDb ids are only unique WITHIN a media type; a movie and a show can share 550.
    # The id must include media_type, or one would overwrite the other in Qdrant.
    assert point_id(make_title(media_type="movie")) != point_id(make_title(media_type="tv"))


def test_different_tmdb_ids_differ():
    assert point_id(make_title(tmdb_id=550)) != point_id(make_title(tmdb_id=551))


def test_point_id_is_a_valid_uuid():
    # Qdrant accepts a UUID string as a point id; make sure that's what we produce.
    value = point_id(make_title())
    assert str(uuid.UUID(value)) == value


def test_point_id_ignores_non_identifying_fields():
    # Editing a plot or rating must NOT change the id -- otherwise a re-index would
    # create a duplicate instead of updating the existing point.
    a = point_id(make_title(overview="old", vote_average=1.0))
    b = point_id(make_title(overview="new", vote_average=9.0))
    assert a == b
