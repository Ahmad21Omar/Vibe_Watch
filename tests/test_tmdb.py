"""Tests for to_title(): mapping raw TMDb JSON onto our unified Title model.

This is the trickiest bit of the ingestion -- it irons out the movie/TV field
differences and guards against missing/dirty data. Exactly the kind of code that
deserves tests, because a silent mistake here corrupts every downstream vector.
"""

from vibewatch.tmdb import to_title

GENRES = {18: "Drama", 53: "Thriller", 35: "Comedy"}


def test_movie_uses_title_and_release_date():
    raw = {
        "id": 550,
        "title": "Fight Club",
        "release_date": "1999-10-15",
        "overview": "An insomniac...",
        "genre_ids": [18, 53],
    }
    title = to_title(raw, media_type="movie", genre_names=GENRES)
    assert title.title == "Fight Club"
    assert title.release_year == 1999
    assert title.genres == ["Drama", "Thriller"]


def test_tv_uses_name_and_first_air_date():
    # TV shows use different field names -- the whole reason to_title() exists.
    raw = {
        "id": 1396,
        "name": "Breaking Bad",
        "first_air_date": "2008-01-20",
        "overview": "A chemistry teacher...",
        "genre_ids": [18],
    }
    title = to_title(raw, media_type="tv", genre_names=GENRES)
    assert title.title == "Breaking Bad"
    assert title.release_year == 2008


def test_missing_date_yields_none_year():
    # A missing or empty date must not crash int(""); it should become None.
    raw = {"id": 1, "title": "No Date", "overview": "x", "genre_ids": []}
    assert to_title(raw, media_type="movie", genre_names=GENRES).release_year is None


def test_unknown_genre_ids_are_dropped():
    # If TMDb sends a genre id we have no name for, we skip it instead of crashing.
    raw = {"id": 2, "title": "X", "overview": "x", "genre_ids": [18, 999]}
    assert to_title(raw, media_type="movie", genre_names=GENRES).genres == ["Drama"]


def test_absent_optional_fields_get_defaults():
    # The bare minimum TMDb could return; optional fields fall back to safe defaults.
    raw = {"id": 3, "title": "Sparse", "overview": "x"}
    title = to_title(raw, media_type="movie", genre_names=GENRES)
    assert title.genres == []
    assert title.popularity == 0.0
    assert title.vote_count == 0
    assert title.poster_path is None
