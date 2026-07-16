"""Tests for the Title data model.

These are pure, fast unit tests: no API calls, no Qdrant, no network. They pin down
the behaviour we rely on everywhere else -- especially embedding_text(), because what
we embed decides what we can find.
"""

import pytest
from pydantic import ValidationError


def test_embedding_text_contains_title_genres_and_plot(make_title):
    text = make_title().embedding_text()
    assert "Fight Club" in text
    assert "Drama, Thriller" in text
    assert "underground fight club" in text


def test_embedding_text_labels_movies_and_tv_differently(make_title):
    assert make_title(media_type="movie").embedding_text().startswith("Movie:")
    assert make_title(media_type="tv").embedding_text().startswith("TV show:")


def test_embedding_text_handles_missing_genres(make_title):
    # A title with no genres must still produce sensible text, not "Genres: ".
    assert "Genres: unknown" in make_title(genres=[]).embedding_text()


def test_release_year_may_be_none(make_title):
    # Some titles genuinely have no release date; the model must allow it.
    assert make_title(release_year=None).release_year is None


def test_media_type_is_restricted(make_title):
    # Literal["movie", "tv"] must reject anything else -- this guards our filters.
    with pytest.raises(ValidationError):
        make_title(media_type="documentary")


def test_types_are_coerced(make_title):
    # TMDb sometimes sends numbers as strings; pydantic should coerce them.
    title = make_title(tmdb_id="550", vote_average="8.4")
    assert title.tmdb_id == 550
    assert title.vote_average == pytest.approx(8.4)
