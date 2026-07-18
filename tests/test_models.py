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


def test_context_block_includes_grounding_facts(make_title):
    block = make_title().as_context_block()
    assert "Fight Club (1999)" in block
    assert "Drama, Thriller" in block
    assert "underground fight club" in block


def test_context_block_formats_rating_to_one_decimal(make_title):
    # The LLM prompt should read "8.4/10", not "8.4000000001/10".
    assert "Rating: 8.4/10" in make_title(vote_average=8.4).as_context_block()


def test_context_block_handles_missing_year(make_title):
    # A title without a release year must still produce a clean block, not "(None)".
    assert "(unknown)" in make_title(release_year=None).as_context_block()


def test_context_block_labels_media_type(make_title):
    assert "Movie" in make_title(media_type="movie").as_context_block()
    assert "TV show" in make_title(media_type="tv").as_context_block()
