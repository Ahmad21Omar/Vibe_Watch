"""Shared pytest fixtures.

pytest auto-discovers this file and makes its fixtures available to every test in the
folder -- no import needed. Centralising the Title factory here keeps each test focused
on the ONE thing it checks instead of repeating ten boilerplate fields.
"""

import pytest

from vibewatch.models import Title


@pytest.fixture
def make_title():
    """Return a factory that builds a valid Title, overriding only what a test needs.

    We return a *function* rather than a Title so each test can tweak fields:
        def test_x(make_title):
            title = make_title(media_type="tv")
    """

    def _make(**overrides) -> Title:
        defaults = dict(
            tmdb_id=550,
            media_type="movie",
            title="Fight Club",
            overview="An insomniac office worker forms an underground fight club.",
            original_language="en",
            genres=["Drama", "Thriller"],
            release_year=1999,
            popularity=61.4,
            vote_average=8.4,
            vote_count=27000,
            poster_path="/poster.jpg",
        )
        return Title(**{**defaults, **overrides})

    return _make
