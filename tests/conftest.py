"""Shared pytest fixtures.

pytest auto-discovers this file and makes its fixtures available to every test in the
folder -- no import needed. Centralising the Title factory here keeps each test focused
on the ONE thing it checks instead of repeating ten boilerplate fields.
"""

import pytest

from vibewatch.config import settings
from vibewatch.models import Title
from vibewatch.vector_store import COLLECTION_NAME, get_client


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


def _index_ready() -> bool:
    """True only if Qdrant is reachable AND the collection actually has points."""
    try:
        client = get_client()
        return (
            client.collection_exists(COLLECTION_NAME)
            and client.count(COLLECTION_NAME).count > 0
        )
    except Exception:
        return False


@pytest.fixture
def live_services():
    """Skip the test unless Gemini and a populated Qdrant are both actually available.

    Deliberately a fixture rather than a module-level `skipif`: a skipif condition is
    evaluated at COLLECTION time, so a bare `pytest` -- which deselects the integration
    tests entirely -- would still have opened a socket to Qdrant and waited for it to time
    out. A fixture runs only if the test using it runs.
    """
    if not settings.gemini_api_key or not _index_ready():
        pytest.skip("needs GEMINI_API_KEY and a populated Qdrant 'titles' collection")
