"""Tests for the Streamlit UI, headless -- no browser, no Qdrant, no Gemini.

`AppTest` runs app.py exactly as Streamlit would and exposes the resulting widgets and
elements. The page is an API client now, so the boundary we patch is the HTTP client --
no server, no Qdrant, no Gemini, while every line of UI logic runs for real.

The logic worth testing is the part that quietly changes what the user gets:

- the sidebar -> filter-kwargs translation, especially "slider at its extreme means NO
  bound". Sending 1950 instead of None would silently drop every title whose release year
  is unknown -- a filter the user never asked for.
- the `relaxed` warning. If that ever stopped rendering, the app would present results for
  a search the user did not ask for, and look perfectly fine doing it.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import streamlit as st
from streamlit.testing.v1 import AppTest

# Absolute, derived from this file's location. A bare "app.py" is resolved against the
# CALLING file in current Streamlit, i.e. tests/app.py -- which does not exist. (Older
# versions resolved against the working directory, so the relative form passed locally and
# failed in CI. Deriving the path removes the dependency on both cwd and library version.)
APP = str(Path(__file__).parent.parent / "app.py")


def _hit(title="The Road", **overrides):
    return {
        "score": 0.68,
        "tmdb_id": 20766,
        "media_type": "movie",
        "title": title,
        "overview": "A father and his son walk alone through burned America.",
        "original_language": "en",
        "genres": ["Drama", "Adventure"],
        "release_year": 2009,
        "popularity": 20.0,
        "vote_average": 7.0,
        "vote_count": 4000,
        "poster_path": "/poster.jpg",
        **overrides,
    }


@pytest.fixture
def app(monkeypatch):
    """An AppTest wired to fakes, plus the log of calls the UI made to the pipeline.

    Streamlit's caches live in the PROCESS, not in the AppTest instance, so without this
    clear a later test would silently be served an earlier test's cached answer -- and
    pass for the wrong reason. (In the app itself that persistence is the point: it is
    what stops a re-render from spending Gemini quota again.)
    """
    st.cache_data.clear()
    st.cache_resource.clear()

    calls: list[dict] = []

    def fake_recommend(query, *, limit=5, **filters):
        calls.append({"query": query, "limit": limit, **filters})
        return {"query": query, "hits": [_hit()], "answer": "Watch The Road (2009)."}

    # Patched on the client module app.py imports, before the page runs.
    monkeypatch.setattr("vibewatch.client.recommend", fake_recommend)
    monkeypatch.setattr("vibewatch.client.genres", lambda: ["Drama", "Horror"])

    at = AppTest.from_file(APP, default_timeout=30)
    at.calls = calls
    return at


def _search(app, query="dark survival"):
    app.run()
    app.text_input[0].set_value(query).run()
    return app


def test_a_plain_query_sends_no_filters(app):
    # Untouched sidebar must mean "search the whole catalogue". Any stray filter here
    # would narrow every single search in the app.
    _search(app)

    assert app.calls == [{"query": "dark survival", "limit": 5}]


def test_answer_and_sources_are_both_rendered(app):
    _search(app)

    rendered = " ".join(element.value for element in app.markdown)
    assert "Watch The Road (2009)." in rendered
    # The retrieved title must be visible next to the answer -- that is what lets a user
    # check the recommendation against what it was actually based on.
    assert "The Road" in rendered


def test_sidebar_choices_become_filter_kwargs(app):
    app.run()
    app.radio[0].set_value("Movies")
    app.multiselect[0].set_value(["Drama"])
    app.slider[0].set_value((2000, 2020))  # year range
    app.slider[1].set_value(8)  # how many to retrieve
    app.text_input[0].set_value("dark survival").run()

    assert app.calls[-1] == {
        "query": "dark survival",
        "limit": 8,
        "media_type": "movie",
        "genres": ["Drama"],
        "release_year_min": 2000,
        "release_year_max": 2020,
    }


def test_year_slider_at_its_extremes_sends_no_bound(app):
    # THE subtle one: 1950/2030 are slider ends, not user intent. Passing them as a real
    # range would exclude every title with an unknown release year.
    app.run()
    app.slider[0].set_value((1950, 2020))
    app.text_input[0].set_value("dark").run()

    assert "release_year_min" not in app.calls[-1]
    assert app.calls[-1]["release_year_max"] == 2020


def test_relaxed_result_warns_that_filters_were_dropped(app, monkeypatch):
    monkeypatch.setattr(
        "vibewatch.client.recommend",
        lambda query, *, limit=5, **filters: {
            "hits": [_hit()],
            "answer": "Watch The Road (2009).",
            "relaxed": True,
        },
    )

    app.run()
    app.radio[0].set_value("TV shows")
    app.text_input[0].set_value("dark").run()

    warnings = " ".join(element.value for element in app.warning)
    assert "filters" in warnings.lower()


def test_a_dead_backend_shows_an_error_instead_of_crashing(app, monkeypatch):
    from vibewatch.client import ApiError

    def boom(query, *, limit=5, **filters):
        raise ApiError("Qdrant unreachable")

    monkeypatch.setattr("vibewatch.client.recommend", boom)

    _search(app)

    assert not app.exception, "the app must handle a dead backend itself"
    errors = " ".join(element.value for element in app.error)
    assert "Qdrant unreachable" in errors


def test_inferred_filters_are_shown_to_the_user(app, monkeypatch):
    # Query understanding is invisible unless the UI says what it did. An applied filter
    # nobody was told about looks exactly like a bug from the user's side.
    monkeypatch.setattr(
        "vibewatch.client.recommend",
        lambda query, *, limit=5, **filters: {
            "hits": [_hit()],
            "answer": "Watch The Road (2009).",
            "inferred_filters": {"media_type": "movie", "release_year_max": 1999},
        },
    )

    _search(app, "funny movies from before 2000")

    shown = " ".join(element.value for element in app.info)
    assert "media type" in shown
    assert "1999" in shown


def test_nothing_is_claimed_when_nothing_was_inferred(app):
    # A pure mood infers no filters, so the app must stay silent rather than print an
    # empty "Understood from your request:" banner.
    _search(app, "dark survival")

    assert not any("Understood" in element.value for element in app.info)
