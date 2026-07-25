"""End-to-end test of the full RAG pipeline against LIVE services.

This is the one test that exercises everything at once: a mood query goes in, Gemini embeds
it, Qdrant retrieves, Gemini writes the recommendation, and a finished answer comes out.
Marked `integration` and deselected by default -- run it with `pytest -m integration` after
`docker compose up -d`.

The interesting assertion is FAITHFULNESS: the answer must name a title we actually
retrieved. That is the property RAG exists to guarantee, and the one that breaks silently
-- a mis-wired prompt still produces fluent, confident, entirely invented recommendations.
"""

import pytest

from vibewatch.graph import recommend

pytestmark = pytest.mark.integration


def test_full_pipeline_answers_from_retrieved_titles(live_services):
    state = recommend("survival, people fighting to stay alive in a hostile world", limit=5)

    assert state["hits"], "retrieval returned nothing"
    answer = state["answer"]
    assert answer.strip(), "the model produced an empty answer"

    # At least one retrieved title must appear verbatim in the answer. We do not demand a
    # specific film (the catalogue and the ranking shift), only that the recommendation is
    # anchored in what we actually gave the model.
    retrieved_titles = [hit["title"] for hit in state["hits"]]
    assert any(title in answer for title in retrieved_titles), (
        f"answer names none of the retrieved titles {retrieved_titles}: {answer}"
    )


def test_filters_constrain_the_pipeline(live_services):
    # A filter that the catalogue can satisfy must survive the whole way through, and the
    # rescue path must therefore stay untouched.
    state = recommend("something funny and light", media_type="movie")

    assert all(hit["media_type"] == "movie" for hit in state["hits"])
    assert state.get("relaxed") is not True


def test_impossible_filter_triggers_the_relax_retry(live_services):
    # No title in the catalogue is from the year 3000, so the filtered search must come back
    # empty -- and the graph must rescue the run instead of giving up.
    state = recommend("dark survival", release_year_min=3000)

    assert state["relaxed"] is True
    assert state["hits"], "the relaxed retry should have found something"
    assert state["answer"].strip()
