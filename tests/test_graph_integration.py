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


def test_constraints_in_the_sentence_become_real_filters(live_services):
    """The capability query understanding added, end to end against live services.

    Before this node existed, "funny movies from before 2000" was embedded whole: the
    words "before 2000" pulled the vector around without constraining anything, and the
    results included titles from 2001 and 2025. Now the sentence is split -- "funny" is
    searched, the type and the year become Qdrant filters.

    Asserted on the FILTERS, not on which comedies come back: the catalogue changes, but
    "no title may violate a constraint the user stated" must hold forever.
    """
    state = recommend("funny movies from before 2000", limit=5)

    assert state["hits"], "expected pre-2000 comedies in the catalogue"
    for hit in state["hits"]:
        assert hit["media_type"] == "movie", hit["title"]
        assert hit["release_year"] <= 1999, f"{hit['title']} ({hit['release_year']})"

    # The filters must also be reported, or the UI cannot tell the user what it inferred.
    assert state["inferred_filters"]["media_type"] == "movie"
    assert state["inferred_filters"]["release_year_max"] == 1999


def test_a_pure_mood_infers_no_filters(live_services):
    # The opposite failure mode: inventing a constraint nobody asked for silently removes
    # correct results. A mood with no constraints must produce an empty filter set.
    state = recommend("dark and hopeless", limit=5)

    assert state["inferred_filters"] == {}
    assert state["hits"]


def test_a_follow_up_keeps_the_constraints_it_did_not_revoke(live_services):
    """Multi-turn, end to end. "Something funnier" is meaningless on its own.

    Retrieval has no memory and never will, so the follow-up is resolved against the
    earlier turn BEFORE anything is searched. What must survive is what the user did not
    take back: still korean, still a series.
    """
    state = recommend(
        "something funnier", limit=5, history=["korean series about revenge"]
    )

    assert state["inferred_filters"].get("media_type") == "tv"
    assert state["inferred_filters"].get("original_language") == "ko"
    assert state["hits"]
    for hit in state["hits"]:
        assert hit["media_type"] == "tv", hit["title"]
        assert hit["original_language"] == "ko", hit["title"]


def test_a_new_topic_drops_the_old_constraints(live_services):
    # The opposite failure mode, and the harder one: constraints that HAUNT a fresh
    # request. After "korean series", asking for space documentaries must not still be
    # filtered to korean television.
    state = recommend(
        "space documentaries",
        limit=5,
        history=["korean series about revenge", "something funnier"],
    )

    assert "original_language" not in state["inferred_filters"]
    assert state["hits"]
