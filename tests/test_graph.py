"""Unit tests for the LangGraph flow -- the real compiled graph, fake workers.

We do not mock the graph itself: `build_graph()` compiles exactly what production runs,
and we inject fake retrieval/generation. So these tests prove the WIRING -- that the query
reaches retrieval, that retrieval's hits reach generation, that the final state carries
both -- without Qdrant, Gemini, or a single token of quota.
"""

from vibewatch.graph import build_graph as _real_build_graph
from vibewatch.query_understanding import QueryIntent


def _passthrough_understand(query, genres=None, history=None):
    """Understanding that extracts nothing -- the mood is the whole query."""
    return QueryIntent(search_text=query)


def build_graph(**kwargs):
    """Compile the real graph with every external boundary faked by default.

    Understanding, retrieval and generation all reach out to Qdrant or Gemini in
    production. A unit test that forgets one of them does not fail -- it HANGS, waiting on
    a socket. Defaulting them here means a new test cannot make that mistake.
    """
    kwargs.setdefault("understand_fn", _passthrough_understand)
    kwargs.setdefault("genres_fn", lambda: ["Drama", "Comedy"])
    kwargs.setdefault("generate_fn", lambda query, hits, **kw: "ok")
    return _real_build_graph(**kwargs)


def _fake_retrieve(hits, log=None):
    def _retrieve(query, **kwargs):
        if log is not None:
            log.append({"query": query, **kwargs})
        return hits

    return _retrieve


def test_graph_passes_query_through_to_the_answer():
    hits = [{"score": 0.9, "title": "The Road"}]
    graph = build_graph(
        retrieve_fn=_fake_retrieve(hits),
        generate_fn=lambda query, hits, **kw: f"{len(hits)} pick(s) for {query}",
    )

    state = graph.invoke({"query": "dark survival"})

    assert state["answer"] == "1 pick(s) for dark survival"
    # The sources travel with the answer -- a UI shows them, step 6 evaluates against them.
    assert state["hits"] == hits


def test_retrieved_hits_are_what_generation_receives():
    # This is THE seam of RAG: if generation ever ran on anything other than the retrieved
    # hits, the answer would stop being grounded, silently.
    hits = [{"score": 0.9, "title": "The Road"}, {"score": 0.8, "title": "Alien"}]
    seen = {}

    def spy_generate(query, hits, **kw):
        seen["query"] = query
        seen["hits"] = hits
        return "ok"

    build_graph(retrieve_fn=_fake_retrieve(hits), generate_fn=spy_generate).invoke(
        {"query": "dark survival"}
    )

    assert seen["query"] == "dark survival"
    assert seen["hits"] == hits


def test_filters_and_limit_reach_retrieval():
    # Filters live in the state as a dict and must arrive at retrieve() as keyword
    # arguments -- that unpacking is easy to break and invisible when it does.
    # Non-empty hits keep this focused on the happy path (empty ones trigger the retry
    # below, which would log a second, filterless call).
    log = []
    graph = build_graph(
        retrieve_fn=_fake_retrieve([{"score": 0.9, "title": "Alien"}], log=log),
        generate_fn=lambda query, hits, **kw: "ok",
    )

    graph.invoke(
        {"query": "dark", "limit": 3, "filters": {"media_type": "movie", "genres": ["Drama"]}}
    )

    assert log == [
        {"query": "dark", "limit": 3, "media_type": "movie", "genres": ["Drama"]}
    ]


def test_defaults_apply_when_only_a_query_is_given():
    log = []
    graph = build_graph(
        retrieve_fn=_fake_retrieve([{"score": 0.9, "title": "Alien"}], log=log),
        generate_fn=lambda query, hits, **kw: "ok",
    )

    graph.invoke({"query": "dark"})

    # No filters given must mean "search everything", not a crash on a missing state key.
    assert log == [{"query": "dark", "limit": 5}]


# --- the conditional edge: retry without filters -----------------------------------------


def _retrieve_empty_then(second_result, log):
    """Fake retrieval that finds nothing on the first call and `second_result` after."""

    def _retrieve(query, **kwargs):
        log.append(kwargs)
        return [] if len(log) == 1 else second_result

    return _retrieve


def test_empty_filtered_result_retries_without_filters():
    # Hard filters can rule out everything even when the mood is perfectly matchable.
    # One retry over the whole catalogue beats answering "nothing found".
    log = []
    rescued = [{"score": 0.7, "title": "Alien"}]
    graph = build_graph(
        retrieve_fn=_retrieve_empty_then(rescued, log),
        generate_fn=lambda query, hits, **kw: f"{len(hits)} pick(s)",
    )

    state = graph.invoke({"query": "dark", "filters": {"release_year_min": 2024}})

    assert len(log) == 2, "expected exactly one retry"
    assert log[0]["release_year_min"] == 2024  # first attempt kept the user's filter
    assert "release_year_min" not in log[1]  # the retry searched everything
    assert state["hits"] == rescued
    assert state["answer"] == "1 pick(s)"
    # The UI must be able to tell the user their filters were dropped.
    assert state["relaxed"] is True


def test_hits_on_the_first_try_skip_the_retry():
    log = []
    graph = build_graph(
        retrieve_fn=_fake_retrieve([{"score": 0.9, "title": "Alien"}], log=log),
        generate_fn=lambda query, hits, **kw: "ok",
    )

    state = graph.invoke({"query": "dark", "filters": {"media_type": "movie"}})

    assert len(log) == 1
    assert state.get("relaxed") is not True


def test_empty_result_without_filters_does_not_loop():
    # Nothing found and nothing left to loosen: go straight to generate, which answers
    # honestly. Without the guard this is exactly where a graph loops forever.
    log = []
    graph = build_graph(
        retrieve_fn=_fake_retrieve([], log=log),
        generate_fn=lambda query, hits, **kw: "nothing found",
    )

    state = graph.invoke({"query": "dark"})

    assert len(log) == 1
    assert state["answer"] == "nothing found"


def test_retry_that_still_finds_nothing_stops_after_one_relax():
    log = []
    graph = build_graph(
        retrieve_fn=_fake_retrieve([], log=log),
        generate_fn=lambda query, hits, **kw: "nothing found",
    )

    state = graph.invoke({"query": "dark", "filters": {"media_type": "tv"}})

    # One filtered attempt + one relaxed attempt, then stop -- never a third.
    assert len(log) == 2
    assert state["answer"] == "nothing found"


# --- the understanding node -------------------------------------------------------------


def _understanding(search_text, **filters):
    """An extractor that always returns the given intent, ignoring the query."""
    return lambda query, genres=None, history=None: QueryIntent(
        search_text=search_text, **filters
    )


def test_understood_filters_reach_retrieval():
    # The whole point of the node: words the user typed become real Qdrant filters.
    log = []
    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}], log=log),
        understand_fn=_understanding("funny", media_type="movie", release_year_max=1999),
    ).invoke({"query": "funny movies from before 2000"})

    assert log[0]["media_type"] == "movie"
    assert log[0]["release_year_max"] == 1999


def test_retrieval_searches_the_mood_not_the_whole_sentence():
    # The constraint words must NOT be embedded: "movies from before 2000" would drag the
    # vector towards titles ABOUT the year 2000 instead of titles FROM it.
    log = []
    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}], log=log),
        understand_fn=_understanding("funny", media_type="movie"),
    ).invoke({"query": "funny movies from before 2000"})

    assert log[0]["query"] == "funny"


def test_explicit_filters_override_inferred_ones():
    # A sidebar click is a deliberate choice; the inference is our guess about their
    # words. Letting the guess win is how a UI starts feeling possessed.
    log = []
    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}], log=log),
        understand_fn=_understanding("revenge", media_type="tv"),
    ).invoke({"query": "series about revenge", "filters": {"media_type": "movie"}})

    assert log[0]["media_type"] == "movie"


def test_inferred_filters_are_exposed_for_the_ui():
    # A filter the system inferred but never showed is indistinguishable from a bug to
    # the person wondering where half the catalogue went.
    state = build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}]),
        understand_fn=_understanding("funny", media_type="movie"),
    ).invoke({"query": "funny movies"})

    assert state["inferred_filters"] == {"media_type": "movie"}
    assert state["search_text"] == "funny"


def test_generation_still_answers_the_original_question():
    # Retrieval searches the stripped mood, but the ANSWER must address what was asked --
    # "funny" alone would produce a reply that ignores the year and the media type.
    seen = {}

    def spy_generate(query, hits, **kw):
        seen["query"] = query
        return "ok"

    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}]),
        generate_fn=spy_generate,
        understand_fn=_understanding("funny", media_type="movie"),
    ).invoke({"query": "funny movies from before 2000"})

    assert seen["query"] == "funny movies from before 2000"


def test_a_failing_understanding_step_does_not_take_the_query_down():
    # Understanding is a convenience layer sitting in front of EVERY search. If it can
    # raise, it can break the whole app -- so a failure must degrade to plain search.
    log = []

    def boom(query, genres=None, history=None):
        raise ConnectionError("Gemini down")

    state = build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}], log=log), understand_fn=boom
    ).invoke({"query": "funny movies"})

    assert log[0]["query"] == "funny movies"  # the sentence as typed
    assert state["inferred_filters"] == {}


def test_an_unreachable_genre_vocabulary_does_not_take_the_query_down():
    # Fetching the genre list hits Qdrant. That call failing must not cost us the query.
    log = []

    def boom():
        raise ConnectionError("Qdrant down")

    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}], log=log), genres_fn=boom
    ).invoke({"query": "funny movies"})

    assert log[0]["query"] == "funny movies"


def test_conversation_history_reaches_understanding():
    # A follow-up like "but shorter" is meaningless without the earlier turns, and
    # retrieval has no memory -- so the history has to arrive at the one step that can
    # rewrite the request into something self-contained.
    seen = {}

    def spy_understand(query, genres=None, history=None):
        seen["history"] = history
        return QueryIntent(search_text=query)

    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}]), understand_fn=spy_understand
    ).invoke({"query": "but shorter", "history": ["dark survival movies"]})

    assert seen["history"] == ["dark survival movies"]


def test_a_first_turn_passes_no_history():
    # An empty list must arrive as None, so the prompt keeps no empty "earlier requests"
    # block that could invite the model to invent context.
    seen = {}

    def spy_understand(query, genres=None, history=None):
        seen["history"] = history
        return QueryIntent(search_text=query)

    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}]), understand_fn=spy_understand
    ).invoke({"query": "dark survival"})

    assert seen["history"] is None


# --- staged relaxation ------------------------------------------------------------------
# Regression tests for a real failure: "a korean thriller, a bit older" matched nothing
# (the catalogue holds no pre-2015 korean thriller), the all-or-nothing retry dropped
# "korean" as well, and the user was offered turkish soap operas.


def _retrieve_empty_until(condition, log):
    """Fake retrieval that keeps returning nothing until `condition(filters)` holds."""

    def _retrieve(query, **kwargs):
        filters = {k: v for k, v in kwargs.items() if k != "limit"}
        log.append(filters)
        return [{"title": "found", "media_type": "tv"}] if condition(filters) else []

    return _retrieve


def test_the_genre_is_given_up_before_the_language():
    # THE bug. Dropping the genre alone leaves 14 older korean titles; dropping everything
    # leaves the whole catalogue, which is how turkish soaps ended up as the answer.
    log = []
    graph = build_graph(retrieve_fn=_retrieve_empty_until(lambda f: "genres" not in f, log))

    state = graph.invoke(
        {
            "query": "korean thriller, a bit older",
            "filters": {
                "genres": ["Thriller"],
                "original_language": "ko",
                "release_year_max": 2015,
            },
        }
    )

    # Second attempt: genre gone, language and year still enforced.
    assert "genres" not in log[1]
    assert log[1]["original_language"] == "ko"
    assert log[1]["release_year_max"] == 2015
    assert state["dropped_filters"] == {"genres": ["Thriller"]}


def test_relaxation_continues_stage_by_stage_when_still_empty():
    # Each round gives up one more group, in the order of least painful first.
    log = []
    graph = build_graph(
        retrieve_fn=_retrieve_empty_until(lambda f: "original_language" not in f, log)
    )

    state = graph.invoke(
        {
            "query": "x",
            "filters": {
                "genres": ["Thriller"],
                "release_year_max": 2015,
                "original_language": "ko",
            },
        }
    )

    assert [sorted(attempt) for attempt in log] == [
        ["genres", "original_language", "release_year_max"],  # as asked
        ["original_language", "release_year_max"],            # genre given up
        ["original_language"],                                # year given up
        [],                                                   # language given up
    ]
    assert set(state["dropped_filters"]) == {
        "genres",
        "release_year_max",
        "original_language",
    }


def test_stages_that_would_drop_nothing_are_skipped():
    # A query carrying only a language filter must not loop through empty rounds for the
    # genre and year stages it never had.
    log = []
    graph = build_graph(retrieve_fn=_retrieve_empty_until(lambda f: not f, log))

    graph.invoke({"query": "x", "filters": {"original_language": "ko"}})

    assert log == [{"original_language": "ko"}, {}]


def test_relaxation_stops_instead_of_looping_forever():
    log = []
    graph = build_graph(retrieve_fn=_retrieve_empty_until(lambda f: False, log))

    state = graph.invoke({"query": "x", "filters": {"genres": ["Thriller"]}})

    assert len(log) == 2, "one filtered attempt, one relaxed, then stop"
    assert state["answer"] == "ok"


def test_generation_is_told_which_constraints_were_dropped():
    # Without this the model justifies whatever it was handed -- describing a 2018 show as
    # "slightly older" for a request that asked for older korean thrillers.
    seen = {}

    def spy_generate(query, hits, **kw):
        seen.update(kw)
        return "ok"

    build_graph(
        retrieve_fn=_retrieve_empty_until(lambda f: "genres" not in f, []),
        generate_fn=spy_generate,
    ).invoke({"query": "x", "filters": {"genres": ["Thriller"], "original_language": "ko"}})

    assert seen["dropped_filters"] == {"genres": ["Thriller"]}


def test_a_successful_search_reports_nothing_dropped():
    seen = {}

    def spy_generate(query, hits, **kw):
        seen.update(kw)
        return "ok"

    build_graph(
        retrieve_fn=_fake_retrieve([{"title": "x"}]), generate_fn=spy_generate
    ).invoke({"query": "x", "filters": {"genres": ["Thriller"]}})

    assert not seen.get("dropped_filters")
