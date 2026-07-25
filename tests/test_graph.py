"""Unit tests for the LangGraph flow -- the real compiled graph, fake workers.

We do not mock the graph itself: `build_graph()` compiles exactly what production runs,
and we inject fake retrieval/generation. So these tests prove the WIRING -- that the query
reaches retrieval, that retrieval's hits reach generation, that the final state carries
both -- without Qdrant, Gemini, or a single token of quota.
"""

from vibewatch.graph import build_graph


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
        generate_fn=lambda query, hits: f"{len(hits)} pick(s) for {query}",
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

    def spy_generate(query, hits):
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
        generate_fn=lambda query, hits: "ok",
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
        generate_fn=lambda query, hits: "ok",
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
        generate_fn=lambda query, hits: f"{len(hits)} pick(s)",
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
        generate_fn=lambda query, hits: "ok",
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
        generate_fn=lambda query, hits: "nothing found",
    )

    state = graph.invoke({"query": "dark"})

    assert len(log) == 1
    assert state["answer"] == "nothing found"


def test_retry_that_still_finds_nothing_stops_after_one_relax():
    log = []
    graph = build_graph(
        retrieve_fn=_fake_retrieve([], log=log),
        generate_fn=lambda query, hits: "nothing found",
    )

    state = graph.invoke({"query": "dark", "filters": {"media_type": "tv"}})

    # One filtered attempt + one relaxed attempt, then stop -- never a third.
    assert len(log) == 2
    assert state["answer"] == "nothing found"
