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
    log = []
    graph = build_graph(
        retrieve_fn=_fake_retrieve([], log=log),
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
        retrieve_fn=_fake_retrieve([], log=log),
        generate_fn=lambda query, hits: "ok",
    )

    graph.invoke({"query": "dark"})

    # No filters given must mean "search everything", not a crash on a missing state key.
    assert log == [{"query": "dark", "limit": 5}]
