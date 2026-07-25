"""The LangGraph flow that ties retrieval and generation into one pipeline.

                    hits found
    query --> [retrieve] ------> [generate] --> answer
                  ^   |
                  |   | nothing found, but filters were set
                  |   v
              [relax_filters]

Honest framing, because this is the question an interviewer will ask: two sequential steps
alone would not justify a graph -- `generate_recommendation(query, retrieve(query))` does
that in one line. The retry loop is where it starts paying off: the route out of `retrieve`
depends on its RESULT, and expressing that as a conditional edge keeps each node a small
pure function instead of growing an if/else pyramid inside one procedure. What follows
later rides on the same structure:

- extra nodes -- query understanding, re-ranking, a guard that checks the answer only
  names retrieved titles
- state that every node reads and extends, instead of threading arguments through calls
- streaming, checkpointing and per-node tracing that come for free once the flow is a graph

The state is the contract between nodes: each node receives the whole dict and returns
ONLY the keys it wants to change, which LangGraph merges in. That is what keeps the nodes
independent -- `generate` does not care who filled `hits`, only that they are there.

`build_graph()` takes its two workers as arguments so tests can compile the real graph with
fake retrieval/generation: the wiring gets verified, no Docker and no API key involved.
"""

from collections.abc import Callable
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from vibewatch.generation import generate_recommendation
from vibewatch.retrieval import retrieve

DEFAULT_LIMIT = 5


class RecommendationState(TypedDict):
    """Everything that flows through the graph.

    `query` is the only required input; the rest is filled in as the run progresses.
    Keeping `hits` in the state (rather than passing it straight to the LLM) means the
    caller gets the sources back alongside the answer -- posters, ratings, and the ability
    to check the recommendation against what was actually retrieved.
    """

    query: str
    filters: NotRequired[dict]
    limit: NotRequired[int]
    hits: NotRequired[list[dict]]
    answer: NotRequired[str]
    # True once the filters were dropped to rescue an empty result. It also doubles as the
    # loop guard, and the UI should surface it -- silently ignoring what the user asked for
    # ("only movies since 2020") would be worse than showing nothing.
    relaxed: NotRequired[bool]


def build_graph(
    *,
    retrieve_fn: Callable[..., list[dict]] = retrieve,
    generate_fn: Callable[..., str] = generate_recommendation,
):
    """Compile the retrieve -> generate graph. Returns a runnable graph."""

    def retrieve_node(state: RecommendationState) -> dict:
        hits = retrieve_fn(
            state["query"],
            limit=state.get("limit", DEFAULT_LIMIT),
            **state.get("filters", {}),
        )
        return {"hits": hits}

    def relax_filters_node(state: RecommendationState) -> dict:
        """Drop the hard filters so the retry searches the whole catalogue."""
        return {"filters": {}, "relaxed": True}

    def generate_node(state: RecommendationState) -> dict:
        answer = generate_fn(state["query"], state["hits"])
        return {"answer": answer}

    def route_after_retrieve(state: RecommendationState) -> str:
        """Decide where to go based on what retrieval actually found.

        Hard filters are unforgiving by design -- "TV shows from 2024 tagged Western" can
        easily match nothing in a 900-title catalogue. Rather than answering "nothing
        found" when the MOOD is perfectly matchable, we retry once without the filters.
        The `relaxed` flag makes that at most one retry: without it, a genuinely empty
        catalogue would send the graph round forever.
        """
        if state["hits"]:
            return "generate"
        if state.get("filters") and not state.get("relaxed"):
            return "relax_filters"
        # Nothing found and nothing left to loosen -- generate() answers honestly.
        return "generate"

    builder = StateGraph(RecommendationState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("relax_filters", relax_filters_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    # The explicit destination list is not redundant: it is what lets LangGraph draw and
    # validate the graph, since the possible targets cannot be inferred from a `-> str`.
    builder.add_conditional_edges(
        "retrieve", route_after_retrieve, ["relax_filters", "generate"]
    )
    builder.add_edge("relax_filters", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()


# Compiled once at import: building the graph is pure wiring (no connection is opened),
# and recompiling it per request would be wasted work.
_graph = build_graph()


def recommend(query: str, *, limit: int = DEFAULT_LIMIT, **filters) -> RecommendationState:
    """Run the full pipeline for one query and return the final state.

    The state -- not just the text -- is the return value on purpose: a UI needs the
    `hits` to show posters next to the `answer`, and an evaluation harness (step 6) needs
    both to judge whether the answer is faithful to its sources.
    """
    return _graph.invoke({"query": query, "limit": limit, "filters": filters})
