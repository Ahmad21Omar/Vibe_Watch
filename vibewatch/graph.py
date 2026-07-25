"""The LangGraph flow that ties retrieval and generation into one pipeline.

    query --> [retrieve] --> hits --> [generate] --> answer

Honest framing, because this is the question an interviewer will ask: for two sequential
steps, LangGraph buys nothing that `generate_recommendation(query, retrieve(query))` would
not. The reason to introduce it HERE, while the flow is still trivial, is what comes next:

- conditional edges -- "no hits? drop the filters and search again" (added in the next step)
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

    def generate_node(state: RecommendationState) -> dict:
        answer = generate_fn(state["query"], state["hits"])
        return {"answer": answer}

    builder = StateGraph(RecommendationState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
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
