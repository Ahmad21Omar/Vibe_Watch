"""The LangGraph flow that ties understanding, retrieval and generation into one pipeline.

                                    hits found
    query --> [understand] --> [retrieve] ------> [generate] --> answer
                                   ^   |
                                   |   | nothing found, but filters were set
                                   |   v
                               [relax_filters]

Honest framing, because this is the question an interviewer will ask: a straight line of
steps alone would not justify a graph -- nested function calls do that in one line. Two
things here earn it:

- **The conditional edge.** The route out of `retrieve` depends on its RESULT, and
  expressing that as an edge keeps each node a small pure function instead of growing an
  if/else pyramid inside one procedure.
- **The shared state.** `understand` writes `search_text` and `filters`; `retrieve` reads
  them without knowing who produced them; `relax_filters` rewrites `filters` and sends the
  flow back. Threading that through call arguments would mean every node signature changes
  whenever a new one is inserted.

What follows later rides on the same structure: re-ranking, a guard that checks the answer
only names retrieved titles, streaming, checkpointing, per-node tracing.

The state is the contract between nodes: each node receives the whole dict and returns
ONLY the keys it wants to change, which LangGraph merges in. That is what keeps the nodes
independent -- `generate` does not care who filled `hits`, only that they are there.

`build_graph()` takes its two workers as arguments so tests can compile the real graph with
fake retrieval/generation: the wiring gets verified, no Docker and no API key involved.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from vibewatch.generation import generate_recommendation
from vibewatch.query_understanding import QueryIntent, understand
from vibewatch.retrieval import retrieve
from vibewatch.vector_store import available_genres, get_client

DEFAULT_LIMIT = 5


@lru_cache(maxsize=1)
def catalogue_genres() -> list[str]:
    """The genre vocabulary handed to query understanding.

    Cached for the process: the list only changes when the catalogue is re-indexed, and
    fetching it per query would add a Qdrant round trip to every single request.
    """
    return available_genres(get_client())


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
    # What `understand` extracted: the mood with the constraints removed, and the filters
    # it found. Both stay in the state so the UI can SHOW them -- a filter the system
    # inferred but never displayed is indistinguishable from a bug to the person who
    # wonders why half the catalogue disappeared.
    search_text: NotRequired[str]
    inferred_filters: NotRequired[dict]


def build_graph(
    *,
    retrieve_fn: Callable[..., list[dict]] = retrieve,
    generate_fn: Callable[..., str] = generate_recommendation,
    understand_fn: Callable[..., QueryIntent] = understand,
    genres_fn: Callable[[], list[str]] = catalogue_genres,
):
    """Compile the understand -> retrieve -> generate graph. Returns a runnable graph."""

    def understand_node(state: RecommendationState) -> dict:
        """Split the request into a mood and hard filters before anything is searched."""
        try:
            intent = understand_fn(state["query"], genres=genres_fn())
        except Exception:
            # Understanding is a convenience layer. If it cannot run -- Qdrant down while
            # fetching the genre vocabulary, API failure -- the query still deserves an
            # answer, so fall through to searching the sentence as typed.
            return {"search_text": state["query"], "inferred_filters": {}}

        inferred = intent.filters()
        explicit = state.get("filters", {})
        return {
            "search_text": intent.search_text,
            "inferred_filters": inferred,
            # EXPLICIT FILTERS WIN. If the sidebar says "movies" and the sentence implies
            # "series", the click is the stronger signal: the user chose it deliberately,
            # while the inference is our guess about their words. Overriding a deliberate
            # choice with a guess is how a UI starts feeling possessed.
            "filters": {**inferred, **explicit},
        }

    def retrieve_node(state: RecommendationState) -> dict:
        hits = retrieve_fn(
            # The mood alone, with the constraint words removed -- searching the full
            # sentence would put "movies from before 2000" back into the embedding.
            state.get("search_text") or state["query"],
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
    builder.add_node("understand", understand_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("relax_filters", relax_filters_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "retrieve")
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
