"""HTTP API in front of the recommendation pipeline.

Run:  uvicorn vibewatch.api:app --reload        (docs at http://localhost:8000/docs)

WHY A SERVICE AT ALL, when Streamlit could just call `recommend()` directly -- and did,
until now. Three reasons, in order of how much they actually matter:

1. **One pipeline, many clients.** The UI, the CLI, a future mobile app or a colleague's
   script all ask the same question. Without a service each of them needs Python, the
   dependencies, an API key and a Qdrant connection; with one, they need a URL.
2. **The failure surface becomes explicit.** A direct call fails with whatever exception
   happens to bubble up. Here every failure mode is a decided status code -- 422 for a
   malformed request (FastAPI, from the schema), 503 when the index is unreachable -- so
   clients can react instead of parsing tracebacks.
3. **It is where the operational concerns belong.** Rate limits, auth, caching, tracing
   and metrics attach to a service. Bolting them onto a Streamlit script means they only
   apply to the UI.

The API stays THIN on purpose: validate, call the graph, shape the response. All the
thinking lives in `graph.py`, which is why this file has no tests about recommendation
quality -- those already exist a layer down.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vibewatch.graph import DEFAULT_LIMIT, recommend
from vibewatch.vector_store import COLLECTION_NAME, available_genres, get_client

app = FastAPI(
    title="Vibewatch API",
    version="1.0.0",
    summary="Mood-based movie and TV recommendations, grounded in a vector search.",
)


class RecommendRequest(BaseModel):
    """A recommendation request.

    Only `query` is required -- filters are optional because the pipeline infers them
    from the sentence anyway. Anything set here is an EXPLICIT choice and overrides what
    was inferred (the same precedence the UI relies on).
    """

    query: str = Field(min_length=1, description="what you are in the mood for")
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=20)
    # The client sends its own conversation, which keeps this service STATELESS: no
    # sticky sessions, no shared store, and a restart drops nothing. Capped because an
    # unbounded history is an unbounded prompt -- and someone else's bill.
    history: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="earlier requests in this conversation, oldest first",
    )
    media_type: str | None = Field(default=None, pattern="^(movie|tv)$")
    genres: list[str] | None = None
    release_year_min: int | None = Field(default=None, ge=1800, le=2100)
    release_year_max: int | None = Field(default=None, ge=1800, le=2100)
    original_language: str | None = Field(default=None, min_length=2, max_length=2)

    def filters(self) -> dict:
        """Only the constraints the caller actually set."""
        given = self.model_dump(exclude={"query", "limit", "history"}, exclude_none=True)
        return {key: value for key, value in given.items() if value != []}


class Hit(BaseModel):
    """One retrieved title. Self-contained, so a client needs no second request."""

    title: str
    media_type: str
    release_year: int | None = None
    genres: list[str] = Field(default_factory=list)
    vote_average: float = 0.0
    overview: str = ""
    poster_path: str | None = None
    score: float


class RecommendResponse(BaseModel):
    """The answer AND its sources -- never one without the other.

    Returning `hits` alongside `answer` is the API-level expression of the same principle
    the UI follows: a grounded recommendation should always travel with the evidence it
    was grounded in, so any client can show or check it.
    """

    query: str
    answer: str
    hits: list[Hit]
    # What the pipeline understood from the sentence, and whether it had to drop filters.
    # A client that hides these leaves the user guessing why results look the way they do.
    inferred_filters: dict = Field(default_factory=dict)
    relaxed: bool = False


@app.get("/health", summary="Is the service able to answer requests?")
def health() -> dict:
    """Readiness, not liveness: reports whether the INDEX can actually serve queries.

    A process that is running but whose Qdrant is empty or unreachable answers nothing
    useful, and a health check that returns 200 for it is worse than none -- it tells a
    load balancer to send traffic into a hole.
    """
    try:
        client = get_client()
        points = client.count(COLLECTION_NAME).count
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"index unavailable: {error}") from error

    if points == 0:
        raise HTTPException(status_code=503, detail="index is empty -- run scripts.index_titles")
    return {"status": "ok", "indexed_titles": points}


@app.get("/genres", summary="The genres that occur in the catalogue")
def genres() -> dict:
    """Lets a client build a filter UI without hardcoding a list that will drift."""
    try:
        return {"genres": available_genres(get_client())}
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"index unavailable: {error}") from error


@app.post("/recommend", response_model=RecommendResponse, summary="Get a recommendation")
def recommend_endpoint(request: RecommendRequest) -> RecommendResponse:
    """Run the full pipeline: understand -> retrieve -> generate."""
    try:
        state = recommend(
            request.query,
            limit=request.limit,
            history=request.history,
            **request.filters(),
        )
    except Exception as error:
        # The dependencies this endpoint needs (Qdrant, Gemini) are external, so their
        # failure is a 503 -- "try again", not "your request was wrong".
        raise HTTPException(status_code=503, detail=f"pipeline failed: {error}") from error

    return RecommendResponse(
        query=request.query,
        answer=state["answer"],
        hits=state["hits"],
        inferred_filters=state.get("inferred_filters") or {},
        relaxed=bool(state.get("relaxed")),
    )
