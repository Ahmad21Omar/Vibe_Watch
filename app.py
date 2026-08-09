"""Streamlit frontend: describe a mood, get a grounded recommendation.

Run:  uvicorn vibewatch.api:app     (the service this page talks to)
      streamlit run app.py
Or simply `docker compose up -d --build`, which starts both.

The layout is a deliberate argument about RAG, not just a form: the generated answer sits
next to the titles it was generated FROM. A recommender that only shows prose asks you to
trust it; showing the retrieved evidence alongside lets anyone check the answer against
its sources -- which is exactly what grounding means and the first thing a reviewer of an
LLM feature should be able to do.

This page is an ORDINARY API CLIENT. It could import the pipeline directly -- it used to --
but then the service would be decoration, the two paths could drift apart, and the UI would
prove nothing about the API. As a client, the most-used part of the app exercises the same
contract every other consumer gets.

Streamlit re-runs this whole script on every interaction, so anything expensive has to be
cached explicitly. That is what `@st.cache_data` is for below -- without it, every click
would spend Gemini quota again.
"""

import streamlit as st

from vibewatch import client
from vibewatch.client import ApiError

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w200"

# Bounds of the year slider. At either extreme we send NO bound to Qdrant rather than the
# number itself -- otherwise a title with an unknown release year would be filtered out by
# a range the user never actually narrowed.
EARLIEST_YEAR = 1950
LATEST_YEAR = 2030

EXAMPLE_QUERIES = [
    "survival, dark, fighting to stay alive",
    # Deliberately shows off query understanding: type, genre and year are extracted from
    # the sentence, so the sidebar is optional rather than required.
    "funny movies from before 2000",
    "korean series about revenge",
    "mind-bending sci-fi that makes me think",
]

st.set_page_config(page_title="Vibewatch", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner=False)
def _genres() -> list[str]:
    """The catalogue's real genres. Cached: they change only when we re-index."""
    try:
        return client.genres()
    except ApiError:
        # A dead API should not blank the page -- the query below will report it properly.
        return []


@st.cache_data(show_spinner=False)
def _recommend(query: str, limit: int, filters: dict) -> dict:
    """Cached API call.

    Keyed on the query AND the filters, so re-rendering (a checkbox, a resize) is free
    while a genuinely new request still goes through. Every uncached call costs one
    embedding of the daily quota plus two LLM calls on the server side.
    """
    return client.recommend(query, limit=limit, **filters)


def render_hit(hit: dict) -> None:
    """One retrieved title: poster, name, facts, plot."""
    poster, facts = st.columns([1, 3])

    with poster:
        if hit.get("poster_path"):
            st.image(f"{POSTER_BASE_URL}{hit['poster_path']}")
        else:
            st.markdown("### 🎬")

    with facts:
        year = hit.get("release_year") or "—"
        kind = "Movie" if hit["media_type"] == "movie" else "TV show"
        st.markdown(f"**{hit['title']}** ({year}) · {kind}")
        st.caption(
            f"{', '.join(hit.get('genres') or []) or 'unknown'} · "
            f"⭐ {hit.get('vote_average', 0):.1f} · similarity {hit['score']:.3f}"
        )
        st.write(hit.get("overview", ""))


st.title("🎬 Vibewatch")
st.caption(
    "Describe a mood or theme — not a title. The catalogue is searched by meaning, "
    "and constraints like *“movies from before 2000”* are understood from your sentence."
)

with st.sidebar:
    st.header("Filters")
    st.caption(
        "Optional — constraints are also read from your sentence. Anything set here "
        "overrides what was inferred."
    )

    media_label = st.radio("Type", ["Anything", "Movies", "TV shows"], horizontal=True)
    chosen_genres = st.multiselect("Genres", _genres(), help="Matches ANY of the selected")
    # A two-handle range rather than two "any"-able dropdowns: at the extremes it means
    # "no bound at all" (see below), so the common case needs no interaction.
    year_from, year_to = st.slider(
        "Release year",
        min_value=EARLIEST_YEAR,
        max_value=LATEST_YEAR,
        value=(EARLIEST_YEAR, LATEST_YEAR),
    )
    limit = st.slider("Titles to retrieve", min_value=3, max_value=10, value=5)

query = st.text_input(
    "What are you in the mood for?",
    placeholder=EXAMPLE_QUERIES[0],
)

st.caption("Try: " + " · ".join(f"*{example}*" for example in EXAMPLE_QUERIES))

if st.button("Recommend", type="primary") or query:
    if not query.strip():
        st.info("Describe a mood to get started.")
        st.stop()

    filters = {
        key: value
        for key, value in {
            "media_type": {"Movies": "movie", "TV shows": "tv"}.get(media_label),
            "genres": chosen_genres or None,
            "release_year_min": year_from if year_from > EARLIEST_YEAR else None,
            "release_year_max": year_to if year_to < LATEST_YEAR else None,
        }.items()
        if value is not None
    }

    with st.spinner("Searching the catalogue and writing a recommendation..."):
        try:
            state = _recommend(query, limit, filters)
        except ApiError as error:
            st.error(f"Something went wrong: {error}")
            st.caption(
                "Is the API running (`docker compose up -d`, or "
                "`uvicorn vibewatch.api:app`)? Check `/health` for the index status."
            )
            st.stop()

    # Show what was inferred from the sentence. A filter the system applied but never
    # displayed is indistinguishable from a bug to the person wondering where half the
    # catalogue went -- and it is also the feature's only visible proof that it worked.
    inferred = state.get("inferred_filters") or {}
    if inferred:
        st.info(
            "Understood from your request: "
            + " · ".join(f"**{key.replace('_', ' ')}** {value}" for key, value in inferred.items()),
            icon="🧠",
        )

    # Never silently ignore what the user asked for: if the filters matched nothing and
    # the graph retried without them, say so.
    if state.get("relaxed"):
        st.warning(
            "No title matched your filters, so they were dropped for this search.",
            icon="⚠️",
        )

    answer, sources = st.columns([3, 2], gap="large")

    with answer:
        st.subheader("Recommendation")
        st.write(state["answer"])

    with sources:
        st.subheader("Retrieved titles")
        st.caption("The only titles the model was allowed to recommend from.")
        for hit in state["hits"]:
            render_hit(hit)
            st.divider()
