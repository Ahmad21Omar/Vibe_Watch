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

import os

import streamlit as st

# --- secrets bridge, before anything reads configuration --------------------------------
# Streamlit Community Cloud hands secrets to the app through `st.secrets` (a TOML blob),
# NOT as environment variables. Our settings are pydantic-settings, which reads the
# environment -- and does so at IMPORT time. So the values have to be in os.environ before
# the first `vibewatch` import, which is why this sits above them and they carry a noqa.
# `setdefault`: a real environment variable (Docker, a local shell) still wins.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    # No secrets.toml locally -- that is the normal case; .env covers it.
    pass

from vibewatch import client
from vibewatch.client import ApiError
from vibewatch.config import settings

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


def _code_version() -> str:
    """When this file was last written -- i.e. which deployment is actually running.

    A hosted app gives you no way to ask "did my fix reach production?", and guessing from
    the UI costs an hour. This answers it in one line.
    """
    import datetime
    import pathlib

    stamp = pathlib.Path(__file__).stat().st_mtime
    return datetime.datetime.fromtimestamp(stamp, datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")


@st.cache_resource(show_spinner="Starting the API...")
def _ensure_api() -> str | None:
    """On single-process hosts, bring the API up inside this process. Once per process.

    `cache_resource` is what makes it once: Streamlit re-runs this script on every
    interaction, and without the cache each click would try to bind port 8000 again.
    """
    if not settings.embedded_api:
        return None
    from vibewatch.embedded import serve_api_in_background

    return serve_api_in_background()


_ensure_api()


@st.cache_data(show_spinner=False)
def _genres() -> list[str]:
    """The catalogue's real genres. Cached: they change only when we re-index."""
    try:
        return client.genres()
    except ApiError:
        # A dead API should not blank the page -- the query below will report it properly.
        return []


@st.cache_data(show_spinner=False)
def _recommend(query: str, limit: int, filters: dict, history: tuple[str, ...] = ()) -> dict:
    """Cached API call.

    Keyed on the query, the filters AND the history -- the last one matters: "something
    shorter" means something different after "korean series" than after "space
    documentaries", so caching on the query alone would serve the wrong answer. (A tuple
    because the key has to be hashable.)

    Every uncached call costs one embedding of the daily quota plus two LLM calls
    server-side, so re-rendering (a slider, a resize) staying free is worth the care.
    """
    return client.recommend(query, limit=limit, history=list(history), **filters)


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

    with st.expander("Diagnostics"):
        # Every deployment failure so far has come down to one of these three lines, and
        # from outside a hosted app there is no other way to see them. Names and booleans
        # only -- never a value, because this panel is as public as the app.
        st.caption(f"code loaded: {_code_version()}")
        st.caption(f"embedded API: {settings.embedded_api}  ·  API at {settings.api_url}")
        configured = [
            name
            for name in ("GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "EMBEDDED_API")
            if os.environ.get(name)
        ]
        st.caption("configuration found: " + (", ".join(configured) or "none"))
        try:
            client.genres()
            st.caption("API: reachable ✓")
        except ApiError as error:
            st.caption(f"API: unreachable — {error}")


def render_turn(turn: dict) -> None:
    """One exchange: what was asked, what was inferred, the answer and its sources."""
    with st.chat_message("user"):
        st.write(turn["query"])

    with st.chat_message("assistant"):
        # Show what was inferred from the sentence. A filter the system applied but never
        # displayed is indistinguishable from a bug to the person wondering where half the
        # catalogue went -- and it is the feature's only visible proof that it worked.
        inferred = turn["state"].get("inferred_filters") or {}
        if inferred:
            st.info(
                "Understood from your request: "
                + " · ".join(
                    f"**{key.replace('_', ' ')}** {value}" for key, value in inferred.items()
                ),
                icon="🧠",
            )

        # Never silently ignore what the user asked for. Naming the SPECIFIC constraint
        # that was given up matters: "your filters were dropped" leaves the user guessing
        # which part of their request survived, and the search only ever gives up as
        # little as it can.
        dropped = turn["state"].get("dropped_filters") or {}
        if dropped:
            st.warning(
                "Nothing matched all of your request, so this search ignored: "
                + " · ".join(
                    f"**{key.replace('_', ' ')}** {value}" for key, value in dropped.items()
                ),
                icon="⚠️",
            )
        elif turn["state"].get("relaxed"):
            st.warning(
                "No title matched your filters, so they were dropped for this search.",
                icon="⚠️",
            )

        answer, sources = st.columns([3, 2], gap="large")
        with answer:
            st.write(turn["state"]["answer"])
        with sources:
            st.caption("The only titles the model was allowed to recommend from.")
            for hit in turn["state"]["hits"]:
                render_hit(hit)
                st.divider()


# The conversation lives in the SESSION, not on the server. That is what keeps the API
# stateless: no sticky sessions, no shared store, and a restart of the service drops
# nothing. Each request carries the history it wants to be understood against.
if "turns" not in st.session_state:
    st.session_state.turns = []

for past_turn in st.session_state.turns:
    render_turn(past_turn)

if not st.session_state.turns:
    st.caption("Try: " + " · ".join(f"*{example}*" for example in EXAMPLE_QUERIES))

query = st.chat_input("What are you in the mood for?")

if query and query.strip():
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

    # Only the user's own words -- the generated answers are long, would dominate the
    # prompt, and what a follow-up refers to is what was ASKED, not what we replied.
    history = [past["query"] for past in st.session_state.turns]

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Searching the catalogue and writing a recommendation..."):
        try:
            state = _recommend(query, limit, filters, tuple(history))
        except ApiError as error:
            st.error(f"Something went wrong: {error}")
            # Advice that fits where the app is actually deployed. Telling someone on a
            # single-process host to run `docker compose up` is worse than saying nothing:
            # it sends them looking for a shell that does not exist. The one configuration
            # mistake that produces exactly this failure is naming it explicitly.
            if not settings.embedded_api:
                st.caption(
                    "**`EMBEDDED_API` is not set.** On a host that runs a single process "
                    "(Streamlit Community Cloud), add `EMBEDDED_API = \"true\"` to the "
                    "app's secrets so the API starts inside this process. Running the API "
                    "separately instead? Then `API_URL` must point at it — it is currently "
                    f"`{settings.api_url}`."
                )
            else:
                st.caption(
                    "The API is embedded in this process, so this is most likely the index "
                    "or the model: check `/health`, and that `QDRANT_URL` (with `:6333`), "
                    "`QDRANT_API_KEY` and `GEMINI_API_KEY` are set."
                )
            st.stop()

    st.session_state.turns.append({"query": query, "state": state})
    # Re-run so the new turn is rendered by render_turn() like every other one, instead
    # of being drawn twice by two slightly different code paths.
    st.rerun()

if st.session_state.turns and st.sidebar.button("Start a new conversation"):
    st.session_state.turns = []
    st.rerun()
