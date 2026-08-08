"""Turn a free-text request into a mood plus hard filters.

THE GAP THIS CLOSES.
"Funny movies from before 2000" is two different requests wearing one sentence: a mood
("funny") and two hard constraints (movies only, released before 2000). Until now only the
sidebar could express constraints, so typing them achieved nothing -- worse than nothing,
actually: the words "movies from before 2000" went into the embedding and diluted it, so
the query matched titles *about* the year 2000 instead of titles *from* it.

This module splits the sentence:

    "funny movies from before 2000"
        -> search_text        "funny"
        -> media_type         "movie"
        -> release_year_max   1999

The mood goes to the vector search, the constraints go to the Qdrant filter, and each
mechanism gets the part of the question it is actually good at (see the retrieval section
of the README on why hard constraints do not belong in a vector).

THREE DECISIONS WORTH DEFENDING:

1. **The genre list comes from the catalogue, not from the model.** The available genres
   are passed into the prompt, because a model left to itself writes "Sci-Fi" while the
   index contains "Science Fiction" -- a filter that silently matches nothing. Grounding
   the vocabulary is the same principle as grounding the answer.

2. **`search_text` is returned separately, not just stripped of a few words.** What is
   left after removing the constraints is what should be embedded. Sending the original
   sentence would put the filter words back into the vector.

3. **Failure is not fatal.** If the model errors or returns nonsense, we fall back to
   "no filters, embed the whole sentence" -- exactly the old behaviour. A query
   understanding step that can take the app down is a bad trade for a convenience feature.

COST: one extra LLM call per query, before retrieval even starts. That is real latency
(~1s) and real quota. It is worth it because the alternative is a request the system
cannot answer at all -- but it is the reason this runs once per query and not per hit.
"""

import datetime

from pydantic import BaseModel, Field

# Deterministic extraction: the same sentence must produce the same filters every time.
# Unlike the recommendation text, there is nothing creative to do here.
EXTRACTION_TEMPERATURE = 0.0

PROMPT = """Extract the search intent from a user's request for something to watch.

Split the request into a MOOD (what it should feel like or be about) and HARD CONSTRAINTS
(type, genre, release year). Return them separately.

Rules:
- search_text: only the mood/theme/plot part, with the constraints removed. If the request
  is nothing but constraints ("comedies from the 90s"), use the genre or topic words as
  the mood ("comedy").
- media_type: "movie" or "tv", ONLY if the user clearly asked for one. "Films" or "movies"
  -> movie; "series", "shows", "anime" -> tv. If unclear, leave it out.
- genres: ONLY values from this exact list, copied character for character:
  {genres}
  Leave empty if the user described a mood rather than naming a genre.
- release_year_min / release_year_max: inclusive bounds. Convert relative wording using
  the current year, {year}: "from the 90s" -> 1990 to 1999; "before 2000" -> max 1999;
  "recent" or "new" -> min {recent_year}; "classic" or "old" -> max 1990.
- original_language: the ISO 639-1 code of the language the title was MADE in, only when
  the user names an origin: "korean" -> "ko", "anime" or "japanese" -> "ja",
  "bollywood" -> "hi". Not the language they want to watch in.
- Leave any field out when the user did not ask for it. Do not guess. An invented filter
  removes correct results the user wanted.

Request: {query}"""


class QueryIntent(BaseModel):
    """The structured form of a request. Every constraint is optional by design."""

    search_text: str = Field(description="the mood/theme part, constraints removed")
    media_type: str | None = None
    genres: list[str] = Field(default_factory=list)
    release_year_min: int | None = None
    release_year_max: int | None = None
    original_language: str | None = None

    def filters(self) -> dict:
        """The constraints as retrieve()-compatible kwargs, omitting anything unset."""
        candidates = {
            "media_type": self.media_type if self.media_type in {"movie", "tv"} else None,
            "genres": self.genres or None,
            "release_year_min": self.release_year_min,
            "release_year_max": self.release_year_max,
            # Two letters or nothing: the payload stores ISO 639-1 codes, so a model
            # answering "korean" instead of "ko" would filter the catalogue down to zero.
            "original_language": (
                self.original_language
                if self.original_language and len(self.original_language) == 2
                else None
            ),
        }
        return {key: value for key, value in candidates.items() if value is not None}


def build_prompt(query: str, genres: list[str], today: datetime.date | None = None) -> str:
    """Assemble the extraction prompt, with the catalogue's real genres baked in."""
    year = (today or datetime.date.today()).year
    return PROMPT.format(
        query=query,
        genres=", ".join(genres) if genres else "(none available)",
        year=year,
        # "Recent" is a moving target; deriving it from today keeps the prompt honest
        # instead of hardcoding a year that quietly ages.
        recent_year=year - 5,
    )


def _call_gemini(prompt: str) -> str:
    from google.genai import types

    from vibewatch.gemini import call_with_retry
    from vibewatch.generation import LLM_MODEL, _get_client

    def request() -> str:
        response = _get_client().models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=EXTRACTION_TEMPERATURE,
                response_mime_type="application/json",
                response_schema=QueryIntent,
            ),
        )
        return response.text or "{}"

    return call_with_retry(request)


def understand(
    query: str,
    *,
    genres: list[str] | None = None,
    extract=_call_gemini,
    today: datetime.date | None = None,
) -> QueryIntent:
    """Parse `query` into a QueryIntent. Never raises -- falls back to the plain query.

    The fallback is the whole safety story: whatever goes wrong (API down, malformed
    JSON, a hallucinated genre), the caller still gets a usable intent that searches the
    way the app did before this module existed.
    """
    try:
        intent = QueryIntent.model_validate_json(
            extract(build_prompt(query, genres or [], today))
        )
    except Exception:
        return QueryIntent(search_text=query)

    # Drop genres the catalogue does not have. The prompt asks for exact values, but a
    # model can still invent one, and an invented genre filters away everything.
    if genres is not None:
        intent.genres = [genre for genre in intent.genres if genre in genres]

    # An empty search_text would embed nothing at all; the original query is the safer
    # thing to search for.
    if not intent.search_text.strip():
        intent.search_text = query

    return intent
