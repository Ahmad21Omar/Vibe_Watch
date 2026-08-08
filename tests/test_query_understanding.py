"""Unit tests for query understanding -- no API call.

Two areas, and the second is the one that keeps the app alive:

- the PROMPT and the mapping to filter kwargs (does the catalogue vocabulary reach the
  model? does an unset constraint stay unset?);
- the FAILURE behaviour. This step sits in front of every single search, so anything it
  does badly, it does to the whole app. A hallucinated genre must be discarded, and a
  broken API call must degrade to plain search rather than take the query down with it.
"""

import datetime
import json

from vibewatch.query_understanding import QueryIntent, build_prompt, understand

CATALOGUE_GENRES = ["Comedy", "Drama", "Science Fiction"]


def _extractor(**intent_fields):
    """A fake extractor returning the given intent as the JSON the model would emit."""
    payload = {"search_text": "something", **intent_fields}
    return lambda prompt: json.dumps(payload)


# --- the prompt -------------------------------------------------------------------------


def test_prompt_lists_the_catalogues_real_genres():
    # A model left to its own vocabulary writes "Sci-Fi" while the index holds
    # "Science Fiction" -- a filter that silently matches nothing.
    prompt = build_prompt("funny stuff", CATALOGUE_GENRES)

    assert "Science Fiction" in prompt
    assert "funny stuff" in prompt


def test_prompt_anchors_relative_years_to_today():
    # "Recent" is a moving target. The current year must be in the prompt, or the model
    # anchors on whenever its training data ended.
    prompt = build_prompt("recent movies", CATALOGUE_GENRES, today=datetime.date(2031, 3, 1))

    assert "2031" in prompt
    assert "2026" in prompt  # "recent" = the last five years


# --- intent -> filter kwargs ------------------------------------------------------------


def test_unset_constraints_produce_no_filters():
    # THE default case: a pure mood must not smuggle in any filter, or every search in the
    # app quietly narrows.
    assert QueryIntent(search_text="dark survival").filters() == {}


def test_filters_include_only_what_was_asked_for():
    intent = QueryIntent(search_text="funny", media_type="movie", release_year_max=1999)

    assert intent.filters() == {"media_type": "movie", "release_year_max": 1999}


def test_an_invalid_media_type_is_dropped_rather_than_forwarded():
    # "documentary" is not one of our two media types; passing it through would filter
    # away the entire catalogue.
    assert QueryIntent(search_text="x", media_type="documentary").filters() == {}


def test_language_must_be_an_iso_code_to_be_used_as_a_filter():
    # The payload stores ISO 639-1 ("ko"). A model answering "korean" would produce a
    # filter matching zero titles -- an empty result for a request we could have served.
    assert QueryIntent(search_text="x", original_language="ko").filters() == {
        "original_language": "ko"
    }
    assert QueryIntent(search_text="x", original_language="korean").filters() == {}


# --- understand() -----------------------------------------------------------------------


def test_constraints_are_separated_from_the_mood():
    # The core promise: the year and the type leave the text, so they stop polluting the
    # embedding and start acting as real filters.
    intent = understand(
        "funny movies from before 2000",
        genres=CATALOGUE_GENRES,
        extract=_extractor(search_text="funny", media_type="movie", release_year_max=1999),
    )

    assert intent.search_text == "funny"
    assert intent.filters() == {"media_type": "movie", "release_year_max": 1999}


def test_a_hallucinated_genre_is_discarded():
    # The prompt asks for exact catalogue values, but a model can still invent one -- and
    # an invented genre matches nothing, so the user would get an empty result for a
    # perfectly good request.
    intent = understand(
        "sci-fi stuff",
        genres=CATALOGUE_GENRES,
        extract=_extractor(genres=["Sci-Fi", "Drama"]),
    )

    assert intent.genres == ["Drama"]


def test_a_failing_extractor_degrades_to_plain_search():
    # This step runs before EVERY search. If it can raise, it can take the app down.
    def boom(prompt):
        raise ConnectionError("API down")

    intent = understand("dark survival", genres=CATALOGUE_GENRES, extract=boom)

    assert intent.search_text == "dark survival"  # unchanged, as before this module
    assert intent.filters() == {}


def test_malformed_json_degrades_to_plain_search():
    intent = understand("dark survival", extract=lambda prompt: "not json at all")

    assert intent.search_text == "dark survival"
    assert intent.filters() == {}


def test_an_empty_search_text_falls_back_to_the_original_query():
    # "Comedies from the 90s" is all constraints. Embedding an empty string would search
    # for nothing at all.
    intent = understand(
        "comedies from the 90s",
        genres=CATALOGUE_GENRES,
        extract=_extractor(search_text="   ", genres=["Comedy"], release_year_min=1990),
    )

    assert intent.search_text == "comedies from the 90s"
    assert intent.filters()["genres"] == ["Comedy"]
