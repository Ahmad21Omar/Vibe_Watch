"""Unit tests for the generation step -- no Gemini call, no quota spent.

What actually matters here is the PROMPT: it is the only thing standing between a grounded
recommendation and a hallucinated one. So we build prompts from fake hits and assert that
the user's words, the retrieved facts, and the "these are the only titles" framing all
arrive intact -- and that the network call receives exactly that text.
"""

from vibewatch.generation import (
    NO_RESULTS_MESSAGE,
    build_prompt,
    generate_recommendation,
)


def _hit(title, score=0.9, **overrides):
    """A search hit as retrieval produces it: score + a dumped Title payload."""
    payload = {
        "tmdb_id": 550,
        "media_type": "movie",
        "title": title,
        "overview": "An insomniac office worker forms an underground fight club.",
        "original_language": "en",
        "genres": ["Drama", "Thriller"],
        "release_year": 1999,
        "popularity": 61.4,
        "vote_average": 8.4,
        "vote_count": 27000,
        "poster_path": "/poster.jpg",
    }
    return {"score": score, **payload, **overrides}


def test_prompt_contains_the_users_words():
    # The mood description is the whole request -- paraphrasing or dropping it would let
    # the model recommend by genre alone.
    prompt = build_prompt("dark survival stories", [_hit("Fight Club")])
    assert "dark survival stories" in prompt


def test_prompt_renders_every_hit_as_a_numbered_candidate():
    prompt = build_prompt("dark", [_hit("Fight Club"), _hit("Alien"), _hit("The Road")])

    assert "--- CANDIDATE 1 ---" in prompt
    assert "--- CANDIDATE 3 ---" in prompt
    # All three must be offered; silently truncating the list would waste retrieval work.
    for title in ("Fight Club", "Alien", "The Road"):
        assert title in prompt


def test_candidate_blocks_carry_the_grounding_facts():
    # The model may only use what is in the block. If year, genres, rating or plot went
    # missing, it would have to fall back on its training memory -- i.e. hallucinate.
    prompt = build_prompt("dark", [_hit("Fight Club")])

    assert "1999" in prompt
    assert "Drama, Thriller" in prompt
    assert "8.4/10" in prompt
    assert "insomniac office worker" in prompt


def test_prompt_hides_the_similarity_score():
    # Scores are a retrieval implementation detail. Leaking them invites the model to
    # discuss relevance numbers instead of the films.
    prompt = build_prompt("dark", [_hit("Fight Club", score=0.6576)])
    assert "0.6576" not in prompt


def test_generate_passes_the_built_prompt_to_the_model():
    seen = {}

    def fake_generate(prompt: str) -> str:
        seen["prompt"] = prompt
        return "Watch Fight Club (1999)."

    hits = [_hit("Fight Club")]
    answer = generate_recommendation("dark survival", hits, generate=fake_generate)

    assert answer == "Watch Fight Club (1999)."
    assert seen["prompt"] == build_prompt("dark survival", hits)


def test_no_hits_returns_honest_message_without_calling_the_model():
    # With nothing retrieved there is no grounding, so any generated text would be
    # invented. Short-circuiting is both safer and free.
    def fail_if_called(prompt: str) -> str:
        raise AssertionError("the LLM must not be called without context")

    assert (
        generate_recommendation("dark", [], generate=fail_if_called) == NO_RESULTS_MESSAGE
    )
