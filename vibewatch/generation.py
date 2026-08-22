"""The G in RAG: turn retrieved titles into a written, grounded recommendation.

Retrieval (step 4) answers *which* titles match a mood. This module answers *why* --
it asks Gemini to write the recommendation. The whole point of RAG is that the model
must reason ONLY from the titles we handed it, never from its own training memory.
Two mechanisms enforce that here:

1. The prompt is split into a static SYSTEM_INSTRUCTION (the rules and the persona) and
   a dynamic user prompt (the query + the retrieved candidates). Rules that never change
   live in the instruction, where the model weighs them most heavily; the volatile data
   stays small and separate.

2. Every candidate is rendered as a numbered, factual block via `Title.as_context_block()`
   -- the same normalized shape we indexed. The model is told these are the only titles
   that exist. If it names something else, that is a hallucination we can detect by
   comparing against the block list.

Like the rest of the codebase, the network call is injectable (`generate=`), so the
prompt-building logic can be unit-tested without spending a single API token.
"""

from google import genai
from google.genai import types

from vibewatch.config import settings
from vibewatch.gemini import call_with_retry
from vibewatch.models import Title

# A pinned GA model, not a `-preview` one and not the moving `gemini-flash-latest` alias:
# a portfolio project should give the same answer next month as it does today. Flash rather
# than Pro because the task is reasoning over text we already supply, not world knowledge.
# Verified against the API -- older names like `gemini-2.5-flash` are listed but rejected
# for new keys, the same trap as `text-embedding-004` in step 3: always probe, never assume.
LLM_MODEL = "gemini-3.5-flash"

# Low but not zero. We want the facts pinned to the context blocks (0.0-ish behaviour),
# while still allowing enough phrasing freedom that the answer reads like a person
# recommending a film rather than a database dump.
TEMPERATURE = 0.3

# What the user sees when retrieval came back empty. We do NOT ask the LLM in that case:
# with no grounding, anything it writes would be invented -- exactly what RAG prevents.
NO_RESULTS_MESSAGE = (
    "I could not find anything in the catalogue that matches that. "
    "Try describing the mood differently, or loosen the filters."
)

SYSTEM_INSTRUCTION = """You are Vibewatch, a film and TV recommender.

You will be given a user's description of what they are in the mood for, plus a numbered \
list of candidate titles retrieved from our catalogue, ordered by semantic similarity.

Rules you must follow:
1. Recommend ONLY titles from the candidate list. Never mention a film or show that is \
not in that list, not even as a comparison.
2. Use ONLY the facts given in each candidate block (plot, genres, year, rating). Do not \
add plot details, cast, or trivia from your own knowledge -- if it is not in the block, \
you do not know it.
3. Pick the 2-3 candidates that genuinely fit the request, best fit first. The list is \
ranked by similarity, not by suitability, so you may skip a higher-ranked candidate -- \
but say nothing about the ranking itself.
4. For each pick, name the title and its year, then explain in one or two sentences why \
it matches what the user asked for, referring to concrete details from its block.
5. If none of the candidates really fit, say so honestly and name the closest one anyway.
6. Write in English, warm and direct, no bullet-point headers, under 150 words. Do not \
mention retrieval, scores, candidates, or that you were given a list."""


# Created lazily so importing this module (e.g. in a unit test) never opens a connection.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.require("gemini_api_key"))
    return _client


def _call_gemini(prompt: str) -> str:
    """Send one prompt to Gemini and return the plain-text answer.

    Wrapped in the shared retry policy: a busy model (503) or a rate limit (429) is a
    temporary condition, and a user watching a spinner would much rather wait a few
    seconds than read an error. Anything else fails immediately.
    """

    def request() -> str:
        response = _get_client().models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=TEMPERATURE,
            ),
        )
        return (response.text or "").strip()

    return call_with_retry(request)


def _context_block(hit: dict) -> str:
    """Render one search hit as the factual block the LLM is allowed to reason from.

    A hit is `{"score": ..., **payload}`, and the payload is exactly a dumped Title -- so
    we can revive the model and reuse its `as_context_block()`. That keeps ONE definition
    of "how a title is described to the LLM" instead of a second, drifting copy here.
    The extra `score` key is ignored by pydantic; it is a retrieval detail, and telling the
    model about it would only invite it to talk about relevance numbers.
    """
    return Title.model_validate(hit).as_context_block()


# Prepended when the search had to give up constraints to find anything at all. Without
# it the model does what any helpful assistant does: it justifies the results it was
# handed. A real run asked for older korean thrillers, got turkish soap operas after the
# filters were dropped, and described a 2018 show as "slightly older than our newest
# releases" -- fluent, confident, and answering a question nobody asked.
RELAXED_NOTE = """IMPORTANT: nothing in the catalogue matched part of the request, so the \
search IGNORED these constraints: {dropped}.

The candidates below therefore do NOT satisfy them. Open your answer by saying plainly \
which part could not be met, then recommend from what is there. Do not pretend the \
candidates fit the ignored constraints, and do not invent reasons why they nearly do.

"""


def _describe(dropped: dict) -> str:
    """Render dropped filters the way a person would say them, for the prompt."""
    labels = {
        "genres": "genre",
        "release_year_min": "earliest year",
        "release_year_max": "latest year",
        "original_language": "original language",
        "media_type": "type",
    }
    return ", ".join(
        f"{labels.get(key, key)} = {value}" for key, value in sorted(dropped.items())
    )


def build_prompt(query: str, hits: list[dict], dropped_filters: dict | None = None) -> str:
    """Assemble the dynamic half of the prompt: the request plus the allowed titles.

    Numbering the candidates is not cosmetic -- it gives the model a way to refer to them
    internally and makes the boundary of "what exists" unmistakable.
    """
    candidates = "\n\n".join(
        f"--- CANDIDATE {number} ---\n{_context_block(hit)}"
        for number, hit in enumerate(hits, start=1)
    )
    note = RELAXED_NOTE.format(dropped=_describe(dropped_filters)) if dropped_filters else ""
    return (
        f"{note}"
        f'The user is in the mood for: "{query}"\n\n'
        f"Candidate titles from our catalogue:\n\n{candidates}"
    )


def generate_recommendation(
    query: str,
    hits: list[dict],
    *,
    dropped_filters: dict | None = None,
    generate=_call_gemini,
) -> str:
    """Write the recommendation for `query`, grounded in `hits`.

    `dropped_filters` names the constraints the search had to abandon to find anything.
    Passing them in is what lets the answer be honest about what it could not do -- an
    answer that quietly ignores a stated constraint is a subtler failure than an empty
    result, because it looks like success.

    `generate` is injectable so tests can assert on the prompt we build without calling
    the API. Empty `hits` short-circuits: no context means no grounded answer is possible.
    """
    if not hits:
        return NO_RESULTS_MESSAGE
    return generate(build_prompt(query, hits, dropped_filters))
