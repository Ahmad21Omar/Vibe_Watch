"""LLM-as-judge: does the written recommendation actually follow from its sources?

Retrieval quality is measured in `evaluation.py` with recall@k and MRR -- deterministic
numbers over hand-labelled queries. The generation side cannot be scored that way: there
is no single correct paragraph, so "is this text good?" has to be judged, not computed.

The property worth judging is **faithfulness**: every factual claim in the answer must be
supported by the context blocks we handed the model. That is precisely what RAG promises,
and precisely what fails silently -- a hallucinated plot detail reads exactly like a real
one.

METHOD (the standard decomposition, also used by RAGAS):
  1. break the answer into atomic factual claims,
  2. check each claim against the context ALONE -- not against world knowledge,
  3. faithfulness = supported claims / total claims.

Why hand-rolled instead of installing RAGAS: it is ~80 lines against one API call, it
keeps the prompt visible and tunable, and it avoids pulling a large LangChain dependency
into a project that deliberately talks to Gemini directly. The trade-off is honest --
RAGAS is better validated and has more metrics; this is enough to catch a regression.

A judge is itself an LLM and therefore fallible: it can miss a subtle contradiction or
flag a harmless paraphrase. Treat the score as a comparable signal, not as truth.
"""

import json

from pydantic import BaseModel

from vibewatch.generation import _context_block

# The judge must be conservative and literal, so a low temperature matters more here than
# in generation: we want the same verdict on the same input, not creative reading.
JUDGE_TEMPERATURE = 0.0

JUDGE_PROMPT = """You are evaluating whether a recommendation is faithful to its sources.

Below are the SOURCE BLOCKS that were given to a recommender, followed by the ANSWER it \
produced. Your job is to check the answer against the sources -- and against nothing else.

Step 1: break the answer into atomic factual claims. A claim is one checkable statement, \
e.g. "The Martian is from 2015" or "Interstellar involves space travel". Ignore pure \
opinion or address to the reader ("you will love it", "if you are in the mood for...").

Step 2: for each claim, decide whether the SOURCE BLOCKS support it.
- supported = the claim follows from the blocks, including reasonable paraphrase.
- unsupported = the blocks do not say it, EVEN IF you personally know it is true. A claim \
that is correct in the real world but absent from the blocks is unsupported: the \
recommender was not allowed to know it.
- A title that does not appear in the blocks at all makes every claim about it unsupported.

SOURCE BLOCKS:
{context}

ANSWER:
{answer}"""


class Claim(BaseModel):
    claim: str
    supported: bool
    reason: str


class Verdict(BaseModel):
    """The judge's structured reply. A schema beats free text: no parsing guesswork."""

    claims: list[Claim]


def build_judge_prompt(hits: list[dict], answer: str) -> str:
    """Assemble the judging prompt from the same context blocks generation received."""
    context = "\n\n".join(
        f"--- SOURCE {number} ---\n{_context_block(hit)}"
        for number, hit in enumerate(hits, start=1)
    )
    return JUDGE_PROMPT.format(context=context, answer=answer)


def _call_judge(prompt: str) -> str:
    """Ask Gemini for a structured verdict. Imported lazily to keep this module importable."""
    from google.genai import types

    from vibewatch.gemini import call_with_retry
    from vibewatch.generation import LLM_MODEL, _get_client

    def request() -> str:
        response = _get_client().models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=JUDGE_TEMPERATURE,
                response_mime_type="application/json",
                response_schema=Verdict,
            ),
        )
        return response.text or "{}"

    return call_with_retry(request)


def score_faithfulness(hits: list[dict], answer: str, *, judge=_call_judge) -> dict:
    """Return {"faithfulness": float, "claims": [...]} for one answer.

    An answer with no checkable claims scores 1.0 rather than dividing by zero: there is
    nothing unsupported in it. That is a judgement call, and it is the safe one -- the
    alternative (0.0) would punish a short, honest "nothing here really fits".
    """
    verdict = Verdict.model_validate_json(judge(build_judge_prompt(hits, answer)))
    claims = verdict.claims

    if not claims:
        return {"faithfulness": 1.0, "claims": []}

    supported = sum(claim.supported for claim in claims)
    return {
        "faithfulness": supported / len(claims),
        "claims": [claim.model_dump() for claim in claims],
    }


def unsupported_claims(result: dict) -> list[str]:
    """The claims the judge rejected -- what you actually read when a score drops."""
    return [claim["claim"] for claim in result["claims"] if not claim["supported"]]


def format_report(results: list[dict]) -> str:
    """One-line-per-query summary plus the mean, for the evaluation script."""
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(f"  query {index}: faithfulness {result['faithfulness']:.2f}")
        for claim in unsupported_claims(result):
            lines.append(f"      unsupported: {claim}")
    mean_score = sum(r["faithfulness"] for r in results) / len(results) if results else 0.0
    lines.append(f"\n  mean faithfulness  {mean_score:.3f}  over {len(results)} answers")
    return "\n".join(lines)


def as_json(results: list[dict]) -> str:
    """Full verdicts, for inspecting exactly what the judge objected to."""
    return json.dumps(results, indent=2)
