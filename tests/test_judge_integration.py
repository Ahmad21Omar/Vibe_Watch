"""Live test of the faithfulness judge -- and, more importantly, OF the judge.

An evaluator that always answers "looks fine" is worse than no evaluator: it produces a
confident number that hides regressions. So the second test here is the one that matters.
It feeds the judge a deliberately hallucinated answer and requires it to notice. If that
ever starts passing at a high score, every other faithfulness number in the project
becomes meaningless.

Marked `integration`: two LLM calls per run. Deselected by default, run with
`pytest -m integration`.
"""

import pytest

from vibewatch.graph import recommend
from vibewatch.judge import score_faithfulness
from vibewatch.retrieval import retrieve

pytestmark = pytest.mark.integration

QUERY = "survival in space, one person against impossible odds"


def test_a_real_recommendation_is_faithful_to_its_sources(live_services):
    state = recommend(QUERY, limit=3)

    result = score_faithfulness(state["hits"], state["answer"])

    # Not == 1.0: the judge is an LLM and may quibble over a paraphrase. What must hold is
    # that a genuine, grounded answer scores high -- a real regression drops far below.
    assert result["faithfulness"] >= 0.8, result["claims"]


def test_the_judge_rejects_a_hallucinated_answer(live_services):
    # This validates the INSTRUMENT. Three planted claims, three different failure modes:
    hits = retrieve(QUERY, limit=3)
    hallucinated = (
        "You should watch Titanic (1997), a sweeping romance about a doomed ocean liner. "
        "Also try The Martian (2015), which stars Matt Damon and won four Oscars."
    )

    result = score_faithfulness(hits, hallucinated)

    # - a title that was never retrieved at all,
    # - a detail that is TRUE in the world but absent from the context (the subtle one:
    #   for RAG it is still a hallucination),
    # - a detail that is simply false.
    assert result["faithfulness"] <= 0.5, result["claims"]
