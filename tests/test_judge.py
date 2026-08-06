"""Unit tests for the faithfulness judge -- no API call, no quota.

Two things are worth pinning here, and they are different in kind:

- the SCORING arithmetic (supported / total, and the empty-claims edge case), which is
  pure logic and must simply be right;
- the PROMPT, because it is the whole instrument. If the context blocks or the answer
  went missing from it, the judge would score a text it never saw and hand back a
  confident number -- the worst failure mode an evaluation can have.
"""

import json

from vibewatch.judge import (
    build_judge_prompt,
    format_report,
    score_faithfulness,
    unsupported_claims,
)


def _hit(title="The Martian", year=2015, **overrides):
    return {
        "score": 0.7,
        "tmdb_id": 286217,
        "media_type": "movie",
        "title": title,
        "overview": "An astronaut is stranded on Mars and must find a way to survive.",
        "original_language": "en",
        "genres": ["Science Fiction", "Drama"],
        "release_year": year,
        "popularity": 30.0,
        "vote_average": 7.7,
        "vote_count": 16000,
        "poster_path": None,
        **overrides,
    }


def _judge_returning(*claims):
    """A fake judge that replies with the given (text, supported) pairs as JSON."""

    def _judge(prompt: str) -> str:
        return json.dumps(
            {
                "claims": [
                    {"claim": text, "supported": supported, "reason": "because"}
                    for text, supported in claims
                ]
            }
        )

    return _judge


def test_prompt_contains_the_sources_and_the_answer():
    prompt = build_judge_prompt([_hit()], "Watch The Martian (2015).")

    assert "--- SOURCE 1 ---" in prompt
    assert "stranded on Mars" in prompt  # the block's facts
    assert "Watch The Martian (2015)." in prompt  # the text under judgement


def test_prompt_numbers_every_source():
    prompt = build_judge_prompt([_hit(), _hit("Interstellar", 2014)], "...")

    assert "--- SOURCE 1 ---" in prompt
    assert "--- SOURCE 2 ---" in prompt
    assert "Interstellar" in prompt


def test_all_claims_supported_scores_one():
    result = score_faithfulness(
        [_hit()],
        "Watch The Martian.",
        judge=_judge_returning(("It is from 2015", True), ("Set on Mars", True)),
    )

    assert result["faithfulness"] == 1.0
    assert unsupported_claims(result) == []


def test_score_is_the_share_of_supported_claims():
    result = score_faithfulness(
        [_hit()],
        "Watch The Martian, starring Matt Damon.",
        judge=_judge_returning(
            ("It is from 2015", True),
            ("Set on Mars", True),
            ("It stars Matt Damon", False),  # true in reality, absent from the block
        ),
    )

    assert result["faithfulness"] == 2 / 3
    # The point of the judge is not the number but this list: it names what to fix.
    assert unsupported_claims(result) == ["It stars Matt Damon"]


def test_an_answer_without_checkable_claims_is_not_a_division_by_zero():
    # "Nothing here really fits" makes no factual claim. Scoring it 0.0 would punish the
    # most honest possible answer.
    result = score_faithfulness([_hit()], "Nothing here really fits.", judge=_judge_returning())

    assert result["faithfulness"] == 1.0


def test_report_lists_the_mean_and_every_rejected_claim():
    results = [
        {"faithfulness": 1.0, "claims": []},
        {
            "faithfulness": 0.5,
            "claims": [
                {"claim": "ok", "supported": True, "reason": ""},
                {"claim": "invented detail", "supported": False, "reason": ""},
            ],
        },
    ]

    report = format_report(results)

    assert "0.750" in report  # mean of 1.0 and 0.5
    assert "invented detail" in report
