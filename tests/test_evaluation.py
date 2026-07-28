"""Unit tests for the retrieval metrics.

A metric that is subtly wrong is worse than no metric: it produces confident numbers that
send you optimizing in the wrong direction. So the edge cases -- nothing found, everything
found, position sensitivity -- are pinned explicitly.
"""

from vibewatch.evaluation import evaluate, recall_at_k, reciprocal_rank


def test_recall_counts_only_the_top_k():
    # "Alien" is retrieved, but at position 4 -- outside k=3, so it must not count.
    retrieved = ["The Road", "Fight Club", "Toy Story", "Alien"]
    assert recall_at_k(retrieved, ["The Road", "Alien"], k=3) == 0.5


def test_recall_is_one_when_every_relevant_title_is_found():
    assert recall_at_k(["The Road", "Alien"], ["Alien", "The Road"], k=5) == 1.0


def test_recall_is_zero_when_nothing_relevant_is_found():
    assert recall_at_k(["Toy Story", "Up"], ["Alien"], k=5) == 0.0


def test_recall_of_an_unlabelled_query_is_zero_not_a_crash():
    # A query nobody labelled should not blow up the whole evaluation run.
    assert recall_at_k(["Toy Story"], [], k=5) == 0.0


def test_reciprocal_rank_rewards_the_first_relevant_position():
    # First place -> 1.0, second -> 0.5. This is the metric that notices re-ranking.
    assert reciprocal_rank(["Alien", "Toy Story"], ["Alien"]) == 1.0
    assert reciprocal_rank(["Toy Story", "Alien"], ["Alien"]) == 0.5


def test_reciprocal_rank_uses_the_first_hit_only():
    # Two relevant hits do not score higher than one at the same position -- that is
    # exactly the difference to recall, and the reason we report both.
    assert reciprocal_rank(["Toy Story", "Alien", "The Road"], ["Alien", "The Road"]) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_retrieved():
    assert reciprocal_rank(["Toy Story"], ["Alien"]) == 0.0


def test_evaluate_averages_over_queries_not_over_hits():
    # Query A: perfect. Query B: nothing found, and it carries three labels. Averaging
    # over hits would let B's label count drag the score down disproportionately; over
    # queries the answer is a clean 0.5.
    results = [
        (["Alien"], ["Alien"]),
        (["Toy Story"], ["The Road", "Mad Max", "Lost"]),
    ]
    report = evaluate(results, k=5)

    assert report["queries"] == 2
    assert report["recall@5"] == 0.5
    assert report["mrr"] == 0.5


def test_evaluate_handles_an_empty_run():
    assert evaluate([], k=5) == {"queries": 0, "recall@5": 0.0, "mrr": 0.0}
