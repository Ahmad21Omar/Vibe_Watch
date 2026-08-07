"""Unit tests for the BM25 sparse vectors.

Scoring code is worth testing precisely, because it fails *quantitatively*: a wrong
exponent or a missing normalisation does not crash, it just ranks slightly worse forever.
So instead of pinning magic numbers, each test states a PROPERTY the formula must have --
rarity beats frequency, saturation flattens, longer is not automatically better -- which
is what BM25 is actually for.
"""

import math

from vibewatch.bm25 import (
    document_vectors,
    query_vector,
    token_id,
    tokenize,
)


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Mad Max: Fury Road!") == ["mad", "max", "fury", "road"]


def test_tokenize_drops_stop_words_and_single_characters():
    # "a", "of", "the" appear in nearly every plot; keeping them only inflates the index.
    assert tokenize("A story of the survivors") == ["story", "survivors"]


def test_token_id_is_stable_across_processes():
    # THE correctness requirement: ids are assigned while indexing and recomputed at query
    # time, in a different process. Python's hash() is randomised per process and would
    # make every lookup silently miss. This pins the value, so a switch to hash() breaks.
    assert token_id("survival") == 3882210509
    assert token_id("survival") == token_id("survival")


def test_a_rare_term_outweighs_a_common_one():
    # The core of IDF: a word in one document out of four is informative, a word in all
    # four is not. Both appear exactly once in the document, so only rarity separates them.
    documents = [
        "spaceship mars",
        "spaceship ocean",
        "spaceship forest",
        "spaceship desert",
    ]
    vectors = document_vectors(documents)

    assert vectors[0][token_id("mars")] > vectors[0][token_id("spaceship")]


def test_term_frequency_saturates():
    # Ten mentions must not score ten times one mention -- otherwise keyword stuffing wins.
    once, ten_times = document_vectors(["survival", " ".join(["survival"] * 10)])

    ratio = ten_times[token_id("survival")] / once[token_id("survival")]
    assert 1.0 < ratio < 3.0, "term frequency should saturate, not scale linearly"


def test_length_normalisation_penalises_padding():
    # Same single occurrence of the term, but one document is padded with unrelated words.
    # Without length normalisation both would score identically.
    short, padded = document_vectors(
        ["survival mars", "survival " + " ".join(f"filler{i}" for i in range(50))]
    )

    assert short[token_id("survival")] > padded[token_id("survival")]


def test_query_vector_weights_are_all_one():
    # All statistics live in the document vectors, so the query side stays stateless --
    # no corpus data needed at query time. If weights ever stop being 1.0, the dot product
    # is no longer BM25.
    vector = query_vector("dark survival story")

    assert set(vector.values()) == {1.0}
    assert set(vector) == {token_id(t) for t in ("dark", "survival", "story")}


def test_query_repeats_do_not_double_count():
    assert query_vector("survival survival") == query_vector("survival")


def test_query_vector_of_only_stop_words_is_empty():
    # "is it the" carries no lexical signal. An empty sparse vector means the dense half
    # decides alone -- correct behaviour, and it must not crash.
    assert query_vector("is it the") == {}


def test_dot_product_reproduces_the_bm25_score():
    # The whole design rests on this: Qdrant scores a sparse pair by dot product, so
    # document weights x query weights must equal the textbook BM25 score.
    documents = ["a story about survival on mars", "a comedy about a wedding"]
    doc_vector = document_vectors(documents)[0]
    q_vector = query_vector("survival mars")

    dot_product = sum(weight * doc_vector.get(index, 0.0) for index, weight in q_vector.items())

    # Recomputed by hand: both terms occur once, in 1 of 2 documents.
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    tokens_in_doc = 4  # story about survival mars -- "a"/"on" are stop words
    average_length = (4 + 3) / 2
    saturation = (1 * (1.5 + 1)) / (1 + 1.5 * (1 - 0.75 + 0.75 * tokens_in_doc / average_length))
    assert dot_product == 2 * idf * saturation


def test_empty_corpus_returns_no_vectors():
    assert document_vectors([]) == []
