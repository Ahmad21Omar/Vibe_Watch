"""BM25 sparse vectors -- the keyword half of hybrid search.

WHY, given that we already have semantic search.
Embeddings are great at "feels like this" and bad at "contains exactly this word". A query
naming a proper noun -- *Inception*, *Frieren*, "Tarantino" -- is a lexical question, and a
3072-dimensional average of meaning is the wrong instrument for it: the vector for a query
mentioning a rare title looks much like the vector for its whole topic. BM25 is the right
instrument, because a rare term is exactly what it weights highest.

So we keep both and fuse the two rankings (see `vector_store.hybrid_search`). Dense finds
what a query *means*, sparse finds what it *says*.

HOW BM25 WORKS, in the three ideas that matter:

1. **Term frequency, with saturation.** A word appearing 10x does not make a document 10x
   more relevant. `tf * (k1+1) / (tf + k1 * ...)` grows quickly at first, then flattens.
2. **Length normalisation.** A long plot naturally contains more words; without `b` it
   would win every match on volume alone.
3. **IDF -- rarity is information.** A term in every document ("the") tells us nothing; a
   term in three documents is a strong signal.

    score = Σ  IDF(term) · (tf · (k1+1)) / (tf + k1 · (1 - b + b · |D|/avgdl))
          terms                  └──────────── computed per document ────────────┘

DESIGN DECISION -- all statistics live in the DOCUMENT vectors.
Qdrant scores a sparse pair by dot product, so the split between the two sides is ours to
choose. The textbook split puts IDF on the query side, which means query time needs the
corpus statistics (document frequencies, average length) -- a file to compute, ship into
the Docker image, and keep in sync with the index. Instead we fold IDF into the document
weights and send query weights of 1.0. The dot product is identical, and **query time
becomes stateless**: tokenize, done.

The honest trade-off: IDF depends on the whole corpus, so adding titles later means
recomputing every document vector. For a batch-indexed catalogue that is free -- we
re-index anyway -- and it would be the wrong choice for a continuously updated index.
"""

import math
import re
import zlib
from collections import Counter

# Standard BM25 parameters. k1 controls how fast term frequency saturates, b how strongly
# length is normalised (b=0 disables it, b=1 normalises fully).
K1 = 1.5
B = 0.75

# Tokens shorter than this carry no lexical signal ("a", "of") and only add noise.
MIN_TOKEN_LENGTH = 2

# A minimal English stop-word list. The app is English-only, and these words appear in
# nearly every plot summary -- IDF would already push them towards zero, but dropping them
# keeps the vectors smaller and the index faster.
STOP_WORDS = frozenset("""
a an and are as at be been but by for from has have he her his in into is it its of on or
she that the their them they this to was were what when where which who will with you your
""".split())

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stop words and very short tokens.

    Deliberately no stemming: "survival" and "surviving" stay different tokens. A stemmer
    would help recall slightly, but the job we hired BM25 for is matching exact and rare
    terms -- titles, names -- where stemming does nothing and can only introduce errors.
    Semantic recall is the dense half's responsibility.
    """
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOP_WORDS
    ]


def token_id(token: str) -> int:
    """Map a token to the uint32 index Qdrant uses for a sparse dimension.

    CRC32, not Python's `hash()`: hash() is randomised per process (PYTHONHASHSEED), so
    the id assigned while indexing would differ from the one computed at query time and
    every lookup would silently miss. A deterministic hash is not an optimisation here,
    it is a correctness requirement.
    """
    return zlib.crc32(token.encode("utf-8"))


def _idf(document_count: int, containing: int) -> float:
    """Inverse document frequency, in the standard BM25 (probabilistic) form.

    The +0.5 terms keep it defined at the extremes; the outer 1 + ... keeps it positive
    for terms that appear in more than half the corpus, which the raw form does not.
    """
    return math.log(1 + (document_count - containing + 0.5) / (containing + 0.5))


def document_vectors(documents: list[str]) -> list[dict[int, float]]:
    """Turn a whole corpus into sparse vectors: {token_id: weight} per document.

    Takes the corpus at once rather than one document at a time -- IDF and the average
    document length are corpus-level facts, so a per-document function could not compute
    them. That is also why this belongs to indexing, not to the query path.
    """
    tokenized = [tokenize(document) for document in documents]
    document_count = len(tokenized)
    if document_count == 0:
        return []

    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / document_count or 1.0

    containing: Counter[str] = Counter()
    for tokens in tokenized:
        containing.update(set(tokens))  # set(): document frequency, not term frequency

    vectors: list[dict[int, float]] = []
    for tokens, length in zip(tokenized, lengths, strict=True):
        frequencies = Counter(tokens)
        vector: dict[int, float] = {}
        for token, frequency in frequencies.items():
            saturation = (frequency * (K1 + 1)) / (
                frequency + K1 * (1 - B + B * length / average_length)
            )
            vector[token_id(token)] = _idf(document_count, containing[token]) * saturation
        vectors.append(vector)
    return vectors


def query_vector(text: str) -> dict[int, float]:
    """Sparse vector for a query: every present term with weight 1.0.

    All the weighting already sits in the document vectors (see the module docstring), so
    the dot product Qdrant computes IS the BM25 score. Repeated query terms are counted
    once -- asking twice for the same word does not make it more important.
    """
    return {token_id(token): 1.0 for token in tokenize(text)}
