"""BM25 scoring with proximity, n-gram, and pattern bonuses."""

from typing import List

from .index import tokenize
from .models import ANSWER_PATTERNS


def proximity_bonus(text: str, query_tokens: List[str], window_size: int = 30) -> float:
    """
    Reward when distinct query tokens co-occur within a short window.

    Returns a score between 0 and 1 based on how close query terms appear together.
    """
    if len(query_tokens) < 2:
        return 0.0

    # Tokenize the text and find positions of query tokens
    text_tokens = text.lower().split()
    if len(text_tokens) < 2:
        return 0.0

    # Find positions of query tokens
    positions = {}
    for i, token in enumerate(text_tokens):
        if token in [qt.lower() for qt in query_tokens]:
            if token not in positions:
                positions[token] = []
            positions[token].append(i)

    # Need at least 2 distinct query terms present
    if len(positions) < 2:
        return 0.0

    # Check for co-occurrence within window
    max_score = 0.0
    token_list = list(positions.keys())

    for i in range(len(token_list)):
        for j in range(i + 1, len(token_list)):
            token1, token2 = token_list[i], token_list[j]

            # Check all position pairs
            for pos1 in positions[token1]:
                for pos2 in positions[token2]:
                    distance = abs(pos1 - pos2)
                    if distance <= window_size:
                        # Closer = higher score
                        score = max(0, 1.0 - (distance / window_size))
                        max_score = max(max_score, score)

    return max_score


def ngram_bonus(text: str, query: str, max_hits: int = 6) -> float:
    """
    N-gram (bi/tri) contiguous phrase bonus.

    Looks for bigrams and trigrams from query appearing in text.
    """
    t = " ".join(text.lower().split())
    toks = tokenize(query)

    bigrams = [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]
    trigrams = [" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)]

    hits = 0
    for ng in bigrams + trigrams:
        if len(ng) >= 5 and ng in t:
            hits += 1
            if hits >= max_hits:
                break

    return min(hits, max_hits) / max_hits


def pattern_bonus(text: str) -> float:
    """
    Bonus for academic/definition patterns like "is a", "we define", etc.
    """
    text_lower = text.lower()
    return sum(1 for p in ANSWER_PATTERNS if p in text_lower) * 0.05
