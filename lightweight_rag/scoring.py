"""BM25 scoring with proximity, n-gram, and pattern bonuses."""

import re
from difflib import SequenceMatcher
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


def gibberish_penalty(text: str, threshold: float = 0.20) -> float:
    """
    Penalize text with high gibberish ratio (line numbers, DOIs, bad OCR).
    
    Returns a penalty multiplier between 0.0 (severe penalty) and 1.0 (no penalty).
    
    Gibberish indicators:
    - Line numbers: "1480", "1481", "1482"
    - DOI patterns: "doi:10.1111/j.1467"
    - Mixed number-letter sequences: "10x", "3x", "p15"
    - Isolated numbers mixed with text
    - Excessive punctuation
    
    Args:
        text: Text to analyze
        threshold: Gibberish ratio above which to apply penalty (default 0.20 = 20%)
    
    Returns:
        Penalty multiplier: 1.0 (no penalty) down to 0.0 (max penalty)
    """
    if not text or len(text) < 20:
        return 1.0
    
    # Count gibberish patterns
    gibberish_chars = 0
    total_chars = len(text)
    
    # Pattern 1: Standalone line numbers (e.g., "1480 ", "1481 ")
    line_numbers = re.findall(r'\b\d{3,5}\b', text)
    gibberish_chars += sum(len(n) for n in line_numbers)
    
    # Pattern 2: DOI patterns
    doi_matches = re.findall(r'doi:\S+|10\.\d{4,}/\S+', text)
    gibberish_chars += sum(len(d) for d in doi_matches)
    
    # Pattern 3: Mixed number-letter patterns (bad OCR artifacts)
    mixed_patterns = re.findall(r'\b\d+[a-zA-Z]+\d*\b|\b[a-zA-Z]+\d+[a-zA-Z]*\b', text)
    # Filter out valid patterns like "p15", "2023", common abbreviations
    valid_patterns = {'p', 'pp', 'ch', 'vol', 'no', 'ed', 'v', 'n'}
    for pattern in mixed_patterns:
        # Only count as gibberish if not a common valid pattern
        if len(pattern) <= 3 and pattern.lower() not in valid_patterns:
            continue
        # Year-like patterns are OK
        if re.match(r'^[12]\d{3}$', pattern):
            continue
        gibberish_chars += len(pattern)
    
    # Pattern 4: Excessive punctuation clusters
    punct_clusters = re.findall(r'[^\w\s]{2,}', text)
    gibberish_chars += sum(len(p) for p in punct_clusters)
    
    # Pattern 5: Words that are mostly numbers
    words = text.split()
    for word in words:
        if len(word) > 2:
            digit_count = sum(c.isdigit() for c in word)
            if digit_count / len(word) > 0.5:
                gibberish_chars += len(word)
    
    # Calculate gibberish ratio
    gibberish_ratio = gibberish_chars / total_chars if total_chars > 0 else 0.0
    
    # Apply penalty if above threshold
    if gibberish_ratio <= threshold:
        return 1.0  # No penalty
    
    # Linear penalty from threshold to 50% gibberish
    # At 20% gibberish: penalty = 1.0
    # At 35% gibberish: penalty = 0.5
    # At 50%+ gibberish: penalty = 0.0
    penalty_range = 0.50 - threshold
    excess_gibberish = min(gibberish_ratio - threshold, penalty_range)
    penalty_multiplier = 1.0 - (excess_gibberish / penalty_range)
    
    return max(0.0, penalty_multiplier)


def fuzzy_match_bonus(text: str, query: str, min_length: int = 20) -> float:
    """
    Reward long exact or near-exact substring matches.
    
    Useful when searching for known quotes or exact passages. Uses Python's
    difflib.SequenceMatcher to find the longest matching substring.
    
    Args:
        text: Document text to search
        query: Query string (may be an exact quote)
        min_length: Minimum match length to consider (default 20 chars)
    
    Returns:
        Bonus score between 0.0 and 1.0:
        - 0.0: No significant match
        - 0.5: Moderate match (20-50 chars, ~80% similarity)
        - 1.0: Strong match (50+ chars, 95%+ similarity)
    """
    if not text or not query or len(query) < min_length:
        return 0.0
    
    # Normalize whitespace
    text_normalized = ' '.join(text.lower().split())
    query_normalized = ' '.join(query.lower().split())
    
    # Find longest matching substring using SequenceMatcher
    matcher = SequenceMatcher(None, text_normalized, query_normalized)
    match = matcher.find_longest_match(0, len(text_normalized), 0, len(query_normalized))
    
    if match.size < min_length:
        return 0.0
    
    # Calculate match quality
    match_length = match.size
    match_ratio = match_length / len(query_normalized)
    
    # Extract the matching substrings to check similarity
    text_match = text_normalized[match.a:match.a + match.size]
    query_match = query_normalized[match.b:match.b + match.size]
    
    # Calculate character-level similarity of the match
    similarity = SequenceMatcher(None, text_match, query_match).ratio()
    
    # Score based on both length and similarity
    # Length component (0.0 to 0.5)
    length_score = min(0.5, match_length / 100)  # Cap at 50 chars for 0.5
    
    # Similarity component (0.0 to 0.5)
    # Strong reward for very high similarity (95%+)
    if similarity >= 0.95:
        similarity_score = 0.5
    elif similarity >= 0.85:
        similarity_score = 0.3 + (similarity - 0.85) * 2.0  # 0.3 to 0.5
    elif similarity >= 0.75:
        similarity_score = 0.1 + (similarity - 0.75) * 2.0  # 0.1 to 0.3
    else:
        similarity_score = similarity * 0.1  # Up to 0.1
    
    total_score = length_score + similarity_score
    
    return min(1.0, total_score)
