"""Semantic reranking with sentence transformers and heuristic reranking."""

import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional

# Optional imports for semantic reranking
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    np = None
    SentenceTransformer = None


_model = None
_model_name = None
_model_lock = Lock()
_embedding_cache: Dict[str, "np.ndarray"] = {}
_cache_lock = Lock()


def tokenize_for_rerank(text: str) -> List[str]:
    """Tokenize text for heuristic reranking (similar to index tokenization)."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    # Keep underscores/hyphens as separators but also preserve them joined
    tokens = re.findall(r"[a-z0-9_]+(?:-[a-z0-9_]+)?", text)
    return tokens


def idf_weight(
    query_terms: List[str], df: Dict[str, int], N: int, floor: float = 1.5
) -> Dict[str, float]:
    """Calculate IDF weights for query terms."""
    weights = {}
    for term in query_terms:
        df_t = max(1, df.get(term, 1))
        weights[term] = max(floor, math.log((N - df_t + 0.5) / (df_t + 0.5)))
    return weights


def coverage_score(query_terms: List[str], doc_terms: List[str], idf: Dict[str, float]) -> float:
    """Score based on coverage of query terms in document."""
    present = set(query_terms) & set(doc_terms)
    if not present:
        return 0.0
    return sum(idf[t] for t in present) / (sum(idf.values()) + 1e-9)


def proximity_score(query_terms: List[str], doc_terms: List[str], window: int = 20) -> float:
    """Score based on proximity of query terms in document."""
    positions = defaultdict(list)
    for i, term in enumerate(doc_terms):
        positions[term].append(i)

    hits = [term for term in query_terms if positions.get(term)]
    if len(hits) < 2:
        return 0.0

    # Compute minimal span covering any two distinct hit terms
    best_span = None
    for i, t1 in enumerate(hits):
        for t2 in hits[i + 1 :]:
            for p1 in positions[t1]:
                # Find closest p2 to p1
                p2 = min(positions[t2], key=lambda x: abs(x - p1))
                span = abs(p2 - p1) + 1
                if best_span is None or span < best_span:
                    best_span = span

    if best_span is None:
        return 0.0
    return max(0.0, (window - best_span) / window)


def phrase_boost(query: str, doc_text: str) -> float:
    """Score boost for phrase matches in document."""
    q_tokens = tokenize_for_rerank(query)
    d_tokens = tokenize_for_rerank(doc_text)

    d = " ".join(d_tokens)

    # Check for bigrams
    bigrams = [" ".join(q_tokens[i : i + 2]) for i in range(len(q_tokens) - 1)]
    hits = sum(1 for bg in bigrams if bg in d)

    return min(1.0, 0.15 * hits)  # Up to +0.15


def heuristic_rerank(
    query: str,
    candidates: List[Dict],
    df: Optional[Dict[str, int]] = None,
    N: int = 100000,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
) -> List[Dict]:
    """
    Heuristic reranking based on coverage, proximity, and phrase matching.

    Args:
        query: Query string
        candidates: List of dicts with keys: {'text': str, 'bm25': float, 'rank': int}
        df: Dict term -> doc freq (optional; uses defaults if None)
        N: Corpus size for IDF calculation
        alpha: Weight for coverage score
        beta: Weight for proximity score
        gamma: Weight for phrase boost

    Returns:
        Candidates list with 'rerank_score' added and sorted by that score
    """
    q_terms = tokenize_for_rerank(query)
    if not q_terms:
        return candidates

    idf = idf_weight(q_terms, df or {}, N)

    for candidate in candidates:
        d_terms = tokenize_for_rerank(candidate["text"])
        cov = coverage_score(q_terms, d_terms, idf)
        prox = proximity_score(q_terms, d_terms, window=24)
        phrase = phrase_boost(query, candidate["text"])

        candidate["rerank_score"] = alpha * cov + beta * prox + gamma * phrase

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


def _normalize(vector: "np.ndarray") -> "np.ndarray":
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _load_model(model_name: str) -> Optional["SentenceTransformer"]:
    global _model, _model_name

    if SentenceTransformer is None:
        return None

    with _model_lock:
        if _model is None or _model_name != model_name:
            try:
                _model = SentenceTransformer(model_name, device="cpu")
                _model_name = model_name
            except Exception as exc:
                print(f"Failed to load sentence transformer model: {exc}")
                _model = None
                _model_name = None
        return _model


def embed_texts(
    texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Optional["np.ndarray"]:
    """
    Return normalized embeddings or None if sentence-transformers not installed.

    Lazy loads the model to keep cold-start times low.
    """

    if SentenceTransformer is None or np is None:
        return None

    model = _load_model(model_name)
    if model is None:
        return None

    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms
    except Exception as exc:
        print(f"Failed to encode texts: {exc}")
        return None


def _encode_single_text(model: "SentenceTransformer", text: str) -> "np.ndarray":
    embedding = model.encode([text], convert_to_numpy=True)[0]
    return _normalize(embedding)


def _chunk_embeddings(
    texts: List[str],
    model: "SentenceTransformer",
    max_workers: Optional[int],
) -> Optional[List["np.ndarray"]]:
    embeddings: List[Optional["np.ndarray"]] = [None] * len(texts)
    uncached = []

    with _cache_lock:
        for idx, text in enumerate(texts):
            cached = _embedding_cache.get(text)
            if cached is not None:
                embeddings[idx] = cached
            else:
                uncached.append((idx, text))

    if uncached:
        suggested = min(4, (len(uncached) or 1))
        worker_count = max(1, max_workers or suggested)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(_encode_single_text, model, text): (idx, text)
                for idx, text in uncached
            }
            for future in as_completed(future_to_idx):
                idx, text = future_to_idx[future]
                try:
                    embedding = future.result()
                except Exception as exc:
                    print(f"Failed to encode text during semantic rerank: {exc}")
                    return None
                embeddings[idx] = embedding
                with _cache_lock:
                    _embedding_cache[text] = embedding

    if any(embedding is None for embedding in embeddings):
        return None

    return embeddings  # type: ignore[return-value]


def semantic_rerank(
    query: str,
    texts: List[str],
    scores: List[float],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_workers: Optional[int] = None,
) -> List[float]:
    """
    Rerank using semantic similarity.

    Returns updated scores that combine BM25 + semantic similarity.
    If semantic reranking fails, returns original scores.
    """
    if not texts or SentenceTransformer is None:
        return scores

    if SentenceTransformer is None or np is None:
        return scores

    model = _load_model(model_name)
    if model is None:
        return scores

    query_embedding = embed_texts([query], model_name)
    if query_embedding is None:
        return scores

    chunk_embeddings = _chunk_embeddings(texts, model, max_workers)
    if chunk_embeddings is None or not chunk_embeddings:
        return scores

    # Calculate cosine similarities (already normalized)
    query_emb = query_embedding[0]
    text_embs = np.vstack(chunk_embeddings)

    similarities = text_embs.dot(query_emb).flatten()

    # Combine BM25 and semantic scores
    # Simple linear combination - can be tuned
    alpha = 0.7  # Weight for BM25 score
    beta = 0.3  # Weight for semantic similarity

    # Normalize BM25 scores to 0-1 range for combining
    if len(scores) > 1:
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            norm_scores = [1.0] * len(scores)
    else:
        norm_scores = [1.0] * len(scores)

    # Combine scores
    combined_scores = [alpha * norm_scores[i] + beta * similarities[i] for i in range(len(scores))]

    return combined_scores
