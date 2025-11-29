"""Reciprocal Rank Fusion (RRF) and multi-run ranking system."""

import re
from typing import Dict, List, Optional, Tuple

from .index import tokenize
from .models import Chunk
from .rerank import semantic_rerank
from .scoring import ngram_bonus, pattern_bonus, proximity_bonus


def rrf_fuse(runs: List[List[int]], C: int = 60, cap: int = 200) -> List[int]:
    """
    Fuse multiple ranked lists of corpus indices using Reciprocal Rank Fusion.

    Args:
        runs: List of ranked lists of corpus indices
        C: RRF parameter (default 60 is common)
        cap: Trim each run to top-N to keep it focused

    Returns:
        Fused list of indices, best first
    """
    score: Dict[int, float] = {}
    for run in runs:
        for rank, idx in enumerate(run[:cap], start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (C + rank)
    return [i for i, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]


def robustify_query(query: str) -> str:
    """
    Create a robust variant of query to catch minor phrasing drift.

    Args:
        query: Original query string

    Returns:
        Robustified query (lowercase, punctuation normalized)
    """
    # Lowercase; strip punctuation that fractures phrases but keep alphanumerics
    q2 = re.sub(r"[^A-Za-z0-9\s]", " ", query.lower())
    q2 = re.sub(r"\s+", " ", q2).strip()
    return q2


def rank_by_bm25_order(
    bm25, query_str: str, pool: List[int], tokenized: List[List[str]]
) -> List[int]:
    """
    Re-rank a pool of candidates using BM25 scores for a given query.

    Args:
        bm25: BM25 model
        query_str: Query string
        pool: List of candidate indices to rank
        tokenized: Tokenized corpus (for compatibility)

    Returns:
        Pool indices sorted by BM25 score (best first)
    """
    q_tokens = tokenize(query_str)
    all_scores = bm25.get_scores(q_tokens)
    return sorted(pool, key=lambda i: all_scores[i], reverse=True)


def build_ranking_runs(
    query: str,
    corpus: List[Chunk],
    bm25,
    tokenized: List[List[str]],
    pool: List[int],
    baseline_scores: List[float],
    config: Dict,
    semantic_query: Optional[str] = None,
) -> List[List[int]]:
    """
    Build multiple ranking runs over the same candidate pool.

    Args:
        query: Search query
        corpus: Document corpus
        bm25: BM25 model
        tokenized: Tokenized corpus
        pool: Candidate pool indices
        baseline_scores: Baseline BM25+bonuses scores
        config: Configuration dictionary
        semantic_query: Optional query to use for semantic reranking (e.g. HyDE expansion)

    Returns:
        List of ranking runs (each is a list of indices)
    """
    runs = []

    # Run A: Baseline BM25 with bonuses (already computed)
    run_base = sorted(pool, key=lambda i: baseline_scores[i], reverse=True)
    runs.append(run_base)

    # Run B: RM3 pseudo-relevance feedback (optional)
    if config.get("prf", {}).get("enabled", False):
        from .prf import rm3_expand_query

        expanded = rm3_expand_query(
            query,
            bm25,
            tokenized,
            corpus,
            fb_docs=config["prf"]["fb_docs"],
            fb_terms=config["prf"]["fb_terms"],
            alpha=config["prf"]["alpha"],
        )
        if expanded != query:
            run_rm3 = rank_by_bm25_order(bm25, expanded, pool, tokenized)
            runs.append(run_rm3)

    # Run C: Heuristic reranking (lightweight)
    if config.get("rerank", {}).get("heuristic", {}).get("enabled", True):
        from .rerank import heuristic_rerank

        # Prepare candidates for heuristic reranking
        heuristic_topn = min(
            config.get("rerank", {}).get("heuristic", {}).get("topn", 150), len(pool)
        )
        candidates_for_heuristic = pool[:heuristic_topn]

        candidate_dicts = []
        for i in candidates_for_heuristic:
            candidate_dicts.append(
                {
                    "text": corpus[i].text,
                    "bm25": baseline_scores[i],
                    "rank": candidates_for_heuristic.index(i),
                    "index": i,
                }
            )

        # Apply heuristic reranking
        reranked_candidates = heuristic_rerank(query, candidate_dicts)

        # Extract reranked ordering
        run_heuristic = [c["index"] for c in reranked_candidates]
        runs.append(run_heuristic)

    # Run D: Semantic reranking (optional, heavier)
    if config.get("rerank", {}).get("semantic", {}).get("enabled", False):
        semantic_config = config["rerank"]["semantic"]
        topn = min(semantic_config.get("topn", 120), len(pool))
        candidates_for_semantic = pool[:topn]

        candidate_texts = [corpus[i].text for i in candidates_for_semantic]
        candidate_scores = [baseline_scores[i] for i in candidates_for_semantic]

        # Semantic rerank returns new scores; we need to re-sort
        # Use semantic_query if provided (e.g. HyDE), otherwise original query
        query_for_semantic = semantic_query if semantic_query else query
        reranked_scores = semantic_rerank(query_for_semantic, candidate_texts, candidate_scores)

        # Create mapping from index to new score
        score_map = {
            candidates_for_semantic[i]: reranked_scores[i]
            for i in range(len(candidates_for_semantic))
        }

        # Sort pool by semantic scores (candidates not in semantic keep original scores)
        run_semantic = sorted(
            pool, key=lambda i: score_map.get(i, baseline_scores[i]), reverse=True
        )
        runs.append(run_semantic)

    # Run E: Robust query variant (optional)
    if config.get("fusion", {}).get("robust_query", {}).get("enabled", True):
        robust_q = robustify_query(query)
        if robust_q != query.lower().strip():
            run_robust = rank_by_bm25_order(bm25, robust_q, pool, tokenized)
            runs.append(run_robust)

    return runs


def fused_diversity_selection(
    fused_candidates: List[int],
    corpus: List[Chunk],
    baseline_scores: List[float],
    k: int,
    config: Dict,
) -> List[int]:
    """
    Apply diversity selection to fused candidate list.

    Args:
        fused_candidates: RRF-fused candidate indices
        corpus: Document corpus
        baseline_scores: Original baseline scores for diversity penalty calculation
        k: Number of results to return
        config: Configuration dictionary

    Returns:
        Selected indices with diversity applied
    """
    if not config.get("diversity", {}).get("enabled", False):
        return fused_candidates[:k]

    selected: List[int] = []
    per_doc: Dict[int, int] = {}
    candidates = fused_candidates[:]  # Copy to avoid modifying original

    diversity_config = config["diversity"]
    per_doc_penalty = diversity_config.get("per_doc_penalty", 0.3)
    max_per_doc = diversity_config.get("max_per_doc", 2)

    def doc_key(i: int) -> int:
        return corpus[i].doc_id

    while candidates and len(selected) < k:
        best_idx = None
        best_val = None

        # Consider top candidates for selection
        search_window = candidates[: max(5 * k, 200)]

        for i in search_window:
            base_score = baseline_scores[i]
            count = per_doc.get(doc_key(i), 0)

            # Apply diversity penalty
            adjusted_score = base_score - per_doc_penalty * max(0, count)

            if best_val is None or adjusted_score > best_val:
                best_val = adjusted_score
                best_idx = i

        if best_idx is None:
            break

        selected.append(best_idx)
        dk = doc_key(best_idx)
        per_doc[dk] = per_doc.get(dk, 0) + 1

        # Remove candidates if document limit reached
        if per_doc[dk] >= max_per_doc:
            candidates = [i for i in candidates if doc_key(i) != dk]
        else:
            candidates = [i for i in candidates if i != best_idx]

    return selected
