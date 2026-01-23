"""Main RAG pipeline orchestration."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from .diversity import format_results
from .index import build_bm25, tokenize, update_cache_paths
from .io_pdf import build_corpus
from .models import Chunk
from .prf import rm3_expand_query
from .scoring import (
    fuzzy_match_bonus,
    gibberish_penalty,
    metadata_bonus,
    ngram_bonus,
    pattern_bonus,
    proximity_bonus,
)


def calibrate_confidence(
    scores: Any,
    runs: List[List[int]],
    pool: List[int],
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Compute a confidence label (high/medium/low) based on score distributions
    and stability across multiple ranking runs.
    """
    if not scores or not pool:
        return {
            "level": "low",
            "score": 0.0,
            "spread": 0.0,
            "stability": 0.0,
            "reason": "No results",
        }

    # Handle both list of scores and dictionary of scores
    if isinstance(scores, dict):
        pool_scores = [scores.get(i, 0.0) for i in pool]
    else:
        pool_scores = [scores[i] for i in pool]
        
    if not pool_scores:
        return {
            "level": "low",
            "score": 0.0,
            "spread": 0.0,
            "stability": 0.0,
            "reason": "No results",
        }

    top_score = max(pool_scores)

    # Sort pool scores to find median
    sorted_pool_scores = sorted(pool_scores, reverse=True)
    median_score = sorted_pool_scores[len(sorted_pool_scores) // 2]

    # Spread metric: how much does the top result stand out from the pack?
    # We use a simple ratio-based stand-out factor
    spread = (top_score - median_score) / (top_score + 1e-6) if top_score > 0 else 0

    # 2. Stability across runs: overlap in top results across all runs
    if not runs or len(runs) < 2:
        stability = 0.5  # Neutral if only one run
    else:
        # Check agreement on the top_k results
        top_sets = [set(run[:top_k]) for run in runs]

        # Agreement: intersection / union of all top sets
        intersection = set.intersection(*top_sets)
        union = set.union(*top_sets)
        stability = len(intersection) / len(union) if union else 0.0

    # Combine metrics into a confidence score (0 to 1)
    # Weights: spread is important for local distinction, stability for method agreement
    # We cap spread at 0.5 (representing 50% gap from median) for the score contribution
    confidence_score = (0.5 * min(1.0, spread * 2.0)) + (0.5 * stability)

    if confidence_score > 0.75:
        level = "high"
    elif confidence_score > 0.4:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": round(float(confidence_score), 3),
        "spread": round(float(spread), 3),
        "stability": round(float(stability), 3),
    }


def search_topk(
    corpus: List[Chunk],
    bm25: BM25Okapi,
    tokenized: List[List[str]],
    query: str,
    k: int = 8,
    prox_window: int = 30,
    prox_lambda: float = 0.2,
    ngram_lambda: float = 0.1,
    diversity: bool = True,
    div_lambda: float = 0.3,
    max_per_doc: int = 2,
    semantic: bool = False,
    semantic_topn: int = 80,
    max_snippet_chars: int = 900,
    include_scores: bool = True,
    include_pandoc_cite: bool = False,
    use_pandoc_as_primary: bool = False,
    fusion_config: Optional[Dict] = None,
    bm25_query: Optional[str] = None,
    semantic_query: Optional[str] = None,
    **kwargs,  # Accept additional config parameters
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main search function with RRF fusion of multiple ranking strategies.

    Returns (formatted results, confidence dictionary).
    """
    from .fusion import build_ranking_runs, fused_diversity_selection, rrf_fuse

    if not corpus:
        return [], {}

    # Build baseline BM25 + bonuses scores
    q_tokens = tokenize(query)
    bm25_tokens = tokenize(bm25_query) if bm25_query else q_tokens
    base_scores = bm25.get_scores(bm25_tokens)

    # Apply bonuses
    # Ensure scores are native Python floats to satisfy type checkers and downstream functions
    scores = [float(s) for s in base_scores]
    for i, chunk in enumerate(corpus):
        # Proximity bonus
        if prox_lambda > 0 and prox_window > 0:
            pb = proximity_bonus(chunk.text, q_tokens, window_size=prox_window)
            if pb:
                scores[i] += prox_lambda * pb

        # N-gram bonus
        if ngram_lambda > 0:
            nb = ngram_bonus(chunk.text, query)
            if nb:
                scores[i] += ngram_lambda * nb

        # Pattern bonus
        scores[i] += pattern_bonus(chunk.text)
        
        # Metadata bonus (title, abstract, results, conclusion)
        scores[i] += metadata_bonus(chunk.text, doc_title=chunk.meta.title)
        
        # Gibberish penalty (multiplicative)
        gib_penalty = gibberish_penalty(chunk.text, threshold=0.20)
        if gib_penalty < 1.0:
            scores[i] *= gib_penalty
        
        # Fuzzy match bonus (for exact quote searches)
        fuzzy_bonus = fuzzy_match_bonus(chunk.text, query, min_length=20)
        if fuzzy_bonus > 0:
            scores[i] += 2.0 * fuzzy_bonus  # Weight of 2.0 for strong matches

    # Build candidate pool (larger for fusion)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    pool_size = (
        fusion_config.get("bm25", {}).get("pool_size", 200) if fusion_config else max(3 * k, 200)
    )
    pool = order[:pool_size]

    # Build configuration for ranking runs
    run_config = {
        "prf": kwargs.get("prf_config", {}),
        "rerank": {"semantic": {"enabled": semantic, "topn": semantic_topn}},
        "fusion": fusion_config.get("fusion", {}) if fusion_config else {},
        "diversity": {
            "enabled": diversity,
            "per_doc_penalty": div_lambda,
            "max_per_doc": max_per_doc,
        },
    }

    # Build multiple ranking runs
    runs = build_ranking_runs(
        query, corpus, bm25, tokenized, pool, scores, run_config, semantic_query=semantic_query
    )

    # Calibrate confidence based on spread and run stability
    confidence = calibrate_confidence(scores, runs, pool, top_k=k)

    # Apply RRF fusion if enabled and we have multiple runs
    if (
        len(runs) >= 2
        and fusion_config
        and fusion_config.get("fusion", {}).get("rrf", {}).get("enabled", False)
    ):

        rrf_config = fusion_config["fusion"]["rrf"]
        fused_candidates = rrf_fuse(runs, C=rrf_config.get("C", 60), cap=rrf_config.get("cap", 200))
    else:
        # Use baseline run if fusion disabled or only one run
        fused_candidates = runs[0] if runs else pool

    # Apply diversity selection on fused results
    if diversity:
        selected_indices = fused_diversity_selection(
            fused_candidates, corpus, scores, k, run_config
        )
    else:
        selected_indices = fused_candidates[:k]

    # Apply MMR for final diversification if enabled
    if (
        run_config.get("diversity", {}).get("mmr", {}).get("enabled", False)
        and len(selected_indices) > 1
    ):
        from .diversity import mmr_selection

        mmr_lambda = run_config["diversity"]["mmr"].get("lambda", 0.7)
        mmr_candidates = [
            (idx, corpus[idx].text, scores[idx])
            for idx in selected_indices[: min(20, len(selected_indices))]
        ]
        selected_indices = mmr_selection(query, mmr_candidates, mmr_lambda, k)

    # Format results
    selected_results = [(idx, scores[idx]) for idx in selected_indices]
    final_results = format_results(
        selected_results,
        corpus,
        query,
        max_snippet_chars,
        include_scores,
        include_pandoc_cite,
        use_pandoc_as_primary,
    )

    return final_results, confidence


async def _run_rag_pipeline_internal(
    cfg: Dict[str, Any], query: str
) -> tuple[
    List[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]
]:
    """
    Run the complete RAG pipeline with the given configuration and query.
    Returns (results, summary, summary_debug, confidence).
    """
    from .performance import seed_numpy, sort_results_deterministically

    if cfg["performance"]["deterministic"] and cfg["performance"]["numpy_seed"]:
        seed_numpy(cfg["performance"]["numpy_seed"])

    cache_dir = Path(cfg["paths"]["cache_dir"])
    update_cache_paths(cache_dir)

    pdf_dir = Path(cfg["paths"]["pdf_dir"])
    corpus = await build_corpus(
        pdf_dir,
        max_workers=cfg["performance"]["pdf_thread_workers"],
        cache_seconds=cfg["citations"]["cache_seconds"],
        max_concurrent_api=cfg["performance"]["api_semaphore_size"],
        citation_config=cfg["citations"],
        chunking_config=cfg["indexing"],
    )

    if not corpus:
        print("No documents found or processed.")
        return [], None, None, None

    token_pattern = cfg["bm25"]["token_pattern"]
    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]
    bm25, tokenized = build_bm25(corpus, token_pattern, k1, b)
    print(f"Indexed {len(corpus)} chunks")

    bm25_query = query
    semantic_query = None
    llm_client = None
    hyde_passages = []

    if cfg.get("llm", {}).get("enabled", False):
        from .llm import LLMClient

        llm_client = LLMClient(cfg["llm"])
        print("Generating hypothetical answers for query expansion...")
        hyde_passages = llm_client.generate_hypothetical_answers(query)
        
        if hyde_passages:
            print(f"[HyDE] Generated {len(hyde_passages)} hypothetical passages.")
            
            # Use joined passages for primary expansion if enabled
            hypothetical_joined = " ".join(hyde_passages)
            if cfg["llm"].get("use_for_bm25", True):
                bm25_query = f"{query} {hypothetical_joined}"
            
            if cfg["llm"].get("use_for_semantic", False):
                semantic_query = f"{query} {hypothetical_joined}"
            
            # If multiple passages generated, we can pass them as a list for diversity runs
            if len(hyde_passages) > 1:
                # We'll pass the list via hyde_queries in config or as a special param
                cfg["llm"]["hyde_queries"] = hyde_passages

    if cfg["prf"]["enabled"]:
        bm25_query = rm3_expand_query(
            bm25_query,
            bm25,
            tokenized,
            corpus,
            fb_docs=cfg["prf"]["fb_docs"],
            fb_terms=cfg["prf"]["fb_terms"],
            alpha=cfg["prf"]["alpha"],
        )
        if bm25_query != query:
            print(f"\n[RM3] Expanded query terms added.")

    summary_cfg = cfg.get("llm", {}).get("summary", {})
    summary_enabled = bool(cfg.get("llm", {}).get("enabled", False)) and bool(
        summary_cfg.get("enabled", False)
    )
    summary_top_k = int(summary_cfg.get("top_k", 25)) if summary_enabled else 0
    result_top_k = int(cfg["rerank"]["final_top_k"])
    search_k = max(result_top_k, summary_top_k) if summary_enabled else result_top_k

    results, confidence = search_topk(
        corpus=corpus,
        bm25=bm25,
        tokenized=tokenized,
        query=query,
        bm25_query=bm25_query,
        semantic_query=semantic_query,
        k=search_k,
        prox_window=(
            0
            if not cfg["bonuses"]["proximity"]["enabled"]
            else cfg["bonuses"]["proximity"]["window"]
        ),
        prox_lambda=(
            0.0
            if not cfg["bonuses"]["proximity"]["enabled"]
            else cfg["bonuses"]["proximity"]["weight"]
        ),
        ngram_lambda=(
            cfg["bonuses"]["ngram"]["weight"] if cfg["bonuses"]["ngram"]["enabled"] else 0.0
        ),
        diversity=cfg["diversity"]["enabled"],
        div_lambda=cfg["diversity"]["per_doc_penalty"],
        max_per_doc=cfg["diversity"]["max_per_doc"],
        semantic=cfg["rerank"]["semantic"]["enabled"],
        semantic_topn=cfg["rerank"]["semantic"]["topn"],
        max_snippet_chars=cfg["output"]["max_snippet_chars"],
        include_scores=cfg["output"]["include_scores"],
        include_pandoc_cite=cfg["citations"].get("include_pandoc_cite", False),
        use_pandoc_as_primary=cfg["citations"].get("pandoc_as_primary", False),
        fusion_config=cfg,
        prf_config=cfg["prf"],
    )

    if cfg["performance"]["deterministic"]:
        results = sort_results_deterministically(results)

    summary = None
    summary_debug = None
    
    if llm_client is not None and summary_cfg.get("debug", False):
        summary_debug = {}
    
    if summary_enabled and results and llm_client is not None:
        chunk_texts = [r["text"] for r in results[: min(summary_top_k, len(results))]]
        summary = llm_client.generate_summary(
            query, chunk_texts, max_tokens=summary_cfg.get("max_tokens")
        )
        if summary_debug is not None and llm_client.last_summary_debug:
            prompt = llm_client.last_summary_debug.get("prompt", "")
            summary_debug["prompt_excerpt"] = prompt[:100]
            summary_debug["full_prompt"] = prompt
            summary_debug["summary"] = llm_client.last_summary_debug.get("summary")
            summary_debug["error"] = llm_client.last_summary_debug.get("error")
            summary_debug["raw_response"] = llm_client.last_summary_debug.get("raw_response")

    # Always include HyDE debug info if LLM and debug are enabled, even if summary is off
    if summary_debug is not None and llm_client is not None and llm_client.last_hyde_debug:
        hyde_prompt = llm_client.last_hyde_debug.get("prompt", "")
        summary_debug["hyde_prompt_excerpt"] = hyde_prompt[:100]
        summary_debug["hyde_result"] = llm_client.last_hyde_debug.get("summary")
        summary_debug["hyde_error"] = llm_client.last_hyde_debug.get("error")
        summary_debug["hyde_raw_response"] = llm_client.last_hyde_debug.get("raw_response")

    return results[:result_top_k], summary, summary_debug, confidence


async def run_rag_pipeline(cfg: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Run the complete RAG pipeline with the given configuration and query.
    """
    results, _summary, _summary_debug, _confidence = await _run_rag_pipeline_internal(cfg, query)
    return results


async def run_rag_pipeline_with_summary(cfg: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Run the pipeline and return results plus summary."""
    results, summary, summary_debug, confidence = await _run_rag_pipeline_internal(cfg, query)
    return {
        "results": results,
        "summary": summary,
        "summary_debug": summary_debug,
        "confidence": confidence,
    }


def query_pdfs(query: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Simple API function for querying PDFs.

    This is the main entry point for programmatic use.
    """
    import asyncio

    return asyncio.run(run_rag_pipeline(cfg, query))
