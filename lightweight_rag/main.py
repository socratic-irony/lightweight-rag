"""Main RAG pipeline orchestration."""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from .models import Chunk
from .index import tokenize
from .io_pdf import build_corpus
from .index import build_bm25, update_cache_paths
from .scoring import proximity_bonus, ngram_bonus, pattern_bonus
from .rerank import semantic_rerank  
from .prf import rm3_expand_query
from .diversity import apply_diversity_selection, format_results


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
    fusion_config: Optional[Dict] = None,
    **kwargs  # Accept additional config parameters
) -> List[Dict[str, Any]]:
    """
    Main search function with RRF fusion of multiple ranking strategies.
    
    Returns formatted results ready for display.
    """
    from .fusion import build_ranking_runs, rrf_fuse, fused_diversity_selection
    
    if not corpus:
        return []
    
    # Build baseline BM25 + bonuses scores
    q_tokens = tokenize(query)
    base_scores = bm25.get_scores(q_tokens)

    # Apply bonuses
    scores = list(base_scores)
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

    # Build candidate pool (larger for fusion)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    pool_size = fusion_config.get("bm25", {}).get("pool_size", 200) if fusion_config else max(3 * k, 200)
    pool = order[:pool_size]

    # Build configuration for ranking runs
    run_config = {
        "prf": kwargs.get("prf_config", {}),
        "rerank": {"semantic": {"enabled": semantic, "topn": semantic_topn}},
        "fusion": fusion_config.get("fusion", {}) if fusion_config else {},
        "diversity": {"enabled": diversity, "per_doc_penalty": div_lambda, "max_per_doc": max_per_doc}
    }

    # Build multiple ranking runs
    runs = build_ranking_runs(query, corpus, bm25, tokenized, pool, scores, run_config)
    
    # Apply RRF fusion if enabled and we have multiple runs
    if (len(runs) >= 2 and 
        fusion_config and 
        fusion_config.get("fusion", {}).get("rrf", {}).get("enabled", False)):
        
        rrf_config = fusion_config["fusion"]["rrf"]
        fused_candidates = rrf_fuse(runs, C=rrf_config.get("C", 60), cap=rrf_config.get("cap", 200))
    else:
        # Use baseline run if fusion disabled or only one run
        fused_candidates = runs[0] if runs else pool

    # Apply diversity selection on fused results
    if diversity:
        selected_indices = fused_diversity_selection(fused_candidates, corpus, scores, k, run_config)
    else:
        selected_indices = fused_candidates[:k]

    # Apply MMR for final diversification if enabled
    if (run_config.get("diversity", {}).get("mmr", {}).get("enabled", False) and 
        len(selected_indices) > 1):
        from .diversity import mmr_selection
        
        mmr_lambda = run_config["diversity"]["mmr"].get("lambda", 0.7)
        mmr_candidates = [(idx, corpus[idx].text, scores[idx]) for idx in selected_indices[:min(20, len(selected_indices))]]
        selected_indices = mmr_selection(query, mmr_candidates, mmr_lambda, k)

    # Format results
    selected_results = [(idx, scores[idx]) for idx in selected_indices]
    final_results = format_results(
        selected_results,
        corpus,
        query,
        max_snippet_chars,
        include_scores,
        include_pandoc_cite
    )
    
    return final_results


async def run_rag_pipeline(cfg: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Run the complete RAG pipeline with the given configuration and query.
    """
    # Initialize performance settings
    from .performance import seed_numpy, sort_results_deterministically
    
    # Seed numpy for deterministic behavior
    if cfg["performance"]["deterministic"] and cfg["performance"]["numpy_seed"]:
        seed_numpy(cfg["performance"]["numpy_seed"])
    
    # Update cache paths from config
    cache_dir = Path(cfg["paths"]["cache_dir"])
    update_cache_paths(cache_dir)
    
    # Build or load corpus
    pdf_dir = Path(cfg["paths"]["pdf_dir"])
    corpus = await build_corpus(
        pdf_dir,
        max_workers=cfg["performance"]["pdf_thread_workers"],
        cache_seconds=cfg["citations"]["cache_seconds"],
        max_concurrent_api=cfg["performance"]["api_semaphore_size"],
        citation_config=cfg["citations"],
        chunking_config=cfg["indexing"]
    )
    
    if not corpus:
        print("No documents found or processed.")
        return []
    
    # Build BM25 index with configurable parameters
    token_pattern = cfg["bm25"]["token_pattern"]
    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]
    bm25, tokenized = build_bm25(corpus, token_pattern, k1, b)
    print(f"Indexed {len(corpus)} chunks")

    # Query expansion with RM3 if enabled
    expanded_query = query
    if cfg["prf"]["enabled"]:
        expanded_query = rm3_expand_query(
            query, bm25, tokenized, corpus,
            fb_docs=cfg["prf"]["fb_docs"],
            fb_terms=cfg["prf"]["fb_terms"],
            alpha=cfg["prf"]["alpha"]
        )
        if expanded_query != query:
            print(f"\n[RM3] Expanded query: {expanded_query}")

    # Search with all configured options
    results = search_topk(
        corpus=corpus,
        bm25=bm25,
        tokenized=tokenized,
        query=expanded_query,
        k=cfg["rerank"]["final_top_k"],
        prox_window=(0 if not cfg["bonuses"]["proximity"]["enabled"] 
                    else cfg["bonuses"]["proximity"]["window"]),
        prox_lambda=(0.0 if not cfg["bonuses"]["proximity"]["enabled"] 
                    else cfg["bonuses"]["proximity"]["weight"]),
        ngram_lambda=cfg["bonuses"]["ngram"]["weight"] if cfg["bonuses"]["ngram"]["enabled"] else 0.0,
        diversity=cfg["diversity"]["enabled"],
        div_lambda=cfg["diversity"]["per_doc_penalty"],
        max_per_doc=cfg["diversity"]["max_per_doc"],
        semantic=cfg["rerank"]["semantic"]["enabled"],
        semantic_topn=cfg["rerank"]["semantic"]["topn"],
        max_snippet_chars=cfg["output"]["max_snippet_chars"],
        include_scores=cfg["output"]["include_scores"],
        include_pandoc_cite=cfg["citations"].get("include_pandoc_cite", False),
        fusion_config=cfg,  # Pass full config for fusion
        prf_config=cfg["prf"]  # Pass PRF config separately
    )
    
    # Apply deterministic sorting if enabled
    if cfg["performance"]["deterministic"]:
        results = sort_results_deterministically(results)

    return results


def query_pdfs(query: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Simple API function for querying PDFs.
    
    This is the main entry point for programmatic use.
    """
    import asyncio
    return asyncio.run(run_rag_pipeline(cfg, query))
