"""Main pipeline orchestration for the lightweight RAG system."""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi

from .models import Chunk
from .io_pdf import build_corpus
from .index import build_bm25, tokenize, update_cache_paths
from .scoring import proximity_bonus, ngram_bonus, pattern_bonus
from .prf import rm3_expand_query
from .rerank import semantic_rerank
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
    **kwargs  # Accept additional config parameters
) -> List[Dict[str, Any]]:
    """
    Main search function that combines BM25 with bonuses, optional reranking, and diversity.
    
    Returns formatted results ready for display.
    """
    if not corpus:
        return []
    
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

    # Get candidate pool
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidate_pool_size = max(semantic_topn, k) if semantic else max(3 * k, 100)
    candidates = order[:candidate_pool_size]

    # Optional semantic reranking
    if semantic and len(candidates) > 1:
        candidate_texts = [corpus[i].text for i in candidates]
        candidate_scores = [scores[i] for i in candidates]
        
        # Rerank candidates
        reranked_scores = semantic_rerank(query, candidate_texts, candidate_scores)
        
        # Update scores
        for i, new_score in enumerate(reranked_scores):
            scores[candidates[i]] = new_score
        
        # Re-sort candidates by new scores
        candidates = sorted(candidates, key=lambda i: scores[i], reverse=True)

    # Prepare results for diversity selection
    candidate_results = [(idx, scores[idx]) for idx in candidates]

    # Apply diversity selection
    if diversity:
        diverse_results = apply_diversity_selection(
            candidate_results, corpus, div_lambda, max_per_doc
        )
    else:
        diverse_results = candidate_results

    # Take top k results
    final_results = diverse_results[:k]

    # Format results
    formatted_results = format_results(
        final_results, corpus, query, max_snippet_chars, include_scores
    )

    return formatted_results


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
        max_concurrent_api=cfg["performance"]["api_semaphore_size"]
    )
    
    if not corpus:
        print("No documents found or processed.")
        return []
    
    # Build BM25 index with configurable token pattern
    token_pattern = cfg["bm25"]["token_pattern"]
    bm25, tokenized = build_bm25(corpus, token_pattern)
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
        include_scores=cfg["output"]["include_scores"]
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