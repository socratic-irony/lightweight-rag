"""Diversity selection to avoid too many results from the same document."""

from typing import List, Tuple, Dict, Any

from .models import Chunk


def apply_diversity_selection(
    results: List[Tuple[int, float]], 
    corpus: List[Chunk], 
    div_lambda: float = 0.3, 
    max_per_doc: int = 2
) -> List[Tuple[int, float]]:
    """
    Greedy diversity selection that penalizes additional results from same document.
    
    Args:
        results: List of (chunk_index, score) tuples sorted by score
        corpus: The full corpus of chunks
        div_lambda: Penalty factor for additional hits from same doc
        max_per_doc: Maximum results allowed from the same document
    
    Returns:
        Filtered and re-scored results with diversity applied
    """
    if not results:
        return results
    
    doc_counts = {}  # doc_id -> count
    diverse_results = []
    
    for chunk_idx, score in results:
        chunk = corpus[chunk_idx]
        doc_id = chunk.doc_id
        
        current_count = doc_counts.get(doc_id, 0)
        
        # Skip if we've hit the per-document limit
        if current_count >= max_per_doc:
            continue
        
        # Apply diversity penalty
        penalty = current_count * div_lambda
        adjusted_score = score - penalty
        
        diverse_results.append((chunk_idx, adjusted_score))
        doc_counts[doc_id] = current_count + 1
    
    # Re-sort by adjusted scores
    diverse_results.sort(key=lambda x: x[1], reverse=True)
    return diverse_results


def format_results(
    selected_results: List[Tuple[int, float]], 
    corpus: List[Chunk],
    query: str,
    max_snippet_chars: int = 900,
    include_scores: bool = True
) -> List[Dict[str, Any]]:
    """
    Format final results for output.
    
    Args:
        selected_results: List of (chunk_index, score) tuples
        corpus: The full corpus of chunks
        query: Original query string
        max_snippet_chars: Maximum characters in text snippet
        include_scores: Whether to include relevance scores
    
    Returns:
        List of formatted result dictionaries
    """
    from .cite import author_date_citation
    from .models import window
    
    formatted_results = []
    
    for chunk_idx, score in selected_results:
        chunk = corpus[chunk_idx]
        
        # Create citation
        citation = author_date_citation(chunk.meta, chunk.page)
        
        # Prepare result dictionary
        result = {
            "text": window(chunk.text, max_snippet_chars),
            "citation": citation,
            "source": {
                "file": chunk.source,
                "page": chunk.page,
                "doi": chunk.meta.doi,
                "title": chunk.meta.title
            }
        }
        
        if include_scores:
            result["score"] = round(score, 3)
        
        formatted_results.append(result)
    
    return formatted_results