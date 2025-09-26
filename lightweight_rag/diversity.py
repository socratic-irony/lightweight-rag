"""Diversity selection to avoid too many results from the same document."""

from typing import List, Tuple, Dict, Any, Optional
import re
from collections import Counter

from .models import Chunk

# Try to import numpy for MMR, fallback to TF-IDF if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def simple_tfidf_vectors(texts: List[str], query: str) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Create simple TF-IDF vectors for MMR when numpy is not available.
    
    Returns:
        Tuple of (doc_vectors, query_vector) as sparse dicts
    """
    # Simple tokenization
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    # Build vocabulary and document frequencies
    all_texts = texts + [query]
    vocabulary = set()
    doc_tokens = []
    
    for text in all_texts:
        tokens = tokenize(text)
        doc_tokens.append(tokens)
        vocabulary.update(tokens)
    
    vocabulary = list(vocabulary)
    doc_count = len(texts)
    
    # Calculate document frequencies
    df = {}
    for term in vocabulary:
        df[term] = sum(1 for tokens in doc_tokens[:-1] if term in tokens)
    
    # Calculate TF-IDF vectors
    vectors = []
    for tokens in doc_tokens[:-1]:  # Exclude query
        tf = Counter(tokens)
        vector = {}
        for term in vocabulary:
            if term in tf:
                tf_score = tf[term] / len(tokens)
                idf_score = np.log(doc_count / max(1, df[term])) if HAS_NUMPY else 1.0
                vector[term] = tf_score * idf_score
        vectors.append(vector)
    
    # Query vector
    query_tokens = doc_tokens[-1]
    query_tf = Counter(query_tokens)
    query_vector = {}
    for term in vocabulary:
        if term in query_tf:
            tf_score = query_tf[term] / len(query_tokens)
            idf_score = np.log(doc_count / max(1, df[term])) if HAS_NUMPY else 1.0
            query_vector[term] = tf_score * idf_score
    
    return vectors, query_vector


def cosine_similarity_sparse(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Calculate cosine similarity between sparse vectors."""
    # Get common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())
    if not common_terms:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    
    # Calculate norms
    norm1 = sum(val ** 2 for val in vec1.values()) ** 0.5
    norm2 = sum(val ** 2 for val in vec2.values()) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def mmr_selection(
    query: str,
    candidates: List[Tuple[int, str, float]],  # (index, text, score)
    lambda_param: float = 0.7,
    k: int = 10
) -> List[int]:
    """
    Max Marginal Relevance selection for diversity.
    
    Args:
        query: Query string
        candidates: List of (index, text, relevance_score) tuples
        lambda_param: Balance between relevance and diversity (0.0-1.0)
        k: Number of results to select
        
    Returns:
        List of selected indices in MMR order
    """
    if not candidates or k <= 0:
        return []
    
    if len(candidates) <= k:
        return [idx for idx, _, _ in candidates]
    
    # Extract texts and scores
    texts = [text for _, text, _ in candidates]
    relevance_scores = [score for _, _, score in candidates]
    indices = [idx for idx, _, _ in candidates]
    
    # Create TF-IDF vectors (fallback if numpy not available)
    if HAS_NUMPY:
        try:
            # Try to use simple embeddings if available
            from .rerank import tokenize_for_rerank
            
            # Simple word overlap similarity
            query_terms = set(tokenize_for_rerank(query))
            doc_terms = [set(tokenize_for_rerank(text)) for text in texts]
            
            # Calculate query-document similarities
            query_sims = []
            for terms in doc_terms:
                if not terms:
                    query_sims.append(0.0)
                else:
                    overlap = len(query_terms & terms)
                    query_sims.append(overlap / (len(query_terms) + len(terms) - overlap + 1e-9))
            
            # MMR selection
            selected = []
            remaining = list(range(len(candidates)))
            
            # Select first item with highest relevance score
            best_idx = max(remaining, key=lambda i: relevance_scores[i])
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            while remaining and len(selected) < k:
                mmr_scores = []
                
                for i in remaining:
                    # Relevance component
                    relevance = lambda_param * query_sims[i]
                    
                    # Diversity component (max similarity to already selected)
                    max_sim = 0.0
                    for j in selected:
                        # Simple Jaccard similarity between documents
                        sim = len(doc_terms[i] & doc_terms[j]) / (len(doc_terms[i] | doc_terms[j]) + 1e-9)
                        max_sim = max(max_sim, sim)
                    
                    diversity = (1 - lambda_param) * max_sim
                    mmr_score = relevance - diversity
                    mmr_scores.append((mmr_score, i))
                
                # Select item with highest MMR score
                _, best_idx = max(mmr_scores, key=lambda x: x[0])
                selected.append(best_idx)
                remaining.remove(best_idx)
            
            return [indices[i] for i in selected]
            
        except Exception:
            # Fallback to simple selection
            pass
    
    # Simple fallback: just return top-k by relevance
    sorted_candidates = sorted(enumerate(candidates), key=lambda x: x[1][2], reverse=True)
    return [candidates[i][0] for i, _ in sorted_candidates[:k]]


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
        # Defensive check for index bounds
        if chunk_idx >= len(corpus):
            print(f"Warning: chunk index {chunk_idx} out of range (corpus size: {len(corpus)}). Skipping.")
            continue
            
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
    include_scores: bool = True,
    include_pandoc_cite: bool = False
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
    from .cite import author_date_citation, pandoc_citation
    from .models import window
    
    formatted_results = []
    
    for chunk_idx, score in selected_results:
        chunk = corpus[chunk_idx]
        
        # Create citation
        citation = author_date_citation(chunk.meta, chunk.page)
        
        # Prepare result dictionary
        result: Dict[str, Any] = {
            "text": window(chunk.text, max_snippet_chars),
            "citation": citation,
            "source": {
                "file": chunk.source,
                "page": chunk.page,
                "doi": chunk.meta.doi,
                "title": chunk.meta.title,
                "citekey": chunk.meta.citekey
            }
        }
        if include_pandoc_cite:
            result["pandoc"] = pandoc_citation(chunk.meta, chunk.page)
        
        if include_scores:
            result["score"] = round(score, 3)
        
        formatted_results.append(result)
    
    return formatted_results
