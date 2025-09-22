"""Semantic reranking with sentence transformers."""

from typing import List, Optional

# Optional imports for semantic reranking
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    np = None
    SentenceTransformer = None


_model = None


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional["np.ndarray"]:
    """
    Return normalized embeddings or None if sentence-transformers not installed.
    
    Lazy loads the model to keep cold-start times low.
    """
    global _model
    
    if SentenceTransformer is None or np is None:
        return None
    
    if _model is None:
        try:
            _model = SentenceTransformer(model_name, device='cpu')
        except Exception as e:
            print(f"Failed to load sentence transformer model: {e}")
            return None
    
    try:
        embeddings = _model.encode(texts, convert_to_numpy=True)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    except Exception as e:
        print(f"Failed to encode texts: {e}")
        return None


def semantic_rerank(
    query: str, 
    texts: List[str], 
    scores: List[float], 
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[float]:
    """
    Rerank using semantic similarity.
    
    Returns updated scores that combine BM25 + semantic similarity.
    If semantic reranking fails, returns original scores.
    """
    if not texts or SentenceTransformer is None:
        return scores
    
    # Encode query and texts
    all_texts = [query] + texts
    embeddings = embed_texts(all_texts, model_name)
    
    if embeddings is None:
        return scores
    
    # Calculate cosine similarities (already normalized)
    query_emb = embeddings[0:1]  # Shape (1, dim)
    text_embs = embeddings[1:]   # Shape (n, dim)
    
    similarities = np.dot(text_embs, query_emb.T).flatten()  # Shape (n,)
    
    # Combine BM25 and semantic scores
    # Simple linear combination - can be tuned
    alpha = 0.7  # Weight for BM25 score
    beta = 0.3   # Weight for semantic similarity
    
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
    combined_scores = [
        alpha * norm_scores[i] + beta * similarities[i]
        for i in range(len(scores))
    ]
    
    return combined_scores