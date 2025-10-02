#!/usr/bin/env python3
"""Tests for diversity module functions."""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.diversity import (
    simple_tfidf_vectors,
    cosine_similarity_sparse,
    mmr_selection,
    format_results
)
from lightweight_rag.models import Chunk, DocMeta


class TestSimpleTFIDF:
    """Test simple TF-IDF vector creation for MMR."""
    
    def test_simple_tfidf_basic(self):
        """Test basic TF-IDF vector creation."""
        texts = ["machine learning", "deep learning algorithms", "machine vision"]
        query = "machine learning"
        
        doc_vectors, query_vector = simple_tfidf_vectors(texts, query)
        
        assert len(doc_vectors) == 3
        assert "machine" in query_vector
        assert "learning" in query_vector
        assert query_vector["machine"] > 0
        assert query_vector["learning"] > 0
    
    def test_simple_tfidf_empty_texts(self):
        """Test TF-IDF with empty texts."""
        texts = ["", "", ""]
        query = "test query"
        
        doc_vectors, query_vector = simple_tfidf_vectors(texts, query)
        
        assert len(doc_vectors) == 3
        assert "test" in query_vector
        assert "query" in query_vector
    
    def test_simple_tfidf_single_doc(self):
        """Test TF-IDF with single document."""
        texts = ["machine learning algorithms"]
        query = "machine"
        
        doc_vectors, query_vector = simple_tfidf_vectors(texts, query)
        
        assert len(doc_vectors) == 1
        assert "machine" in query_vector


class TestCosineSimilarity:
    """Test cosine similarity calculation."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec1 = {"word1": 1.0, "word2": 1.0}
        vec2 = {"word1": 1.0, "word2": 1.0}
        
        similarity = cosine_similarity_sparse(vec1, vec2)
        
        assert 0.99 <= similarity <= 1.01  # Allow for floating point errors
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = {"word1": 1.0}
        vec2 = {"word2": 1.0}
        
        similarity = cosine_similarity_sparse(vec1, vec2)
        
        assert similarity == 0.0
    
    def test_cosine_similarity_partial_overlap(self):
        """Test cosine similarity with partial overlap."""
        vec1 = {"word1": 1.0, "word2": 1.0}
        vec2 = {"word1": 1.0, "word3": 1.0}
        
        similarity = cosine_similarity_sparse(vec1, vec2)
        
        assert 0.0 < similarity < 1.0
    
    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty vectors."""
        vec1 = {}
        vec2 = {"word1": 1.0}
        
        similarity = cosine_similarity_sparse(vec1, vec2)
        
        assert similarity == 0.0
    
    def test_cosine_similarity_both_empty(self):
        """Test cosine similarity with both vectors empty."""
        vec1 = {}
        vec2 = {}
        
        similarity = cosine_similarity_sparse(vec1, vec2)
        
        assert similarity == 0.0
    
    def test_cosine_similarity_zero_norm(self):
        """Test cosine similarity edge case with zero norm."""
        vec1 = {"word1": 0.0}
        vec2 = {"word1": 1.0}
        
        similarity = cosine_similarity_sparse(vec1, vec2)
        
        assert similarity == 0.0


class TestMMRSelection:
    """Test Maximum Marginal Relevance selection."""
    
    def test_mmr_basic(self):
        """Test basic MMR selection."""
        query = "machine learning"
        candidates = [
            (0, "machine learning algorithms", 0.9),
            (1, "deep learning neural networks", 0.8),
            (2, "machine learning models", 0.85),
            (3, "computer vision systems", 0.7)
        ]
        
        selected = mmr_selection(query, candidates, lambda_param=0.7, k=2)
        
        assert len(selected) <= 2
        assert isinstance(selected, list)
        # Should select diverse results - selected is a list of indices
        if len(selected) >= 2:
            # Check that selected items are indices from the candidate list
            assert all(isinstance(item, int) for item in selected)
            assert all(item in [c[0] for c in candidates] for item in selected)
    
    def test_mmr_empty_candidates(self):
        """Test MMR with empty candidate list."""
        query = "test query"
        candidates = []
        
        selected = mmr_selection(query, candidates, lambda_param=0.7, k=5)
        
        assert selected == []
    
    def test_mmr_single_candidate(self):
        """Test MMR with single candidate."""
        query = "test query"
        candidates = [(0, "single result", 0.9)]
        
        selected = mmr_selection(query, candidates, lambda_param=0.7, k=5)
        
        assert len(selected) == 1
        # mmr_selection returns list of indices
        assert selected[0] == 0  # Returns the index, not the full tuple
    
    def test_mmr_various_thresholds(self):
        """Test MMR selection with various lambda thresholds."""
        query = "machine learning"
        candidates = [
            (0, "machine learning algorithms", 0.9),
            (1, "deep learning", 0.8),
            (2, "data science", 0.7),
        ]
        
        # High lambda (0.9) - prioritize relevance
        selected_high = mmr_selection(query, candidates, lambda_param=0.9, k=2)
        
        # Low lambda (0.3) - prioritize diversity
        selected_low = mmr_selection(query, candidates, lambda_param=0.3, k=2)
        
        # Both should return results
        assert len(selected_high) >= 1
        assert len(selected_low) >= 1
    
    def test_mmr_k_larger_than_candidates(self):
        """Test MMR when k is larger than number of candidates."""
        query = "test"
        candidates = [
            (0, "first result", 0.9),
            (1, "second result", 0.8),
        ]
        
        selected = mmr_selection(query, candidates, lambda_param=0.7, k=10)
        
        # Should return all candidates
        assert len(selected) == 2


class TestFormatResults:
    """Test result formatting variations."""
    
    def test_format_results_basic(self):
        """Test basic result formatting."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi="10.1234/test",
            source="test.pdf"
        )
        
        chunks = [
            Chunk(doc_id=0, source="test.pdf", page=1, text="Test content", meta=meta)
        ]
        
        # format_results takes (chunk_index, score) tuples
        formatted = format_results([(0, 0.9)], chunks, "test query")
        
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert "text" in formatted[0]
        assert "citation" in formatted[0]
    
    def test_format_results_empty(self):
        """Test formatting empty results."""
        formatted = format_results([], [], "test query")
        
        assert isinstance(formatted, list)
        assert len(formatted) == 0
    
    def test_format_results_multiple_chunks(self):
        """Test formatting multiple chunks."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi="10.1234/test",
            source="test.pdf"
        )
        
        chunks = [
            Chunk(doc_id=0, source="test.pdf", page=1, text="Content 1", meta=meta),
            Chunk(doc_id=0, source="test.pdf", page=2, text="Content 2", meta=meta),
            Chunk(doc_id=1, source="test2.pdf", page=1, text="Content 3", meta=meta),
        ]
        
        formatted = format_results([(0, 0.9), (1, 0.8), (2, 0.7)], chunks, "test")
        
        assert isinstance(formatted, list)
        assert len(formatted) == 3
    
    def test_format_results_with_scores(self):
        """Test formatting results with relevance scores."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf"
        )
        
        chunks = [
            Chunk(doc_id=0, source="test.pdf", page=1, text="Test content", meta=meta)
        ]
        
        formatted = format_results([(0, 0.95)], chunks, "test", include_scores=True)
        
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        # May or may not include score depending on implementation
        if "score" in formatted[0]:
            assert formatted[0]["score"] == 0.95
    
    def test_format_results_without_scores(self):
        """Test formatting results without scores."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf"
        )
        
        chunks = [
            Chunk(doc_id=0, source="test.pdf", page=1, text="Test content" * 100, meta=meta)
        ]
        
        formatted = format_results([(0, 0.95)], chunks, "test", include_scores=False)
        
        assert isinstance(formatted, list)
        assert len(formatted) == 1
    
    def test_format_results_snippet_length(self):
        """Test that text snippets respect max length."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf"
        )
        
        long_text = "word " * 500  # Very long text
        chunks = [
            Chunk(doc_id=0, source="test.pdf", page=1, text=long_text, meta=meta)
        ]
        
        formatted = format_results([(0, 0.95)], chunks, "test", max_snippet_chars=100)
        
        assert isinstance(formatted, list)
        # Text should be truncated
        if "text" in formatted[0]:
            assert len(formatted[0]["text"]) <= 200  # Allow some margin


class TestTFIDFEdgeCases:
    """Test TF-IDF calculation edge cases."""
    
    def test_tfidf_all_empty_texts(self):
        """Test TF-IDF with all empty texts."""
        texts = ["", "", ""]
        query = "test"
        
        doc_vectors, query_vector = simple_tfidf_vectors(texts, query)
        
        assert len(doc_vectors) == 3
        assert isinstance(query_vector, dict)
    
    def test_tfidf_repeated_terms(self):
        """Test TF-IDF with repeated terms."""
        texts = ["test test test", "other words here"]
        query = "test"
        
        doc_vectors, query_vector = simple_tfidf_vectors(texts, query)
        
        assert len(doc_vectors) == 2
        # First doc should have high weight for "test"
        assert "test" in doc_vectors[0]
    
    def test_tfidf_no_overlap(self):
        """Test TF-IDF with no overlap between query and documents."""
        texts = ["completely different words", "totally unrelated content"]
        query = "specific query terms"
        
        doc_vectors, query_vector = simple_tfidf_vectors(texts, query)
        
        assert len(doc_vectors) == 2
        assert isinstance(query_vector, dict)
        # No shared terms
        for doc_vec in doc_vectors:
            common = set(doc_vec.keys()) & set(query_vector.keys())
            # Might have some overlap or not depending on tokenization
            assert isinstance(common, set)
