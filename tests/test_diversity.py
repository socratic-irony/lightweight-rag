#!/usr/bin/env python3
"""Tests for diversity module functions."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.diversity import (
    simple_tfidf_vectors,
    cosine_similarity_sparse
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
