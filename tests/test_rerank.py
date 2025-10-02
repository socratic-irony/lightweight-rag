#!/usr/bin/env python3
"""Tests for rerank module functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.rerank import (
    tokenize_for_rerank,
    idf_weight,
    coverage_score,
    proximity_score,
    phrase_boost
)


class TestTokenizeForRerank:
    """Test tokenization for reranking."""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Machine Learning Algorithms"
        tokens = tokenize_for_rerank(text)
        assert "machine" in tokens
        assert "learning" in tokens
        assert "algorithms" in tokens
    
    def test_tokenize_with_hyphens(self):
        """Test tokenization with hyphenated words."""
        text = "well-known machine-learning"
        tokens = tokenize_for_rerank(text)
        assert len(tokens) > 0
    
    def test_tokenize_with_underscores(self):
        """Test tokenization with underscores."""
        text = "python_code variable_name"
        tokens = tokenize_for_rerank(text)
        assert len(tokens) > 0
    
    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        tokens = tokenize_for_rerank("")
        assert tokens == []


class TestIDFWeight:
    """Test IDF weight calculation."""
    
    def test_idf_weight_basic(self):
        """Test basic IDF weight calculation."""
        query_terms = ["machine", "learning"]
        df = {"machine": 10, "learning": 5}
        N = 100
        
        weights = idf_weight(query_terms, df, N)
        
        assert "machine" in weights
        assert "learning" in weights
        assert weights["machine"] > 0
        assert weights["learning"] > 0
        # Less frequent term should have higher weight
        assert weights["learning"] > weights["machine"]
    
    def test_idf_weight_floor(self):
        """Test IDF weight with floor value."""
        query_terms = ["rare"]
        df = {"rare": 1}
        N = 100
        floor = 2.0
        
        weights = idf_weight(query_terms, df, N, floor=floor)
        
        assert weights["rare"] >= floor
    
    def test_idf_weight_unseen_term(self):
        """Test IDF weight for unseen terms."""
        query_terms = ["unseen"]
        df = {}
        N = 100
        
        weights = idf_weight(query_terms, df, N)
        
        assert "unseen" in weights
        assert weights["unseen"] > 0


class TestCoverageScore:
    """Test coverage score calculation."""
    
    def test_coverage_all_terms_present(self):
        """Test coverage when all query terms are present."""
        query_terms = ["machine", "learning"]
        doc_terms = ["machine", "learning", "algorithms", "deep"]
        idf = {"machine": 2.0, "learning": 2.0}
        
        score = coverage_score(query_terms, doc_terms, idf)
        
        assert score > 0
        assert score <= 1.0
    
    def test_coverage_partial_match(self):
        """Test coverage when only some terms are present."""
        query_terms = ["machine", "learning"]
        doc_terms = ["machine", "algorithms"]
        idf = {"machine": 2.0, "learning": 2.0}
        
        score = coverage_score(query_terms, doc_terms, idf)
        
        assert 0 < score < 1.0
    
    def test_coverage_no_match(self):
        """Test coverage when no terms match."""
        query_terms = ["machine", "learning"]
        doc_terms = ["biology", "chemistry"]
        idf = {"machine": 2.0, "learning": 2.0}
        
        score = coverage_score(query_terms, doc_terms, idf)
        
        assert score == 0.0
    
    def test_coverage_empty_query(self):
        """Test coverage with empty query."""
        query_terms = []
        doc_terms = ["machine", "learning"]
        idf = {}
        
        score = coverage_score(query_terms, doc_terms, idf)
        
        assert score == 0.0


class TestProximityScore:
    """Test proximity score calculation."""
    
    def test_proximity_adjacent_terms(self):
        """Test proximity when terms are adjacent."""
        query_terms = ["machine", "learning"]
        doc_terms = ["machine", "learning", "algorithms"]
        
        score = proximity_score(query_terms, doc_terms, window=10)
        
        assert score > 0
    
    def test_proximity_far_apart(self):
        """Test proximity when terms are far apart."""
        query_terms = ["machine", "learning"]
        doc_terms = ["machine"] + ["filler"] * 50 + ["learning"]
        
        score = proximity_score(query_terms, doc_terms, window=10)
        
        # Score should be lower or zero when terms are far apart
        assert score >= 0
    
    def test_proximity_single_term(self):
        """Test proximity with single query term."""
        query_terms = ["machine"]
        doc_terms = ["machine", "learning", "algorithms"]
        
        score = proximity_score(query_terms, doc_terms, window=10)
        
        # Single term has no proximity to measure
        assert score == 0.0
    
    def test_proximity_no_match(self):
        """Test proximity when terms don't match."""
        query_terms = ["machine", "learning"]
        doc_terms = ["biology", "chemistry"]
        
        score = proximity_score(query_terms, doc_terms, window=10)
        
        assert score == 0.0


class TestPhraseBoost:
    """Test phrase boost calculation."""
    
    def test_phrase_boost_exact_match(self):
        """Test phrase boost with exact match."""
        query = "machine learning"
        doc_text = "This paper discusses machine learning algorithms."
        
        score = phrase_boost(query, doc_text)
        
        assert score > 0
    
    def test_phrase_boost_no_match(self):
        """Test phrase boost with no match."""
        query = "machine learning"
        doc_text = "This paper discusses biology and chemistry."
        
        score = phrase_boost(query, doc_text)
        
        assert score == 0.0
    
    def test_phrase_boost_case_insensitive(self):
        """Test phrase boost is case insensitive."""
        query = "Machine Learning"
        doc_text = "MACHINE LEARNING is a subfield of AI."
        
        score = phrase_boost(query, doc_text)
        
        assert score > 0
