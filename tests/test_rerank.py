#!/usr/bin/env python3
"""Tests for rerank module functions."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightweight_rag.rerank as rerank
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


class TestHeuristicRerank:
    """Integration tests for heuristic reranking."""

    def test_heuristic_rerank_prioritizes_rich_matches(self):
        """Candidates with better coverage should bubble to the top."""
        candidates = [
            {"text": "Machine learning fundamentals", "bm25": 0.2, "rank": 0, "index": 0},
            {"text": "Chemistry overview", "bm25": 0.9, "rank": 1, "index": 1},
        ]

        reranked = rerank.heuristic_rerank(
            "machine learning",
            candidates,
            df={"machine": 2, "learning": 3},
            N=100,
            alpha=0.7,
            beta=0.2,
            gamma=0.1,
        )

        assert reranked[0]["index"] == 0
        assert all("rerank_score" in candidate for candidate in reranked)

    def test_heuristic_rerank_preserves_order_for_empty_query(self):
        """Empty queries should short-circuit and leave candidates untouched."""
        candidates = [
            {"text": "Item A", "bm25": 0.5, "rank": 0, "index": 0},
            {"text": "Item B", "bm25": 0.4, "rank": 1, "index": 1},
        ]

        reranked = rerank.heuristic_rerank("", candidates)

        assert reranked is candidates


class TestEmbeddingHelpers:
    """Tests for embedding helpers that back semantic reranking."""

    def test_embed_texts_normalizes_vectors(self, monkeypatch):
        """Embeddings should be L2-normalized even with zero vectors present."""

        class DummyModel:
            def encode(self, texts, convert_to_numpy=True):
                return np.array([[3.0, 4.0], [0.0, 0.0]], dtype=float)

        monkeypatch.setattr(rerank, "SentenceTransformer", object)
        monkeypatch.setattr(rerank, "np", np)
        monkeypatch.setattr(rerank, "_load_model", lambda name: DummyModel())

        embeddings = rerank.embed_texts(["alpha", "beta"])

        assert embeddings is not None
        assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)
        assert np.array_equal(embeddings[1], np.zeros(2, dtype=float))

    def test_embed_texts_handles_encode_failures(self, monkeypatch):
        """Failures while encoding should result in a graceful fallback."""

        class FailingModel:
            def encode(self, texts, convert_to_numpy=True):  # noqa: D401
                raise RuntimeError("encode boom")

        monkeypatch.setattr(rerank, "SentenceTransformer", object)
        monkeypatch.setattr(rerank, "np", np)
        monkeypatch.setattr(rerank, "_load_model", lambda name: FailingModel())

        assert rerank.embed_texts(["alpha"]) is None

    def test_chunk_embeddings_uses_cache_and_executor(self, monkeypatch):
        """Chunk encoding should reuse cached vectors and populate new ones."""

        monkeypatch.setattr(
            rerank,
            "_embedding_cache",
            {"cached": np.array([1.0, 0.0], dtype=float)},
        )

        encoded = []

        def fake_encode(model, text):
            encoded.append(text)
            return np.array([float(len(text)), 0.0], dtype=float)

        monkeypatch.setattr(rerank, "_encode_single_text", fake_encode)

        results = rerank._chunk_embeddings(["cached", "fresh"], object(), max_workers=1)

        assert results is not None
        assert len(results) == 2
        assert encoded == ["fresh"]
        assert np.array_equal(rerank._embedding_cache["fresh"], np.array([5.0, 0.0]))


class TestSemanticRerank:
    """Semantic reranking combinations and fallbacks."""

    def test_semantic_rerank_combines_scores(self, monkeypatch):
        """Semantic scores should mix with normalized lexical scores."""

        class DummyModel:
            pass

        def fake_embed_texts(texts, model_name):
            if len(texts) == 1:
                return np.array([[1.0, 0.0]], dtype=float)
            return np.vstack([np.array([1.0, 0.0], dtype=float) for _ in texts])

        def fake_chunk_embeddings(texts, model, max_workers):
            return [
                np.array([1.0, 0.0], dtype=float),
                np.array([0.0, 1.0], dtype=float),
            ]

        monkeypatch.setattr(rerank, "SentenceTransformer", object)
        monkeypatch.setattr(rerank, "np", np)
        monkeypatch.setattr(rerank, "_load_model", lambda name: DummyModel())
        monkeypatch.setattr(rerank, "embed_texts", fake_embed_texts)
        monkeypatch.setattr(rerank, "_chunk_embeddings", fake_chunk_embeddings)

        combined = rerank.semantic_rerank(
            "query",
            ["first", "second"],
            [0.2, 0.6],
            model_name="dummy",
            max_workers=1,
        )

        assert combined != [0.2, 0.6]
        assert len(combined) == 2
        assert combined[1] > combined[0]

    def test_semantic_rerank_returns_original_when_unavailable(self):
        """Missing semantic model should return the original scores."""

        original = [0.5, 0.7]

        assert rerank.semantic_rerank("query", ["text"], original) is original
    
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
