#!/usr/bin/env python3
"""Integration tests for lightweight-rag using test fixtures."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from the main module
import importlib.util
spec = importlib.util.spec_from_file_location("lightweight_rag", 
    str(Path(__file__).parent.parent / "lightweight-rag.py"))
lightweight_rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lightweight_rag)

# Import test functions
extract_pdf_pages = lightweight_rag.extract_pdf_pages
build_bm25 = lightweight_rag.build_bm25
search_topk = lightweight_rag.search_topk
rm3_expand_query = lightweight_rag.rm3_expand_query

# Import data classes
Chunk = lightweight_rag.Chunk
DocMeta = lightweight_rag.DocMeta


class TestPDFExtraction:
    """Test PDF extraction functionality with mock PDFs."""
    
    @pytest.mark.skip(reason="Complex mocking required - would need real PDF testing framework")
    def test_extract_pdf_pages_with_mock(self):
        """Test PDF page extraction with a mock PDF."""
        pass
    
    @pytest.mark.skip(reason="Complex mocking required - would need real PDF testing framework")  
    def test_extract_pdf_pages_empty_content(self):
        """Test PDF extraction with empty content."""
        pass


class TestBM25Construction:
    """Test BM25 index construction."""
    
    def create_test_corpus(self):
        """Create a test corpus for BM25 testing."""
        meta = DocMeta(
            title="Test Document",
            authors=["Test Author"],
            year=2023,
            doi=None,
            source="/test/doc.pdf"
        )
        
        return [
            Chunk(doc_id=0, source="/test/doc.pdf", page=1, 
                  text="Machine learning algorithms are powerful tools", meta=meta),
            Chunk(doc_id=0, source="/test/doc.pdf", page=2, 
                  text="Deep learning neural networks process data", meta=meta),
            Chunk(doc_id=1, source="/test/doc2.pdf", page=1, 
                  text="Natural language processing uses machine learning", meta=meta),
            Chunk(doc_id=1, source="/test/doc2.pdf", page=2, 
                  text="Computer vision algorithms detect patterns", meta=meta),
        ]
    
    def test_build_bm25_index(self):
        """Test BM25 index construction."""
        corpus = self.create_test_corpus()
        
        # Mock cache to ensure we test the actual build process
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                assert bm25 is not None
                assert len(tokenized) == len(corpus)
                
                # Check that tokenized documents contain expected tokens
                all_tokens = set()
                for doc_tokens in tokenized:
                    all_tokens.update(doc_tokens)
                
                expected_tokens = {"machine", "learning", "algorithms", "deep", "neural", "data"}
                assert expected_tokens.issubset(all_tokens)
    
    def test_bm25_search_relevance(self):
        """Test that BM25 search returns relevant results."""
        corpus = self.create_test_corpus()
        
        # Mock cache to ensure we test the actual build process
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                # Search for "machine learning"
                query = "machine learning"
                results = search_topk(
                    corpus, bm25, tokenized, query, k=4,
                    prox_window=0, prox_lambda=0, ngram_lambda=0,
                    diversity=False, semantic=False
                )
                
                assert len(results) > 0
                # First result should contain the query terms
                top_result = results[0]
                text_lower = top_result["text"].lower()
                assert "machine" in text_lower or "learning" in text_lower


class TestQueryExpansion:
    """Test query expansion with RM3."""
    
    def test_rm3_expansion(self):
        """Test RM3 query expansion."""
        corpus = [
            Chunk(doc_id=0, source="/test/doc.pdf", page=1, 
                  text="Machine learning algorithms artificial intelligence", meta=None),
            Chunk(doc_id=0, source="/test/doc.pdf", page=2, 
                  text="Deep learning neural networks", meta=None),
            Chunk(doc_id=1, source="/test/doc2.pdf", page=1, 
                  text="Machine learning classification regression", meta=None),
        ]
        
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                original_query = "machine learning"
                expanded = rm3_expand_query(
                    original_query, bm25, tokenized, corpus,
                    fb_docs=2, fb_terms=3, alpha=0.6
                )
                
                # Expanded query should contain original terms
                assert "machine learning" in expanded
                # Should be longer than original (or same if no good expansion terms)
                assert len(expanded) >= len(original_query)
    
    def test_rm3_no_expansion_needed(self):
        """Test RM3 when no expansion is beneficial."""
        corpus = [
            Chunk(doc_id=0, source="/test/doc.pdf", page=1, 
                  text="Completely unrelated content about cooking recipes food", meta=None),
        ]
        
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                original_query = "machine learning"
                expanded = rm3_expand_query(
                    original_query, bm25, tokenized, corpus,
                    fb_docs=1, fb_terms=3, alpha=0.6
                )
                
                # May return original query or expanded with unrelated terms
                # Just check it doesn't crash and contains original
                assert "machine learning" in expanded


class TestSearchFeatures:
    """Test search features like proximity, diversity, etc."""
    
    def create_diverse_corpus(self):
        """Create a corpus with multiple documents for diversity testing."""
        corpus = []
        for doc_id in range(3):
            for page in range(2):
                meta = DocMeta(
                    title=f"Document {doc_id}",
                    authors=[f"Author {doc_id}"],
                    year=2023,
                    doi=None,
                    source=f"/test/doc{doc_id}.pdf"
                )
                chunk = Chunk(
                    doc_id=doc_id, 
                    source=f"/test/doc{doc_id}.pdf", 
                    page=page+1,
                    text=f"machine learning algorithm research document {doc_id} page {page+1}",
                    meta=meta
                )
                corpus.append(chunk)
        return corpus
    
    def test_diversity_control(self):
        """Test that diversity control limits results per document."""
        corpus = self.create_diverse_corpus()
        
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                # Search with diversity enabled (max 1 per document)
                results_diverse = search_topk(
                    corpus, bm25, tokenized, "machine learning", k=6,
                    diversity=True, max_per_doc=1
                )
                
                # Should get some results
                assert len(results_diverse) > 0
                
                # Check that we get at most 1 result per document
                doc_sources = [r["source"]["file"] for r in results_diverse]  # Fixed: access nested structure
                unique_sources = set(doc_sources)
                assert len(doc_sources) == len(unique_sources)
    
    def test_proximity_bonus(self):
        """Test proximity bonus affects search results."""
        meta = DocMeta(title="Test", authors=[], year=2023, doi=None, source="test.pdf")
        
        corpus = [
            Chunk(doc_id=0, source="/test/doc1.pdf", page=1, 
                  text="machine learning algorithms", meta=meta),  # Close proximity
            Chunk(doc_id=1, source="/test/doc2.pdf", page=1, 
                  text="machine " + "word " * 50 + "learning", meta=meta),  # Far apart
        ]
        
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                # Search with proximity bonus
                results = search_topk(
                    corpus, bm25, tokenized, "machine learning", k=2,
                    prox_window=30, prox_lambda=0.5
                )
                
                # Should get some results
                assert len(results) > 0
                # First result should be the one with close proximity
                assert "algorithms" in results[0]["text"]
    
    def test_ngram_bonus(self):
        """Test n-gram bonus affects search results."""
        meta = DocMeta(title="Test", authors=[], year=2023, doi=None, source="test.pdf")
        
        corpus = [
            Chunk(doc_id=0, source="/test/doc1.pdf", page=1, 
                  text="machine learning is a powerful method", meta=meta),  # Contains bigram
            Chunk(doc_id=1, source="/test/doc2.pdf", page=1, 
                  text="machine and learning are separate concepts", meta=meta),  # No bigram
        ]
        
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                # Search with n-gram bonus
                results = search_topk(
                    corpus, bm25, tokenized, "machine learning", k=2,
                    ngram_lambda=0.3
                )
                
                # Should get some results
                assert len(results) > 0
                # First result should be the one with the n-gram match
                assert "powerful method" in results[0]["text"]


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""
    
    @pytest.mark.skip(reason="Complex async mocking required - would test with real PDFs in practice")
    async def test_build_corpus_mock(self):
        """Test corpus building with mocked PDF extraction."""
        pass
    
    def test_complete_search_pipeline(self):
        """Test the complete search pipeline."""
        # Create test corpus
        meta = DocMeta(
            title="Research Paper",
            authors=["Smith, J.", "Doe, J."],
            year=2023,
            doi="10.1234/example",
            source="/test/paper.pdf"
        )
        
        corpus = [
            Chunk(doc_id=0, source="/test/paper.pdf", page=1, 
                  text="This paper introduces a novel machine learning approach", meta=meta),
            Chunk(doc_id=0, source="/test/paper.pdf", page=2, 
                  text="The method we propose uses deep neural networks", meta=meta),
            Chunk(doc_id=0, source="/test/paper.pdf", page=3, 
                  text="Experimental results show significant improvements", meta=meta),
        ]
        
        # Build index
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                # Perform search with all features
                results = search_topk(
                    corpus, bm25, tokenized, "machine learning method", 
                    k=3, prox_window=30, prox_lambda=0.2, ngram_lambda=0.1,
                    diversity=True, max_per_doc=3, semantic=False
                )
                
                assert len(results) > 0
                
                # Check result structure - updated for actual structure
                result = results[0]
                required_keys = ["text", "source", "score", "citation"]
                for key in required_keys:
                    assert key in result
                
                # Check nested source structure
                assert "file" in result["source"]
                assert "page" in result["source"]
                assert "title" in result["source"]
                
                # Check that metadata is included
                assert result["source"]["title"] == "Research Paper"


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_empty_corpus_handling(self, sample_meta):
        """Test handling of minimal corpus (empty corpus would fail with BM25)."""
        # BM25 can't handle truly empty corpus, so test with minimal corpus
        minimal_corpus = [
            Chunk(doc_id=0, source="/test/doc.pdf", page=1, 
                  text="minimal content", meta=sample_meta)
        ]
        
        # Explicitly avoid cache for this test
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            bm25, tokenized = build_bm25(minimal_corpus)
            
            # Should handle minimal corpus gracefully
            assert len(tokenized) == 1
            assert tokenized[0] == ["minimal", "content"]
            assert bm25 is not None
    
    def test_malformed_query_handling(self):
        """Test handling of malformed queries."""
        meta = DocMeta(title="Test", authors=[], year=2023, doi=None, source="test.pdf")
        
        corpus = [
            Chunk(doc_id=0, source="/test/doc.pdf", page=1, 
                  text="Normal content", meta=meta)
        ]
        
        # Mock cache to avoid interference
        with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
            with patch.object(lightweight_rag, 'save_bm25_to_cache'):
                bm25, tokenized = build_bm25(corpus)
                
                # Test with empty query
                results = search_topk(corpus, bm25, tokenized, "", k=1)
                # Should not crash, may return empty results or all results
                
                # Test with special characters only
                results = search_topk(corpus, bm25, tokenized, "!@#$%", k=1)
                # Should not crash