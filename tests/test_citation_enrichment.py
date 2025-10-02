#!/usr/bin/env python3
"""Tests for citation enrichment functionality (OpenAlex, Unpaywall, and combined metadata)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from lightweight_rag.cite import (
    openalex_meta_for_doi, unpaywall_meta_for_doi, 
    enriched_meta_for_doi_cached, _build_docmeta_from_cache
)
from lightweight_rag.models import DocMeta


class TestOpenAlexAPI:
    """Test OpenAlex API integration."""
    
    @pytest.mark.asyncio
    async def test_openalex_successful_response(self):
        """Test successful OpenAlex API response parsing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "host_venue": {
                "display_name": "Nature",
                "publisher": "Springer Nature"
            },
            "concepts": [
                {"display_name": "Machine Learning", "level": 1, "score": 0.8},
                {"display_name": "Computer Science", "level": 0, "score": 0.9},
                {"display_name": "Obscure Detail", "level": 3, "score": 0.2}  # Should be filtered
            ],
            "open_access": {
                "oa_url": "https://example.com/paper.pdf"
            }
        }
        
        with patch.object(httpx.AsyncClient, 'get', return_value=mock_response):
            client = httpx.AsyncClient()
            result = await openalex_meta_for_doi(client, "10.1234/example")
            
            assert result is not None
            assert result["venue"] == "Nature"
            assert result["publisher"] == "Springer Nature"
            assert "Machine Learning" in result["concepts"]
            assert "Computer Science" in result["concepts"] 
            assert "Obscure Detail" not in result["concepts"]  # Filtered by level
            assert result["oa_url"] == "https://example.com/paper.pdf"
    
    @pytest.mark.asyncio
    async def test_openalex_primary_location_fallback(self):
        """Test fallback to primary_location for venue and pdf_url."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "primary_location": {
                "source": {"display_name": "arXiv"},
                "pdf_url": "https://arxiv.org/pdf/1234.5678.pdf"
            },
            "concepts": [],
            "open_access": {}
        }
        
        with patch.object(httpx.AsyncClient, 'get', return_value=mock_response):
            client = httpx.AsyncClient()
            result = await openalex_meta_for_doi(client, "10.1234/example")
            
            assert result is not None
            assert result["venue"] == "arXiv"
            assert result["oa_url"] == "https://arxiv.org/pdf/1234.5678.pdf"
    
    @pytest.mark.asyncio
    async def test_openalex_api_error(self):
        """Test OpenAlex API error handling."""
        with patch.object(httpx.AsyncClient, 'get', side_effect=httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())):
            client = httpx.AsyncClient()
            result = await openalex_meta_for_doi(client, "10.1234/nonexistent")
            
            assert result is None


class TestUnpaywallAPI:
    """Test Unpaywall API integration."""
    
    @pytest.mark.asyncio
    async def test_unpaywall_successful_response(self):
        """Test successful Unpaywall API response parsing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://pubmed.gov/paper.pdf"
            }
        }
        
        with patch.object(httpx.AsyncClient, 'get', return_value=mock_response):
            client = httpx.AsyncClient()
            result = await unpaywall_meta_for_doi(client, "10.1234/example", "test@example.com")
            
            assert result is not None
            assert result["oa_url"] == "https://pubmed.gov/paper.pdf"
            assert result["is_oa"] is True
    
    @pytest.mark.asyncio
    async def test_unpaywall_no_oa_available(self):
        """Test Unpaywall response when no OA is available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "is_oa": False,
            "best_oa_location": None
        }
        
        with patch.object(httpx.AsyncClient, 'get', return_value=mock_response):
            client = httpx.AsyncClient()
            result = await unpaywall_meta_for_doi(client, "10.1234/example")
            
            assert result is not None
            assert result["oa_url"] is None
            assert result["is_oa"] is False
    
    @pytest.mark.asyncio
    async def test_unpaywall_api_error(self):
        """Test Unpaywall API error handling."""
        with patch.object(httpx.AsyncClient, 'get', side_effect=httpx.RequestError("Connection failed")):
            client = httpx.AsyncClient()
            result = await unpaywall_meta_for_doi(client, "10.1234/nonexistent")
            
            assert result is None


class TestEnrichedMetadata:
    """Test combined metadata enrichment functionality."""
    
    def test_build_docmeta_from_cache(self):
        """Test building DocMeta from cached data."""
        cached_data = {
            "crossref": {
                "title": "Test Paper", 
                "authors": ["Smith, J.", "Doe, A."],
                "year": 2023,
                "start_page": 100
            },
            "openalex": {
                "venue": "Nature",
                "publisher": "Springer",
                "concepts": ["Machine Learning", "AI"], 
                "oa_url": "https://openalex.org/paper.pdf"
            },
            "unpaywall": {
                "oa_url": "https://unpaywall.org/paper.pdf",  # Should take precedence
                "is_oa": True
            }
        }
        
        result = _build_docmeta_from_cache("10.1234/test", cached_data)
        
        assert result is not None
        assert result.title == "Test Paper"
        assert result.authors == ["Smith, J.", "Doe, A."]
        assert result.year == 2023
        assert result.start_page == 100
        assert result.venue == "Nature"
        assert result.publisher == "Springer"
        assert result.concepts == ["Machine Learning", "AI"]
        assert result.oa_url == "https://unpaywall.org/paper.pdf"  # Unpaywall preferred
    
    def test_build_docmeta_from_cache_openalex_oa_fallback(self):
        """Test fallback to OpenAlex OA URL when Unpaywall has none."""
        cached_data = {
            "crossref": {"title": "Test Paper", "authors": [], "year": 2023},
            "openalex": {"oa_url": "https://openalex.org/paper.pdf"},
            "unpaywall": {"oa_url": None}
        }
        
        result = _build_docmeta_from_cache("10.1234/test", cached_data)
        
        assert result is not None
        assert result.oa_url == "https://openalex.org/paper.pdf"
    
    def test_build_docmeta_from_cache_minimal_data(self):
        """Test building DocMeta with minimal cached data."""
        cached_data = {
            "crossref": {"title": "Minimal Paper", "authors": [], "year": None}
        }
        
        result = _build_docmeta_from_cache("10.1234/test", cached_data)
        
        assert result is not None
        assert result.title == "Minimal Paper"
        assert result.authors == []
        assert result.year is None
        assert result.venue is None
        assert result.concepts is None
        assert result.oa_url is None
    
    @pytest.mark.asyncio 
    async def test_enriched_meta_cache_fresh(self):
        """Test that fresh cache data is used instead of API calls."""
        with patch('lightweight_rag.cite.is_doi_cache_fresh', return_value=True):
            with patch('lightweight_rag.cite.get_cached_doi_metadata', return_value={
                "crossref": {"title": "Cached Paper", "authors": [], "year": 2023}
            }):
                client = httpx.AsyncClient()
                result = await enriched_meta_for_doi_cached(client, "10.1234/test")
                
                assert result is not None
                assert result.title == "Cached Paper"
    
    @pytest.mark.asyncio
    async def test_enriched_meta_selective_apis(self):
        """Test selective API usage based on configuration."""
        with patch('lightweight_rag.cite.is_doi_cache_fresh', return_value=False):
            with patch('lightweight_rag.cite._fetch_enriched_uncached') as mock_fetch:
                mock_fetch.return_value = DocMeta(
                    title="Test", authors=[], year=2023, doi="10.1234/test", source=""
                )
                
                client = httpx.AsyncClient()
                await enriched_meta_for_doi_cached(
                    client, "10.1234/test",
                    use_crossref=True, use_openalex=False, use_unpaywall=False
                )
                
                mock_fetch.assert_called_once()
                args = mock_fetch.call_args[0]
                assert args[3] is True   # use_crossref
                assert args[4] is False  # use_openalex  
                assert args[5] is False  # use_unpaywall


class TestDocMetaEnhancements:
    """Test the enhanced DocMeta model."""
    
    def test_docmeta_all_fields(self):
        """Test DocMeta with all enriched fields."""
        meta = DocMeta(
            title="Enhanced Paper",
            authors=["Author, First"],
            year=2023,
            doi="10.1234/test",
            source="test.pdf",
            start_page=42,
            venue="Nature Machine Intelligence", 
            publisher="Springer Nature",
            concepts=["Machine Learning", "Computer Vision"],
            oa_url="https://example.com/paper.pdf"
        )
        
        assert meta.title == "Enhanced Paper"
        assert meta.venue == "Nature Machine Intelligence"
        assert meta.publisher == "Springer Nature"
        assert meta.concepts == ["Machine Learning", "Computer Vision"]
        assert meta.oa_url == "https://example.com/paper.pdf"
    
    def test_docmeta_default_values(self):
        """Test DocMeta with default values for new fields."""
        meta = DocMeta(
            title="Basic Paper",
            authors=["Author, First"], 
            year=2023,
            doi="10.1234/test",
            source="test.pdf"
        )
        
        assert meta.venue is None
        assert meta.publisher is None
        assert meta.concepts is None


class TestAuthorDateCitation:
    """Test author-date citation formatting."""
    
    def test_author_date_with_page(self):
        """Test author-date citation with page number."""
        from lightweight_rag.cite import author_date_citation
        
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
            doi="10.1234/test",
            source="test.pdf"
        )
        
        citation = author_date_citation(meta, 42)
        assert "Smith" in citation
        assert "2023" in citation
        assert "42" in citation
    
    def test_author_date_without_page(self):
        """Test author-date citation without page number."""
        from lightweight_rag.cite import author_date_citation
        
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi="10.1234/test",
            source="test.pdf"
        )
        
        citation = author_date_citation(meta, None)
        assert "Smith" in citation
        assert "2023" in citation
    
    def test_author_date_no_authors(self):
        """Test author-date citation with no authors."""
        from lightweight_rag.cite import author_date_citation
        
        meta = DocMeta(
            title="Test Paper",
            authors=[],
            year=2023,
            doi="10.1234/test",
            source="test.pdf"
        )
        
        citation = author_date_citation(meta, 42)
        assert "2023" in citation


class TestBatchLookup:
    """Test batch lookup functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_enriched_lookup_empty(self):
        """Test batch enriched lookup with empty DOI list."""
        from lightweight_rag.cite import batch_enriched_lookup
        
        client = httpx.AsyncClient()
        results = await batch_enriched_lookup(client, [], cache_seconds=1)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_batch_enriched_lookup_with_errors(self):
        """Test batch enriched lookup handles errors gracefully."""
        from lightweight_rag.cite import batch_enriched_lookup
        
        with patch('lightweight_rag.cite.enriched_meta_for_doi_cached', 
                   side_effect=Exception("API Error")):
            client = httpx.AsyncClient()
            results = await batch_enriched_lookup(
                client, ["10.1234/test"], cache_seconds=1, max_concurrent=1
            )
            
            # Should return None for failed lookups
            assert len(results) == 1
            assert results[0] is None