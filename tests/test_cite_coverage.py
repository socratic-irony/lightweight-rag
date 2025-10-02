#!/usr/bin/env python3
"""Additional tests for cite module to improve coverage."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.cite import (
    crossref_meta_for_doi,
    openalex_meta_for_doi,
    batch_crossref_lookup,
    batch_enriched_lookup,
    _print_quiet,
    unpaywall_meta_for_doi,
    author_date_citation,
    pandoc_citation
)
from lightweight_rag.models import DocMeta


class TestCrossrefAPITimeouts:
    """Test Crossref API timeout handling."""
    
    @pytest.mark.asyncio
    async def test_crossref_timeout(self):
        """Test handling of Crossref API timeout."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Request timeout"))
        
        result = await crossref_meta_for_doi(mock_client, "10.1234/test")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_crossref_connection_error(self):
        """Test handling of connection errors."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        
        result = await crossref_meta_for_doi(mock_client, "10.1234/test")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_crossref_http_error(self):
        """Test handling of HTTP errors (404, 500, etc)."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )
        mock_client.get = AsyncMock(return_value=mock_response)
        
        result = await crossref_meta_for_doi(mock_client, "10.1234/notfound")
        assert result is None


class TestOpenAlexAPIErrors:
    """Test OpenAlex API error responses."""
    
    @pytest.mark.asyncio
    async def test_openalex_timeout(self):
        """Test OpenAlex API timeout handling."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        
        result = await openalex_meta_for_doi(mock_client, "10.1234/test")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_openalex_malformed_response(self):
        """Test handling of malformed OpenAlex response."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        result = await openalex_meta_for_doi(mock_client, "10.1234/test")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_openalex_missing_fields(self):
        """Test OpenAlex response with missing optional fields."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        # Minimal response with no venue, publisher, concepts, etc.
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        result = await openalex_meta_for_doi(mock_client, "10.1234/test")
        # Should return dict even if empty or minimal
        assert result is not None
        assert isinstance(result, dict)


class TestInvalidDOIFormats:
    """Test handling of invalid DOI formats."""
    
    @pytest.mark.asyncio
    async def test_empty_doi(self):
        """Test handling of empty DOI."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=Exception("Invalid DOI"))
        
        result = await crossref_meta_for_doi(mock_client, "")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_malformed_doi(self):
        """Test handling of malformed DOI."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=MagicMock(), response=MagicMock()
        )
        mock_client.get = AsyncMock(return_value=mock_response)
        
        result = await crossref_meta_for_doi(mock_client, "invalid-doi-format")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_special_characters_doi(self):
        """Test DOI with special characters."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {}}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        # Should handle DOI with special chars
        result = await crossref_meta_for_doi(mock_client, "10.1234/test(2023)")
        # Returns None if no useful data
        assert result is None or isinstance(result, DocMeta)


class TestBatchProcessingEdgeCases:
    """Test batch processing edge cases."""
    
    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        """Test batch processing with empty DOI list."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        results = await batch_crossref_lookup(mock_client, [])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_batch_single_doi(self):
        """Test batch processing with single DOI."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"title": ["Test"], "author": []}}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        with patch('lightweight_rag.cite.load_doi_cache', return_value={}):
            results = await batch_crossref_lookup(mock_client, ["10.1234/test"])
            assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_batch_mixed_success_failure(self):
        """Test batch processing with mix of successful and failed requests."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        
        async def mock_get(url, timeout):
            if "success" in url:
                mock_response = MagicMock()
                mock_response.json.return_value = {"message": {"title": ["Test"], "author": []}}
                mock_response.raise_for_status = MagicMock()
                return mock_response
            else:
                raise httpx.TimeoutException("Timeout")
        
        mock_client.get = mock_get
        
        with patch('lightweight_rag.cite.load_doi_cache', return_value={}):
            results = await batch_crossref_lookup(
                mock_client, ["10.1234/success", "10.1234/fail"]
            )
            assert len(results) == 2
            # At least one should be None (the failed one)
            assert None in results or any(r is None for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_concurrent_limit(self):
        """Test that batch processing respects concurrent limit."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        call_count = 0
        max_concurrent = 0
        current_concurrent = 0
        
        async def mock_get(url, timeout):
            nonlocal call_count, max_concurrent, current_concurrent
            current_concurrent += 1
            call_count += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            
            await asyncio.sleep(0.01)  # Simulate API call
            
            current_concurrent -= 1
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"title": ["Test"], "author": []}}
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        mock_client.get = mock_get
        
        with patch('lightweight_rag.cite.load_doi_cache', return_value={}):
            dois = [f"10.1234/test{i}" for i in range(10)]
            results = await batch_crossref_lookup(
                mock_client, dois, max_concurrent=3
            )
            assert len(results) == 10
            # Should have limited concurrency
            assert max_concurrent <= 3


class TestCacheLogic:
    """Test cache expiration and invalidation logic."""
    
    @pytest.mark.asyncio
    async def test_cache_fresh_hit(self):
        """Test that fresh cache entries are used."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        
        # Mock cache with fresh entry
        cache_data = {
            "10.1234/test": {
                "timestamp": 9999999999.0,  # Far future
                "title": "Cached Title",
                "authors": ["Author 1"]
            }
        }
        
        with patch('lightweight_rag.cite.load_doi_cache', return_value=cache_data):
            with patch('lightweight_rag.cite.is_doi_cache_fresh', return_value=True):
                # Should use cache without making API call
                results = await batch_crossref_lookup(
                    mock_client, ["10.1234/test"], cache_seconds=3600
                )
                # Should not have called the API
                assert not mock_client.get.called or mock_client.get.call_count == 0


class TestMetadataEnrichment:
    """Test metadata enrichment failures."""
    
    @pytest.mark.asyncio
    async def test_enriched_lookup_all_apis_fail(self):
        """Test enriched lookup when all APIs fail."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        
        with patch('lightweight_rag.cite.load_doi_cache', return_value={}):
            results = await batch_enriched_lookup(
                mock_client,
                ["10.1234/test"],
                use_crossref=True,
                use_openalex=True
            )
            assert len(results) == 1
            # May return a minimal DocMeta or None depending on implementation
            result = results[0]
            if result is not None:
                # If it returns a DocMeta, it should be minimal
                assert isinstance(result, DocMeta)
    
    @pytest.mark.asyncio
    async def test_enriched_lookup_partial_data(self):
        """Test enriched lookup with partial data from different sources."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        
        async def mock_get(url, timeout):
            mock_response = MagicMock()
            if "crossref" in url:
                # Crossref returns basic metadata
                mock_response.json.return_value = {
                    "message": {"title": ["Test Title"], "author": []}
                }
            else:
                # OpenAlex returns additional metadata
                mock_response.json.return_value = {
                    "host_venue": {"display_name": "Nature"},
                    "concepts": []
                }
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        mock_client.get = mock_get
        
        with patch('lightweight_rag.cite.load_doi_cache', return_value={}):
            results = await batch_enriched_lookup(
                mock_client, ["10.1234/test"]
            )
            assert len(results) == 1


class TestQuietMode:
    """Test quiet mode functionality."""
    
    def test_print_quiet_default(self):
        """Test that print_quiet prints by default."""
        with patch('builtins.print') as mock_print:
            _print_quiet("Test message")
            mock_print.assert_called_once_with("Test message")
    
    def test_print_quiet_enabled(self):
        """Test that print_quiet suppresses output when quiet mode enabled."""
        with patch('builtins.print') as mock_print:
            _print_quiet("Test message", config={"_quiet_mode": True})
            mock_print.assert_not_called()
    
    def test_print_quiet_disabled(self):
        """Test that print_quiet prints when quiet mode explicitly disabled."""
        with patch('builtins.print') as mock_print:
            _print_quiet("Test message", config={"_quiet_mode": False})
            mock_print.assert_called_once_with("Test message")


class TestUnpaywallAPI:
    """Test Unpaywall API integration."""
    
    @pytest.mark.asyncio
    async def test_unpaywall_success(self):
        """Test successful Unpaywall API response."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "oa_status": "gold",
            "best_oa_location": {
                "url_for_pdf": "https://example.com/paper.pdf"
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        result = await unpaywall_meta_for_doi(mock_client, "10.1234/test", "test@example.com")
        
        assert result is not None
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_unpaywall_timeout(self):
        """Test Unpaywall timeout handling."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        
        result = await unpaywall_meta_for_doi(mock_client, "10.1234/test", "test@example.com")
        
        assert result is None


class TestCitationFormatting:
    """Test citation string formatting."""
    
    def test_author_date_citation_single_author(self):
        """Test author-date citation with single author."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf"
        )
        
        citation = author_date_citation(meta, page=5)
        
        assert "Smith" in citation
        assert "2023" in citation
        assert "5" in citation or "p." in citation
    
    def test_author_date_citation_multiple_authors(self):
        """Test author-date citation with multiple authors."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John", "Doe, Jane", "Johnson, Bob"],
            year=2023,
            doi=None,
            source="test.pdf"
        )
        
        citation = author_date_citation(meta, page=10)
        
        assert "Smith" in citation
        assert "et al." in citation
        assert "2023" in citation
    
    def test_author_date_citation_no_year(self):
        """Test author-date citation without year."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=None,
            doi=None,
            source="test.pdf"
        )
        
        citation = author_date_citation(meta, page=5)
        
        assert "Smith" in citation
        assert "n.d." in citation
    
    def test_author_date_citation_with_start_page(self):
        """Test author-date citation with start_page offset."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf",
            start_page=100
        )
        
        citation = author_date_citation(meta, page=5)
        
        # Should use actual page (100 + 5 - 1 = 104)
        assert "104" in citation
    
    def test_pandoc_citation_with_citekey(self):
        """Test Pandoc citation with citekey."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf",
            citekey="smith2023test"
        )
        
        citation = pandoc_citation(meta, page=5)
        
        assert citation is not None
        assert "smith2023test" in citation
        assert "@" in citation
    
    def test_pandoc_citation_no_citekey(self):
        """Test Pandoc citation without citekey."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf",
            citekey=None
        )
        
        citation = pandoc_citation(meta, page=5)
        
        assert citation is None
    
    def test_pandoc_citation_with_start_page(self):
        """Test Pandoc citation with start_page offset."""
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi=None,
            source="test.pdf",
            citekey="smith2023test",
            start_page=100
        )
        
        citation = pandoc_citation(meta, page=5)
        
        # Should use actual page (100 + 5 - 1 = 104)
        assert "104" in citation
