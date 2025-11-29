#!/usr/bin/env python3
"""Test incremental PDF caching functionality."""

import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from lightweight_rag.io_pdf import build_corpus
from lightweight_rag.index import update_cache_paths, load_corpus_from_cache, load_manifest


class TestIncrementalCaching:
    """Test incremental caching of PDF corpus."""
    
    @pytest.mark.asyncio
    async def test_initial_corpus_build_creates_cache(self):
        """Test that initial corpus build creates cache files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_dir = Path(temp_dir) / "pdfs"
            cache_dir = Path(temp_dir) / "cache"
            pdf_dir.mkdir()
            cache_dir.mkdir()
            
            # Mock PDF extraction to avoid needing real PDFs
            with patch('lightweight_rag.io_pdf.extract_pdf_pages') as mock_extract:
                mock_extract.return_value = [
                    {"page_number": 1, "text": "This is page 1 content from test PDF."}
                ]
                
                # Create a fake PDF file
                test_pdf = pdf_dir / "test1.pdf"
                test_pdf.write_bytes(b"fake pdf content")
                
                # Update cache paths
                update_cache_paths(cache_dir)
                
                # Build corpus for the first time
                corpus = await build_corpus(pdf_dir, max_workers=1)
                
                # Verify corpus was created
                assert len(corpus) == 1
                assert corpus[0].text == "This is page 1 content from test PDF."  # With context enrichment
                
                # Verify cache files were created
                assert (cache_dir / "corpus.jsonl.gz").exists()
                assert (cache_dir / "manifest.json").exists()
    
    @pytest.mark.asyncio 
    async def test_unchanged_files_load_from_cache(self):
        """Test that unchanged PDFs are loaded from cache on subsequent runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_dir = Path(temp_dir) / "pdfs"
            cache_dir = Path(temp_dir) / "cache"
            pdf_dir.mkdir()
            cache_dir.mkdir()
            
            # Mock PDF extraction
            with patch('lightweight_rag.io_pdf.extract_pdf_pages') as mock_extract:
                mock_extract.return_value = [
                    {"page_number": 1, "text": "This is page 1 content from test PDF."}
                ]
                
                # Create a fake PDF file
                test_pdf = pdf_dir / "test1.pdf"
                test_pdf.write_bytes(b"fake pdf content")
                
                # Update cache paths
                update_cache_paths(cache_dir)
                
                # Build corpus for the first time
                corpus1 = await build_corpus(pdf_dir, max_workers=1)
                assert len(corpus1) == 1
                
                # Reset the mock to track second call
                mock_extract.reset_mock()
                
                # Build corpus again - should load from cache
                corpus2 = await build_corpus(pdf_dir, max_workers=1)
                
                # Verify corpus is the same
                assert len(corpus2) == 1
                assert corpus2[0].text == corpus1[0].text
                
                # Verify extract_pdf_pages was not called again
                mock_extract.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_new_file_triggers_incremental_update(self):
        """Test that adding a new PDF file triggers incremental update."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_dir = Path(temp_dir) / "pdfs"
            cache_dir = Path(temp_dir) / "cache"
            pdf_dir.mkdir()
            cache_dir.mkdir()
            
            # Mock PDF extraction with different returns for different files
            def mock_extract_side_effect(pdf_path):
                if "test1.pdf" in pdf_path:
                    return [{"page_number": 1, "text": "Content from test1 PDF."}]
                elif "test2.pdf" in pdf_path:
                    return [{"page_number": 1, "text": "Content from test2 PDF."}]
                return []
            
            with patch('lightweight_rag.io_pdf.extract_pdf_pages', side_effect=mock_extract_side_effect):
                # Create first PDF file
                test_pdf1 = pdf_dir / "test1.pdf"
                test_pdf1.write_bytes(b"fake pdf content 1")
                
                # Update cache paths
                update_cache_paths(cache_dir)
                
                # Build corpus with one file
                corpus1 = await build_corpus(pdf_dir, max_workers=1)
                assert len(corpus1) == 1
                assert corpus1[0].text == "Content from test1 PDF."  # With context enrichment
                
                # Add second PDF file
                test_pdf2 = pdf_dir / "test2.pdf"
                test_pdf2.write_bytes(b"fake pdf content 2")
                
                # Build corpus again - should include both files
                corpus2 = await build_corpus(pdf_dir, max_workers=1)
                
                # Verify both files are in corpus
                assert len(corpus2) == 2
                texts = [chunk.text for chunk in corpus2]
                assert "Content from test1 PDF." in texts  # With context enrichment
                assert "Content from test2 PDF." in texts  # With context enrichment
    
    @pytest.mark.asyncio
    async def test_modified_file_triggers_reprocessing(self):
        """Test that modifying a PDF file triggers reprocessing of that file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_dir = Path(temp_dir) / "pdfs"
            cache_dir = Path(temp_dir) / "cache"
            pdf_dir.mkdir()
            cache_dir.mkdir()
            
            # Track which files were processed
            processed_files = []
            
            def mock_extract_tracking(pdf_path):
                processed_files.append(pdf_path)
                if "modified_content" in str(Path(pdf_path).read_bytes()):
                    return [{"page_number": 1, "text": "Modified content from PDF."}]
                else:
                    return [{"page_number": 1, "text": "Original content from PDF."}]
            
            with patch('lightweight_rag.io_pdf.extract_pdf_pages', side_effect=mock_extract_tracking):
                # Create PDF file
                test_pdf = pdf_dir / "test1.pdf"
                test_pdf.write_bytes(b"original content")
                
                # Update cache paths
                update_cache_paths(cache_dir)
                
                # Build corpus for the first time
                corpus1 = await build_corpus(pdf_dir, max_workers=1)
                assert len(corpus1) == 1
                assert corpus1[0].text == "Original content from PDF."  # With context enrichment
                assert len(processed_files) == 1
                
                # Modify the PDF file (change content and mtime)
                import time
                time.sleep(0.1)  # Ensure different mtime
                test_pdf.write_bytes(b"modified_content")
                
                # Clear processed files list
                processed_files.clear()
                
                # Build corpus again - should reprocess the modified file
                corpus2 = await build_corpus(pdf_dir, max_workers=1)
                
                # Verify the file was reprocessed
                assert len(corpus2) == 1
                assert corpus2[0].text == "Modified content from PDF."  # With context enrichment
                assert len(processed_files) == 1  # Only modified file should be processed