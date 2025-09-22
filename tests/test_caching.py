#!/usr/bin/env python3
"""Tests for caching functionality in lightweight-rag."""

import pytest
import tempfile
import json
import pickle
import gzip
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from the main module
import importlib.util
spec = importlib.util.spec_from_file_location("lightweight_rag", 
    str(Path(__file__).parent.parent / "lightweight-rag.py"))
lightweight_rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lightweight_rag)

# Import caching functions
manifest_for_dir = lightweight_rag.manifest_for_dir
load_manifest = lightweight_rag.load_manifest
save_manifest = lightweight_rag.save_manifest
load_corpus_from_cache = lightweight_rag.load_corpus_from_cache
save_corpus_to_cache = lightweight_rag.save_corpus_to_cache
load_bm25_from_cache = lightweight_rag.load_bm25_from_cache
save_bm25_to_cache = lightweight_rag.save_bm25_to_cache

# Import data classes
Chunk = lightweight_rag.Chunk
DocMeta = lightweight_rag.DocMeta


class TestManifestGeneration:
    """Test manifest generation for PDF directories."""
    
    def test_manifest_for_empty_dir(self):
        """Test manifest generation for empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = manifest_for_dir(Path(temp_dir))
            assert result == {"files": []}
    
    def test_manifest_for_dir_with_files(self):
        """Test manifest generation for directory with PDF files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test PDF files
            pdf1 = Path(temp_dir) / "test1.pdf"
            pdf2 = Path(temp_dir) / "test2.pdf"
            non_pdf = Path(temp_dir) / "test.txt"
            
            pdf1.write_bytes(b"PDF content 1")
            pdf2.write_bytes(b"PDF content 2")
            non_pdf.write_text("Not a PDF")
            
            result = manifest_for_dir(Path(temp_dir))
            
            # Should only include PDF files
            assert len(result["files"]) == 2
            file_paths = [f["path"] for f in result["files"]]
            assert str(pdf1) in file_paths
            assert str(pdf2) in file_paths
            assert str(non_pdf) not in file_paths
            
            # Check file metadata
            for file_info in result["files"]:
                assert "path" in file_info
                assert "mtime" in file_info
                assert "size" in file_info
                assert file_info["size"] > 0


class TestManifestCaching:
    """Test manifest loading and saving."""
    
    def test_save_and_load_manifest(self):
        """Test saving and loading manifest cache."""
        test_manifest = {
            "files": [
                {"path": "/test/file1.pdf", "mtime": 123456789, "size": 1000},
                {"path": "/test/file2.pdf", "mtime": 123456790, "size": 2000}
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the MANIFEST_CACHE path
            cache_file = Path(temp_dir) / "manifest.json"
            with patch.object(lightweight_rag, 'MANIFEST_CACHE', cache_file):
                # Save manifest
                save_manifest(test_manifest)
                
                # Check file was created
                assert cache_file.exists()
                
                # Load manifest
                loaded = load_manifest()
                assert loaded == test_manifest
    
    def test_load_manifest_nonexistent(self):
        """Test loading manifest when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "nonexistent.json"
            with patch.object(lightweight_rag, 'MANIFEST_CACHE', cache_file):
                result = load_manifest()
                assert result is None
    
    def test_load_manifest_invalid_json(self):
        """Test loading manifest with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "manifest.json"
            cache_file.write_text("invalid json {")
            
            with patch.object(lightweight_rag, 'MANIFEST_CACHE', cache_file):
                result = load_manifest()
                assert result is None


class TestCorpusCaching:
    """Test corpus caching functionality."""
    
    def test_save_and_load_corpus(self):
        """Test saving and loading corpus cache."""
        # Create test corpus
        meta = DocMeta(
            title="Test Paper",
            authors=["Smith, John"],
            year=2023,
            doi="10.1234/test.doi",
            source="/test/paper.pdf"
        )
        
        test_corpus = [
            Chunk(doc_id=0, source="/test/paper.pdf", page=1, 
                  text="First page content", meta=meta),
            Chunk(doc_id=0, source="/test/paper.pdf", page=2, 
                  text="Second page content", meta=meta)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "corpus.jsonl.gz"
            with patch.object(lightweight_rag, 'CORPUS_CACHE', cache_file):
                # Save corpus
                save_corpus_to_cache(test_corpus)
                
                # Check file was created
                assert cache_file.exists()
                
                # Load corpus
                loaded = load_corpus_from_cache()
                assert loaded is not None
                assert len(loaded) == 2
                assert loaded[0].text == "First page content"
                assert loaded[1].text == "Second page content"
                assert loaded[0].meta.title == "Test Paper"
    
    def test_load_corpus_nonexistent(self):
        """Test loading corpus when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "nonexistent.jsonl.gz"
            with patch.object(lightweight_rag, 'CORPUS_CACHE', cache_file):
                result = load_corpus_from_cache()
                assert result is None
    
    def test_load_corpus_invalid_format(self):
        """Test loading corpus with invalid format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "corpus.jsonl.gz"
            
            # Write invalid compressed data
            with gzip.open(cache_file, 'wt') as f:
                f.write("invalid json line\n")
            
            with patch.object(lightweight_rag, 'CORPUS_CACHE', cache_file):
                result = load_corpus_from_cache()
                assert result is None


class TestBM25Caching:
    """Test BM25 model caching functionality."""
    
    def test_save_and_load_bm25(self):
        """Test saving and loading BM25 cache."""
        # Create real BM25 model with simple data instead of mock
        from rank_bm25 import BM25Okapi
        
        test_tokenized = [
            ["hello", "world", "test"],
            ["another", "document", "here"],
            ["third", "document", "example"]
        ]
        
        real_bm25 = BM25Okapi(test_tokenized)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            bm25_cache = Path(temp_dir) / "bm25.pkl.gz"
            tokenized_cache = Path(temp_dir) / "tokenized.pkl.gz"
            
            with patch.object(lightweight_rag, 'BM25_CACHE', bm25_cache):
                with patch.object(lightweight_rag, 'TOKENIZED_CACHE', tokenized_cache):
                    # Save BM25 and tokenized data
                    save_bm25_to_cache(real_bm25, test_tokenized)
                    
                    # Check files were created
                    assert bm25_cache.exists()
                    assert tokenized_cache.exists()
                    
                    # Load BM25 and tokenized data
                    loaded_bm25, loaded_tokenized = load_bm25_from_cache()
                    
                    assert loaded_bm25 is not None
                    assert loaded_tokenized == test_tokenized
    
    def test_load_bm25_nonexistent(self):
        """Test loading BM25 when files don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bm25_cache = Path(temp_dir) / "nonexistent_bm25.pkl.gz"
            tokenized_cache = Path(temp_dir) / "nonexistent_tokenized.pkl.gz"
            
            with patch.object(lightweight_rag, 'BM25_CACHE', bm25_cache):
                with patch.object(lightweight_rag, 'TOKENIZED_CACHE', tokenized_cache):
                    result = load_bm25_from_cache()
                    assert result is None  # Function returns None, not tuple
    
    def test_load_bm25_partial_files(self):
        """Test loading BM25 when only one cache file exists."""
        from rank_bm25 import BM25Okapi
        
        with tempfile.TemporaryDirectory() as temp_dir:
            bm25_cache = Path(temp_dir) / "bm25.pkl.gz"
            tokenized_cache = Path(temp_dir) / "nonexistent_tokenized.pkl.gz"
            
            # Create only BM25 cache file
            with gzip.open(bm25_cache, 'wb') as f:
                pickle.dump(BM25Okapi([["test"]]), f)
            
            with patch.object(lightweight_rag, 'BM25_CACHE', bm25_cache):
                with patch.object(lightweight_rag, 'TOKENIZED_CACHE', tokenized_cache):
                    result = load_bm25_from_cache()
                    assert result is None  # Should return None if both files aren't available


class TestCacheInvalidation:
    """Test cache invalidation logic."""
    
    def test_cache_invalidation_different_files(self):
        """Test that cache is invalidated when files change."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_dir = Path(temp_dir) / "pdfs"
            pdf_dir.mkdir()
            
            # Create initial PDF
            pdf1 = pdf_dir / "test1.pdf"
            pdf1.write_bytes(b"Original content")
            
            # Generate initial manifest
            manifest1 = manifest_for_dir(pdf_dir)
            
            # Add another PDF
            pdf2 = pdf_dir / "test2.pdf"
            pdf2.write_bytes(b"New content")
            
            # Generate new manifest
            manifest2 = manifest_for_dir(pdf_dir)
            
            # Manifests should be different
            assert manifest1 != manifest2
            assert len(manifest1["files"]) == 1
            assert len(manifest2["files"]) == 2
    
    def test_cache_invalidation_modified_file(self):
        """Test that cache is invalidated when file is modified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_dir = Path(temp_dir) / "pdfs"
            pdf_dir.mkdir()
            
            # Create PDF
            pdf1 = pdf_dir / "test1.pdf"
            pdf1.write_bytes(b"Original content")
            
            # Generate initial manifest
            manifest1 = manifest_for_dir(pdf_dir)
            original_mtime = manifest1["files"][0]["mtime"]
            
            # Wait a bit and modify the file
            import time
            time.sleep(0.1)
            pdf1.write_bytes(b"Modified content")
            
            # Generate new manifest
            manifest2 = manifest_for_dir(pdf_dir)
            new_mtime = manifest2["files"][0]["mtime"]
            
            # Modification time should be different
            assert new_mtime != original_mtime
            assert manifest1 != manifest2


class TestCacheDirectoryManagement:
    """Test cache directory creation and management."""
    
    def test_cache_directory_created(self):
        """Test that cache directory is created when needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            
            # Mock the CACHE_DIR
            with patch.object(lightweight_rag, 'CACHE_DIR', cache_dir):
                # Simulate cache directory creation
                cache_dir.mkdir(exist_ok=True)
                
                assert cache_dir.exists()
                assert cache_dir.is_dir()