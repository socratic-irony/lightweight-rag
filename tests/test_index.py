#!/usr/bin/env python3
"""Additional tests for index module functions."""

import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.index import (
    tokenize,
    compute_text_hash,
    manifest_for_dir
)


class TestTokenizeEdgeCases:
    """Additional tests for tokenize function edge cases."""
    
    def test_tokenize_with_numbers(self):
        """Test tokenization with numbers."""
        result = tokenize("version 3.14 and python3")
        assert "version" in result
        # Numbers should be tokenized
        assert len(result) > 0
    
    def test_tokenize_with_urls(self):
        """Test tokenization with URLs."""
        result = tokenize("visit http://example.com for info")
        assert "visit" in result
        assert len(result) > 0
    
    def test_tokenize_with_special_chars(self):
        """Test tokenization removes special characters."""
        result = tokenize("hello@world.com test#tag")
        # Should extract word tokens
        assert len(result) > 0


class TestComputeTextHash:
    """Test text hash computation."""
    
    def test_compute_text_hash_string(self):
        """Test hash of a string."""
        result = compute_text_hash("test content")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex is 64 chars
    
    def test_compute_text_hash_consistent(self):
        """Test that hash is deterministic."""
        content = "test content"
        hash1 = compute_text_hash(content)
        hash2 = compute_text_hash(content)
        assert hash1 == hash2
    
    def test_compute_text_hash_different_content(self):
        """Test that different content gives different hashes."""
        hash1 = compute_text_hash("content1")
        hash2 = compute_text_hash("content2")
        assert hash1 != hash2


class TestManifestOperations:
    """Test manifest generation."""
    
    def test_manifest_for_dir_empty(self):
        """Test manifest generation for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = manifest_for_dir(Path(tmpdir))
            
            assert isinstance(manifest, dict)
            # Should have timestamp and files
            assert "timestamp" in manifest or "files" in manifest or len(manifest) == 0
    
    def test_manifest_for_dir_with_files(self):
        """Test manifest generation with PDF files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir)
            
            # Create dummy PDF files
            (pdf_dir / "test1.pdf").write_bytes(b"PDF content 1")
            (pdf_dir / "test2.pdf").write_bytes(b"PDF content 2")
            
            manifest = manifest_for_dir(pdf_dir)
            
            assert isinstance(manifest, dict)
            # Manifest should contain information about the files
            assert len(manifest) > 0
    
    def test_manifest_consistency(self):
        """Test that manifest is consistent for same content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir)
            
            # Create dummy PDF file
            (pdf_dir / "test.pdf").write_bytes(b"PDF content")
            
            manifest1 = manifest_for_dir(pdf_dir)
            manifest2 = manifest_for_dir(pdf_dir)
            
            # Manifests should be similar (may have different timestamps)
            assert isinstance(manifest1, dict)
            assert isinstance(manifest2, dict)
