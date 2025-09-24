#!/usr/bin/env python3
"""Test configuration for lightweight-rag tests."""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add the parent directory to sys.path to import lightweight_rag package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from modular package
from lightweight_rag.models import DocMeta, Chunk
from lightweight_rag.index import (
    load_bm25_from_cache, save_bm25_to_cache,
    load_corpus_from_cache, save_corpus_to_cache, 
    load_manifest, save_manifest
)


@pytest.fixture
def mock_caching():
    """Mock caching functions to avoid file system dependencies."""
    with patch('lightweight_rag.index.load_bm25_from_cache', return_value=None):
        with patch('lightweight_rag.index.save_bm25_to_cache'):
            with patch('lightweight_rag.index.load_corpus_from_cache', return_value=None):
                with patch('lightweight_rag.index.save_corpus_to_cache'):
                    with patch('lightweight_rag.index.load_manifest', return_value=None):
                        with patch('lightweight_rag.index.save_manifest'):
                            yield


@pytest.fixture
def sample_meta():
    """Provide sample metadata for testing."""
    return DocMeta(
        title="Test Document",
        authors=["Smith, J.", "Doe, A."],
        year=2023,
        doi="10.1234/test.doi",
        source="/test/document.pdf"
    )


@pytest.fixture
def sample_corpus(sample_meta):
    """Provide sample corpus for testing."""
    return [
        Chunk(doc_id=0, source="/test/doc1.pdf", page=1, 
              text="Machine learning algorithms are powerful tools", meta=sample_meta),
        Chunk(doc_id=0, source="/test/doc1.pdf", page=2, 
              text="Deep learning neural networks process data", meta=sample_meta),
        Chunk(doc_id=1, source="/test/doc2.pdf", page=1, 
              text="Natural language processing uses machine learning", meta=sample_meta),
    ]