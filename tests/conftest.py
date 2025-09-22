#!/usr/bin/env python3
"""Test configuration for lightweight-rag tests."""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add the parent directory to sys.path to import lightweight-rag
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from the main module
import importlib.util
spec = importlib.util.spec_from_file_location("lightweight_rag", 
    str(Path(__file__).parent.parent / "lightweight-rag.py"))
lightweight_rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lightweight_rag)


@pytest.fixture(autouse=True)
def mock_caching():
    """Automatically mock caching functions to avoid file system dependencies."""
    with patch.object(lightweight_rag, 'load_bm25_from_cache', return_value=None):
        with patch.object(lightweight_rag, 'save_bm25_to_cache'):
            with patch.object(lightweight_rag, 'load_corpus_from_cache', return_value=None):
                with patch.object(lightweight_rag, 'save_corpus_to_cache'):
                    with patch.object(lightweight_rag, 'load_manifest', return_value=None):
                        with patch.object(lightweight_rag, 'save_manifest'):
                            yield


@pytest.fixture
def sample_meta():
    """Provide sample metadata for testing."""
    return lightweight_rag.DocMeta(
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
        lightweight_rag.Chunk(doc_id=0, source="/test/doc1.pdf", page=1, 
                             text="Machine learning algorithms are powerful tools", meta=sample_meta),
        lightweight_rag.Chunk(doc_id=0, source="/test/doc1.pdf", page=2, 
                             text="Deep learning neural networks process data", meta=sample_meta),
        lightweight_rag.Chunk(doc_id=1, source="/test/doc2.pdf", page=1, 
                             text="Natural language processing uses machine learning", meta=sample_meta),
    ]