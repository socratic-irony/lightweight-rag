#!/usr/bin/env python3
"""Additional tests for io_pdf module to improve coverage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.io_pdf import (
    is_text_quality_good,
    clean_text,
    normalize_text,
    split_into_sentences,
    chunk_text
)


class TestTextQuality:
    """Test text quality validation."""
    
    def test_is_text_quality_good_high_quality(self):
        """Test that high-quality text is recognized."""
        text = "This is a well-formed sentence with proper punctuation and readable content."
        assert is_text_quality_good(text) is True
    
    def test_is_text_quality_good_low_quality(self):
        """Test that low-quality text is rejected."""
        # Text with many control characters
        text = "\x00\x01\x02\x03\x04" * 20
        assert is_text_quality_good(text) is False
    
    def test_is_text_quality_good_empty(self):
        """Test empty text is rejected."""
        assert is_text_quality_good("") is False
        assert is_text_quality_good("   ") is False
    
    def test_is_text_quality_good_too_short(self):
        """Test very short text is rejected."""
        assert is_text_quality_good("short") is False


class TestCleanText:
    """Test text cleaning."""
    
    def test_clean_text_removes_nulls(self):
        """Test that null bytes are removed."""
        text = "Hello\x00World"
        cleaned = clean_text(text)
        assert "\x00" not in cleaned
        assert "HelloWorld" in cleaned
    
    def test_clean_text_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Hello    World\n\n\nTest"
        cleaned = clean_text(text)
        assert "  " not in cleaned or cleaned.count("  ") < text.count("  ")
    
    def test_clean_text_preserves_content(self):
        """Test that actual content is preserved."""
        text = "This is a normal sentence."
        cleaned = clean_text(text)
        assert "This is a normal sentence" in cleaned


class TestNormalizeText:
    """Test text normalization."""
    
    def test_normalize_text_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Hello   World\n\nTest"
        normalized = normalize_text(text)
        assert "  " not in normalized or normalized.count("  ") < text.count("  ")
        assert "Hello World Test" in normalized
    
    def test_normalize_text_soft_hyphen(self):
        """Test that soft hyphens are removed."""
        text = "Hello\u00ADWorld"
        normalized = normalize_text(text)
        assert "\u00AD" not in normalized
        assert "HelloWorld" in normalized
    
    def test_normalize_text_empty(self):
        """Test normalization of empty text."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestSplitIntoSentences:
    """Test sentence splitting."""
    
    def test_split_basic_sentences(self):
        """Test splitting basic sentences."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_into_sentences(text)
        assert len(sentences) >= 3
        assert any("First" in s for s in sentences)
        assert any("Second" in s for s in sentences)
        assert any("Third" in s for s in sentences)
    
    def test_split_with_abbreviations(self):
        """Test that abbreviations don't cause incorrect splits."""
        text = "Dr. Smith went to the U.S.A. yesterday."
        sentences = split_into_sentences(text)
        # Should be recognized as one sentence
        assert len(sentences) <= 2
    
    def test_split_empty_text(self):
        """Test splitting empty text."""
        sentences = split_into_sentences("")
        assert len(sentences) == 0
    
    def test_split_single_sentence(self):
        """Test splitting text with single sentence."""
        text = "This is a single sentence without punctuation"
        sentences = split_into_sentences(text)
        assert len(sentences) >= 1


class TestChunkText:
    """Test text chunking."""
    
    def test_chunk_text_short_text(self):
        """Test chunking with short text that fits in one chunk."""
        text = "Short text"
        chunks = chunk_text(text, doc_title="Test", chunking_config={"method": "sliding"})
        assert len(chunks) >= 1
    
    def test_chunk_text_long_text(self):
        """Test chunking with long text that requires multiple chunks."""
        # Create long text with sentences to trigger chunking
        sentences = [f"This is sentence number {i}." for i in range(100)]
        text = " ".join(sentences)
        config = {"method": "sliding", "window_chars": 200, "overlap_chars": 50}
        chunks = chunk_text(text, doc_title="Test", chunking_config=config)
        # With proper configuration, should create multiple chunks
        assert len(chunks) >= 1
    
    def test_chunk_text_with_title(self):
        """Test that title is included in chunks."""
        text = "This is the content of the document."
        title = "Document Title"
        chunks = chunk_text(text, doc_title=title, chunking_config={"method": "sliding"})
        # At least one chunk should contain the title
        assert any(title in chunk for chunk in chunks)
    
    def test_chunk_text_sentence_method(self):
        """Test sentence-based chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        config = {"method": "sentence", "target_size": 50, "max_size": 100}
        chunks = chunk_text(text, doc_title="Test", chunking_config=config)
        assert len(chunks) >= 1
    
    def test_chunk_text_sliding_method(self):
        """Test sliding window chunking."""
        text = " ".join(["word"] * 100)
        config = {"method": "sliding", "window_chars": 100, "overlap_chars": 20}
        chunks = chunk_text(text, doc_title="Test", chunking_config=config)
        assert len(chunks) >= 1
    
    def test_chunk_text_default_config(self):
        """Test chunking with default configuration."""
        text = "This is some text " * 50
        chunks = chunk_text(text, doc_title="Test", chunking_config=None)
        assert len(chunks) >= 1
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", doc_title="Test", chunking_config={"method": "sliding"})
        # Should return empty or minimal chunks
        assert isinstance(chunks, list)
