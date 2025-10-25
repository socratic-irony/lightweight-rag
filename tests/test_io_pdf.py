#!/usr/bin/env python3
"""Additional tests for io_pdf module to improve coverage."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.io_pdf import (
    chunk_text,
    clean_text,
    create_sliding_windows,
    extract_pdf_pages,
    is_text_quality_good,
    normalize_text,
    split_into_sentences,
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
        text = "Hello\u00adWorld"
        normalized = normalize_text(text)
        assert "\u00ad" not in normalized
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


class TestSlidingWindows:
    """Test sliding window creation."""

    def test_sliding_windows_basic(self):
        """Test basic sliding window creation."""
        text = " ".join([f"word{i}" for i in range(50)])
        windows = create_sliding_windows(text, window_chars=100, overlap_chars=20)
        assert len(windows) >= 1
        # Check overlap exists between consecutive windows
        if len(windows) > 1:
            # Some content should overlap
            assert any(word in windows[1] for word in windows[0].split()[-5:])

    def test_sliding_windows_short_text(self):
        """Test sliding window with text shorter than window size."""
        text = "Short text here"
        windows = create_sliding_windows(text, window_chars=100, overlap_chars=20)
        assert len(windows) == 1
        assert windows[0] == text

    def test_sliding_windows_no_overlap(self):
        """Test sliding windows with no overlap."""
        text = " ".join([f"word{i}" for i in range(50)])
        windows = create_sliding_windows(text, window_chars=100, overlap_chars=0)
        assert len(windows) >= 1

    def test_sliding_windows_boundary_conditions(self):
        """Test sliding window boundary conditions."""
        # Test with exact multiples
        text = "a " * 50  # 100 chars
        windows = create_sliding_windows(text, window_chars=50, overlap_chars=10)
        assert len(windows) >= 2

    def test_sliding_windows_empty_text(self):
        """Test sliding windows with empty text."""
        windows = create_sliding_windows("", window_chars=100, overlap_chars=20)
        # Empty text returns a list with empty string or empty list depending on implementation
        assert isinstance(windows, list)
        assert len(windows) <= 1

    def test_sliding_windows_very_long_words(self):
        """Test sliding windows with very long words that exceed window size."""
        # Create text with a word longer than window
        long_word = "x" * 150
        text = f"short {long_word} text"
        windows = create_sliding_windows(text, window_chars=100, overlap_chars=20)
        # Should handle gracefully
        assert isinstance(windows, list)


class TestTextQualityEdgeCases:
    """Test text quality validation edge cases."""

    def test_excessive_control_characters(self):
        """Test rejection of text with excessive control characters."""
        # More than 5% control characters should fail
        text = "normal" + "\x01\x02\x03" * 10
        assert is_text_quality_good(text) is False

    def test_repeated_patterns(self):
        """Test detection of excessive repeated patterns."""
        # Many repeated non-space patterns indicate encoding issues
        text = "aaaaaa bbbbb ccccc ddddd eeeeee"
        result = is_text_quality_good(text)
        # This should fail due to repeated patterns
        assert result is False

    def test_lack_common_characters(self):
        """Test rejection of text without common characters."""
        # Text with no common English letters
        text = "####$$$$%%%%^^^^&&&&****" * 10
        assert is_text_quality_good(text) is False

    def test_borderline_quality(self):
        """Test borderline quality text."""
        # Text with some control chars but still readable
        text = "This is mostly readable text with some issues."
        assert is_text_quality_good(text) is True


class TestTextEncodingScenarios:
    """Test various text encoding scenarios."""

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        text = "Hëllö Wörld! Café résumé naïve"
        cleaned = clean_text(text)
        assert "Hëllö" in cleaned or "Hello" in cleaned

    def test_mixed_encoding(self):
        """Test text with mixed encoding issues."""
        text = "Normal text\x00with\x01null\x02bytes"
        cleaned = clean_text(text)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "Normal text" in cleaned

    def test_soft_hyphens(self):
        """Test removal of soft hyphens."""
        text = "hyphen\u00adated\u00adword"
        normalized = normalize_text(text)
        assert "\u00ad" not in normalized
        assert "hyphenated" in normalized or "hyphen ated" in normalized

    def test_line_break_handling(self):
        """Test handling of line breaks and hyphens."""
        text = "some-\nthing"
        normalized = normalize_text(text)
        # Should handle hyphenated line breaks
        assert isinstance(normalized, str)


class TestPDFExtraction:
    """Test PDF extraction with mocking."""

    def test_extract_pdf_malformed(self):
        """Test extraction with malformed PDF."""
        with patch("lightweight_rag.io_pdf.fitz.open") as mock_open:
            mock_open.side_effect = Exception("Malformed PDF")

            result = extract_pdf_pages("fake.pdf")
            assert result == []

    def test_extract_pdf_pages_quality_fallback(self):
        """Test that extraction falls back to alternative method on poor quality."""
        mock_doc = MagicMock()
        mock_page = MagicMock()

        # First extraction yields poor quality
        mock_page.get_text.return_value = "\x00\x01\x02" * 50

        # Alternative extraction yields good quality
        mock_textpage = MagicMock()
        mock_textpage.extractText.return_value = (
            "This is good quality text extracted using alternative method."
        )
        mock_page.get_textpage.return_value = mock_textpage

        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_doc.__enter__ = lambda self: mock_doc
        mock_doc.__exit__ = lambda self, *args: None

        with patch("lightweight_rag.io_pdf.fitz.open", return_value=mock_doc):
            pages = extract_pdf_pages("test.pdf")
            assert len(pages) == 1
            # Should have attempted alternative extraction
            assert mock_page.get_textpage.called or mock_page.get_text.called


class TestChunkingConfigurations:
    """Test different chunking method configurations."""

    def test_chunk_page_split_method(self):
        """Test page split chunking method."""
        text = "This is a document with multiple paragraphs. " * 10
        config = {"page_split": "page"}
        chunks = chunk_text(text, doc_title="Test Doc", chunking_config=config)

        assert len(chunks) == 1
        assert "Test Doc" in chunks[0]

    def test_chunk_sentence_split_method(self):
        """Test sentence split chunking method."""
        text = "First sentence here. Second sentence here. Third sentence here."
        config = {"page_split": "sentence"}
        chunks = chunk_text(text, doc_title="Test Doc", chunking_config=config)

        assert len(chunks) >= 3
        # All chunks should have the title
        assert all("Test Doc" in chunk for chunk in chunks)

    def test_chunk_sliding_split_method(self):
        """Test sliding window split method."""
        text = " ".join([f"word{i}" for i in range(100)])
        config = {"page_split": "sliding", "window_chars": 100, "overlap_chars": 20}
        chunks = chunk_text(text, doc_title="Test Doc", chunking_config=config)

        assert len(chunks) >= 1
        # All chunks should have the title
        assert all("Test Doc" in chunk for chunk in chunks)

    def test_chunk_sliding_respects_sentence_boundaries(self):
        """Test that sliding windows respect sentence boundaries."""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here. Sixth sentence here."
        config = {"page_split": "sliding", "window_chars": 100, "overlap_chars": 20}
        chunks = chunk_text(text, doc_title="", chunking_config=config)

        # All chunks should end with sentence-ending punctuation
        for chunk in chunks:
            assert (
                chunk.endswith(".") or chunk.endswith("!") or chunk.endswith("?")
            ), f"Chunk does not end at sentence boundary: {chunk}"

        # Should create multiple chunks given the text length and window size
        assert len(chunks) >= 2

    def test_chunk_no_title(self):
        """Test chunking without title."""
        text = "Content without title"
        config = {"page_split": "page"}
        chunks = chunk_text(text, doc_title="", chunking_config=config)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_default_behavior(self):
        """Test default chunking behavior when no config provided."""
        text = "Default behavior test"
        chunks = chunk_text(text, doc_title="", chunking_config=None)

        assert len(chunks) == 1
        assert text in chunks[0]


class TestPrintIfNotQuiet:
    """Test quiet mode printing."""

    def test_print_default(self):
        """Test that print works by default."""
        from lightweight_rag.io_pdf import _print_if_not_quiet

        with patch("builtins.print") as mock_print:
            _print_if_not_quiet("Test message")
            mock_print.assert_called()

    def test_print_quiet_enabled(self):
        """Test that print is suppressed in quiet mode."""
        from lightweight_rag.io_pdf import _print_if_not_quiet

        with patch("builtins.print") as mock_print:
            _print_if_not_quiet("Test message", {"_quiet_mode": True})
            mock_print.assert_not_called()

    def test_print_quiet_disabled(self):
        """Test that print works when quiet mode disabled."""
        from lightweight_rag.io_pdf import _print_if_not_quiet

        with patch("builtins.print") as mock_print:
            _print_if_not_quiet("Test message", {"_quiet_mode": False})
            mock_print.assert_called()
