#!/usr/bin/env python3
"""Tests for core utility functions in lightweight-rag."""

import pytest
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

# Import the functions we need to test
tokenize = lightweight_rag.tokenize
window = lightweight_rag.window
ngram_bonus = lightweight_rag.ngram_bonus
proximity_bonus = lightweight_rag.proximity_bonus
pattern_bonus = lightweight_rag.pattern_bonus
find_doi_in_text = lightweight_rag.find_doi_in_text
DOI_RE = lightweight_rag.DOI_RE
ANSWER_PATTERNS = lightweight_rag.ANSWER_PATTERNS
STOP = lightweight_rag.STOP


class TestTokenize:
    """Test the tokenize function."""
    
    def test_basic_tokenization(self):
        """Test basic tokenization with alphanumeric characters."""
        result = tokenize("Hello World 123")
        assert result == ["hello", "world", "123"]
    
    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        result = tokenize("hello, world! test@example.com")
        assert result == ["hello", "world", "test", "example", "com"]
    
    def test_empty_string(self):
        """Test tokenization of empty string."""
        result = tokenize("")
        assert result == []
    
    def test_only_special_characters(self):
        """Test tokenization of string with only special characters."""
        result = tokenize("!@#$%^&*()")
        assert result == []
    
    def test_case_insensitive(self):
        """Test that tokenization is case insensitive."""
        result = tokenize("Hello WORLD Test")
        assert result == ["hello", "world", "test"]


class TestWindow:
    """Test the window function for text truncation."""
    
    def test_short_text_unchanged(self):
        """Test that short text is returned unchanged."""
        text = "This is a short text."
        result = window(text, maxlen=100)
        assert result == text
    
    def test_long_text_truncated(self):
        """Test that long text is truncated with ellipsis."""
        text = "A" * 1000
        result = window(text, maxlen=50)
        assert len(result) == 50
        assert result.endswith("...")
        assert result == "A" * 47 + "..."
    
    def test_newlines_replaced(self):
        """Test that newlines are replaced with spaces."""
        text = "Line 1\nLine 2\nLine 3"
        result = window(text)
        assert "\n" not in result
        assert result == "Line 1 Line 2 Line 3"
    
    def test_whitespace_stripped(self):
        """Test that leading/trailing whitespace is stripped."""
        text = "   Hello World   "
        result = window(text)
        assert result == "Hello World"
    
    def test_exact_maxlen(self):
        """Test text exactly at maxlen."""
        text = "A" * 900  # Default maxlen
        result = window(text)
        assert result == text
    
    def test_custom_maxlen(self):
        """Test with custom maxlen parameter."""
        text = "Hello World"
        result = window(text, maxlen=5)
        assert result == "He..."


class TestNgramBonus:
    """Test the n-gram bonus function."""
    
    def test_no_ngrams_match(self):
        """Test when no n-grams match."""
        text = "The quick brown fox"
        query = "elephant zebra"
        result = ngram_bonus(text, query)
        assert result == 0.0
    
    def test_bigram_match(self):
        """Test when bigrams match."""
        text = "The quick brown fox jumps"
        query = "quick brown"
        result = ngram_bonus(text, query)
        assert result > 0.0
    
    def test_trigram_match(self):
        """Test when trigrams match."""
        text = "The quick brown fox jumps"
        query = "quick brown fox"
        result = ngram_bonus(text, query)
        assert result > 0.0
    
    def test_short_ngrams_ignored(self):
        """Test that n-grams shorter than 5 characters are ignored."""
        text = "a b c d"
        query = "a b"
        result = ngram_bonus(text, query)
        assert result == 0.0
    
    def test_max_hits_respected(self):
        """Test that max_hits parameter is respected."""
        # Create a text with many potential matches
        text = "test case " * 20
        query = "test case " * 10
        result = ngram_bonus(text, query, max_hits=3)
        assert result <= 3.0 / 3.0  # Should be capped at max_hits
    
    def test_case_insensitive(self):
        """Test that n-gram matching is case insensitive."""
        text = "QUICK BROWN FOX"
        query = "quick brown"
        result = ngram_bonus(text, query)
        assert result > 0.0


class TestProximityBonus:
    """Test the proximity bonus function."""
    
    def test_no_query_tokens(self):
        """Test with empty query tokens."""
        text = "Hello world"
        result = proximity_bonus(text, [])
        assert result == 0.0
    
    def test_empty_text(self):
        """Test with empty text."""
        result = proximity_bonus("", ["hello", "world"])
        assert result == 0.0
    
    def test_single_token(self):
        """Test with single query token."""
        text = "Hello world"
        result = proximity_bonus(text, ["hello"])
        assert result == 0.0  # Need at least 2 distinct terms
    
    def test_tokens_close_together(self):
        """Test when query tokens are close together."""
        text = "The quick brown fox"
        query_tokens = ["quick", "brown"]
        result = proximity_bonus(text, query_tokens, window_size=30)
        assert result > 0.0
    
    def test_tokens_far_apart(self):
        """Test when query tokens are far apart."""
        text = "quick " + "word " * 50 + "brown"
        query_tokens = ["quick", "brown"]
        result = proximity_bonus(text, query_tokens, window_size=10)
        assert result == 0.0  # Should be beyond window size
    
    def test_multiple_occurrences(self):
        """Test with multiple occurrences of query tokens."""
        text = "quick test brown fox quick brown"
        query_tokens = ["quick", "brown"]
        result = proximity_bonus(text, query_tokens, window_size=30)
        assert result > 0.0
    
    def test_case_insensitive_proximity(self):
        """Test that proximity is case insensitive."""
        text = "QUICK brown"
        query_tokens = ["quick", "brown"]
        result = proximity_bonus(text, query_tokens, window_size=30)
        assert result > 0.0


class TestPatternBonus:
    """Test the pattern bonus function."""
    
    def test_no_patterns(self):
        """Test text with no answer patterns."""
        text = "This is just regular text without any patterns."
        result = pattern_bonus(text)
        assert result == 0.0
    
    def test_single_pattern(self):
        """Test text with one answer pattern."""
        text = "This is a definition of something."
        result = pattern_bonus(text)
        assert result == 0.05  # One pattern * 0.05
    
    def test_multiple_patterns(self):
        """Test text with multiple answer patterns."""
        text = "This is a method we propose for stakeholders include analysis."
        result = pattern_bonus(text)
        assert result > 0.05  # Multiple patterns
    
    def test_case_insensitive_patterns(self):
        """Test that pattern matching is case insensitive."""
        text = "THIS IS A METHOD WE PROPOSE."
        result = pattern_bonus(text)
        assert result > 0.0
    
    def test_all_patterns_present(self):
        """Test with all answer patterns present."""
        text = " ".join(ANSWER_PATTERNS)
        result = pattern_bonus(text)
        expected = len(ANSWER_PATTERNS) * 0.05
        assert result == expected


class TestDOIRegex:
    """Test DOI detection functionality."""
    
    def test_valid_doi_found(self):
        """Test finding a valid DOI in text."""
        text = "This paper has DOI 10.1234/example.doi.123"
        result = find_doi_in_text(text)
        assert result == "10.1234/example.doi.123"
    
    def test_doi_with_special_chars(self):
        """Test DOI with various special characters."""
        text = "DOI: 10.1000/182(01)12345-6"
        result = find_doi_in_text(text)
        assert result == "10.1000/182(01)12345-6"
    
    def test_no_doi_found(self):
        """Test when no DOI is present."""
        text = "This text has no DOI identifier."
        result = find_doi_in_text(text)
        assert result is None
    
    def test_invalid_doi_format(self):
        """Test with invalid DOI format."""
        text = "10.123/invalid"  # Too few digits in prefix
        result = find_doi_in_text(text)
        assert result is None
    
    def test_doi_with_trailing_punctuation(self):
        """Test DOI that ends with punctuation."""
        text = "See DOI 10.1234/example.doi.123."
        result = find_doi_in_text(text)
        assert result == "10.1234/example.doi.123"  # Should strip trailing period
    
    def test_case_insensitive_doi(self):
        """Test that DOI regex is case insensitive."""
        text = "DOI: 10.1234/Example.DOI.123"
        result = find_doi_in_text(text)
        assert result == "10.1234/Example.DOI.123"


class TestConstants:
    """Test important constants."""
    
    def test_answer_patterns_not_empty(self):
        """Test that ANSWER_PATTERNS is not empty."""
        assert len(ANSWER_PATTERNS) > 0
        assert all(isinstance(pattern, str) for pattern in ANSWER_PATTERNS)
    
    def test_stop_words_not_empty(self):
        """Test that STOP words set is not empty."""
        assert len(STOP) > 0
        assert all(isinstance(word, str) for word in STOP)
    
    def test_doi_regex_compiles(self):
        """Test that DOI regex compiles correctly."""
        assert DOI_RE is not None
        # Test that it matches a basic DOI
        assert DOI_RE.search("10.1234/test") is not None