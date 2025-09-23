#!/usr/bin/env python3
"""Test script to verify PDF text quality improvements."""

import sys
import os

# Add the project root to the path so we can import lightweight_rag
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightweight_rag.io_pdf import is_text_quality_good, clean_text


def test_text_quality_checks():
    """Test the text quality validation functions."""
    
    print("Testing text quality validation...\n")
    
    # Test cases
    test_cases = [
        # Good quality text
        ("Good quality text", "This is a normal academic paper with proper text extraction.", True),
        
        # Bad quality text with control characters (like from your example)
        ("Control characters", "\u0000\u0001\u0002\u0003\u0001\u0005\u0006\u0007\b\t \u0005\u000b\u0001\u0005\f\b\t\u0005\r\u000e\u000f", False),
        
        # Mixed good and bad
        ("Mixed quality", "This is normal text \u0000\u0001\u0002 with some control chars.", False),
        
        # Empty text
        ("Empty text", "", False),
        
        # Very short text
        ("Too short", "Hi", False),
        
        # Repeated characters (encoding issue indicator)
        ("Repeated patterns", "aaaaaaaaaa bbbbbbbbb cccccccccc dddddddddd eeeeeeeeee", False),
        
        # Normal text with some repetition (should pass)
        ("Normal repetition", "The the the method is described in detail. We propose a new approach.", True),
        
        # Unicode text (should pass after cleaning)
        ("Unicode text", "This contains some unicode: café, résumé, naïve", True),
        
        # Your actual example (should fail)
        ("Your example", "\u0000\u0001\u0002\u0003\u0001\u0005\u0006\u0007\b\t \u0005\u000b\u0001\u0005\f\b\t\u0005\r\u000e\u000f\u0005\u0010\u000e\u0006 \u000e\u0011\u0005\u0012\u0013\u0014\u0015\u0001\u0005\u0016\u0017\u0018\b\u0019\u001a\b\u001b\u0006\t\u001c\u0005\u001b\u001d\u000e\u0005\u001e\u0006\u0003 \u001f\u0007\u0005\u001f \u0005\u0003\u001b\u000f\b\t\u001c\u000e\u000f\u0003!\u0005\"\u001d\u000e\u0005#\u000e\u000f$\u000e\u0006\u0018\u000e \u0005$\u000f\u000e \u0006%\u0006\u0019\u0006\u001b&\u0005\u001f \u0005\u001f\t\u0019\u0006\t\u000e $\u001f\t\u0003\u001a\u0007\u000e\u000f\u000f\u000e\u0018\u0006\u000e\u001e\u0003\u0005\u001f\t\u0005\u0000\u000e\u0019#\u0011", False),
    ]
    
    for name, text, expected in test_cases:
        # Test raw text
        result = is_text_quality_good(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} {name}: {'PASS' if result else 'FAIL'} (expected {'PASS' if expected else 'FAIL'})")
        
        # Test cleaned text
        cleaned = clean_text(text)
        cleaned_result = is_text_quality_good(cleaned)
        print(f"    After cleaning: {'PASS' if cleaned_result else 'FAIL'}")
        
        # Show first 100 chars of cleaned text for context
        if cleaned and len(cleaned) > 0:
            preview = cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
            print(f"    Cleaned preview: '{preview}'")
        print()


if __name__ == "__main__":
    test_text_quality_checks()