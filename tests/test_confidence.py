#!/usr/bin/env python3
"""Tests for confidence calibration in the main pipeline."""

import pytest
from lightweight_rag.main import calibrate_confidence

def test_calibrate_confidence_no_results():
    """Test confidence with no results."""
    conf = calibrate_confidence([], [], [], top_k=8)
    assert conf["level"] == "low"
    assert conf["score"] == 0.0

def test_calibrate_confidence_high_separation():
    """Test confidence with high score separation and perfect stability."""
    # Top score is 10, median is 1. Large spread.
    scores = {0: 10.0, 1: 5.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
    # Stability: same top results in both runs
    runs = [[0, 1, 2], [0, 1, 2]]
    pool = [0, 1, 2, 3, 4, 5, 6]
    
    conf = calibrate_confidence(scores, runs, pool, top_k=3)
    
    assert conf["level"] == "high"
    assert conf["score"] > 0.75
    assert conf["spread"] > 0.8  # (10-1)/10 = 0.9
    assert conf["stability"] == 1.0

def test_calibrate_confidence_low_stability():
    """Test confidence with low stability across runs."""
    scores = {i: 1.0 for i in range(10)} # No separation
    # No overlap in top results
    runs = [[0, 1, 2], [3, 4, 5]]
    pool = list(range(10))
    
    conf = calibrate_confidence(scores, runs, pool, top_k=3)
    
    assert conf["level"] == "low"
    assert conf["stability"] == 0.0
    assert conf["spread"] == 0.0

def test_calibrate_confidence_medium():
    """Test confidence with some spread and moderate stability."""
    # Spread approx (5-1)/5 = 0.8
    scores = {0: 5.0, 1: 4.0, 2: 1.0, 3: 1.0, 4: 1.0}
    # 2 out of 3 overlap: intersection={0,1}, union={0,1,2,3} -> 2/4 = 0.5
    runs = [[0, 1, 2], [0, 1, 3]]
    pool = [0, 1, 2, 3, 4]
    
    conf = calibrate_confidence(scores, runs, pool, top_k=3)
    
    # spread=0.8, stability=0.5 -> score = 0.5*1.0 + 0.5*0.5 = 0.75 (threshold for high is >0.75)
    # So this should be medium
    assert conf["level"] == "medium"
    assert 0.7 <= conf["score"] <= 0.75
