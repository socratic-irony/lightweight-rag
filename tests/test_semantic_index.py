#!/usr/bin/env python3
"""Tests for semantic ANN index helpers."""

from lightweight_rag.semantic_index import ann_select_candidates


def test_ann_select_candidates_handles_empty_input():
    assert ann_select_candidates([], "query", 10, "model") is None
    assert ann_select_candidates(["a"], "query", 0, "model") is None
