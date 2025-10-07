#!/usr/bin/env python3
"""Tests for fusion module utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightweight_rag.fusion as fusion
from lightweight_rag.fusion import (
    build_ranking_runs,
    fused_diversity_selection,
    robustify_query,
    rrf_fuse,
)
from lightweight_rag.models import Chunk, DocMeta


def make_meta(title: str) -> DocMeta:
    """Create minimal document metadata for test chunks."""

    return DocMeta(
        title=title,
        authors=["Doe, Jane"],
        year=2024,
        doi=None,
        source=f"{title}.pdf",
    )


def make_chunk(doc_id: int, text: str) -> Chunk:
    """Create a chunk with shared metadata helpers."""

    return Chunk(
        doc_id=doc_id,
        source=f"doc{doc_id}.pdf",
        page=1,
        text=text,
        meta=make_meta(f"doc{doc_id}"),
    )


class DummyBM25:
    """Minimal BM25 stub tracking incoming queries."""

    def __init__(self):
        self.calls = []

    def get_scores(self, tokens):
        self.calls.append(tuple(tokens))
        if "expansion" in tokens:
            return [0.7, 0.9, 0.1]
        if "ai" in tokens:
            return [0.9, 0.6, 0.3]
        return [0.7, 0.5, 0.4]


def test_rrf_fuse_combines_runs():
    """RRF should prioritize documents appearing high across runs."""

    runs = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
    fused = rrf_fuse(runs, C=10, cap=3)

    assert fused[:3] == [1, 0, 2]


def test_robustify_query_normalizes_punctuation():
    """Punctuation and casing should be normalized consistently."""

    assert robustify_query("AI & Ethics?!") == "ai ethics"


def test_build_ranking_runs_with_optional_features(monkeypatch):
    """PRF, heuristic, semantic, and robust runs should all be produced."""

    corpus = [
        make_chunk(0, "Machine learning for AI"),
        make_chunk(0, "AI systems overview"),
        make_chunk(1, "General discussion"),
    ]

    bm25 = DummyBM25()
    tokenized = [["machine", "learning"], ["ai", "systems"], ["general", "discussion"]]
    pool = [0, 1, 2]
    baseline_scores = [0.9, 0.6, 0.3]

    config = {
        "prf": {"enabled": True, "fb_docs": 1, "fb_terms": 1, "alpha": 0.5},
        "rerank": {
            "heuristic": {"enabled": True, "topn": 2},
            "semantic": {"enabled": True, "topn": 2},
        },
        "fusion": {"robust_query": {"enabled": True}},
        "diversity": {"enabled": False},
    }

    monkeypatch.setattr(
        "lightweight_rag.prf.rm3_expand_query",
        lambda *a, **k: "AI?? expansion",
    )
    monkeypatch.setattr(
        fusion,
        "semantic_rerank",
        lambda query, texts, scores, **kwargs: [0.1, 0.9][: len(scores)],
    )

    runs = build_ranking_runs("AI??", corpus, bm25, tokenized, pool, baseline_scores, config)

    assert len(runs) == 5
    assert runs[0] == [0, 1, 2]
    assert any(run == [1, 0, 2] for run in runs)  # RM3
    assert any(len(run) == 2 for run in runs)  # Heuristic top-n slice
    assert any(run[0] == 1 for run in runs if len(run) == 3)  # Semantic reorder


def test_fused_diversity_selection_respects_doc_limits():
    """Diversity selection should enforce per-doc caps."""

    corpus = [
        make_chunk(0, "Primary"),
        make_chunk(0, "Secondary"),
        make_chunk(1, "Alt"),
        make_chunk(2, "More"),
    ]
    baseline_scores = [1.0, 0.95, 0.8, 0.7]
    fused_candidates = [0, 1, 2, 3]
    config = {"diversity": {"enabled": True, "per_doc_penalty": 0.5, "max_per_doc": 1}}

    selected = fused_diversity_selection(fused_candidates, corpus, baseline_scores, 3, config)

    assert len(selected) == 3
    doc_ids = [corpus[idx].doc_id for idx in selected]
    assert doc_ids.count(0) == 1
    assert set(doc_ids) == {0, 1, 2}


def test_fused_diversity_selection_passthrough_when_disabled():
    """When disabled, diversity selection should return the top-k slice."""

    corpus = [make_chunk(0, "Primary"), make_chunk(1, "Alt")]
    baseline_scores = [1.0, 0.5]
    fused_candidates = [0, 1]
    config = {"diversity": {"enabled": False}}

    assert fused_diversity_selection(fused_candidates, corpus, baseline_scores, 1, config) == [0]
