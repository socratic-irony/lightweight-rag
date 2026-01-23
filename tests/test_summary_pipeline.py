#!/usr/bin/env python3
"""Tests for summary generation in the main pipeline."""

import pytest

from lightweight_rag import config as config_module
from lightweight_rag import main


@pytest.mark.asyncio
async def test_summary_uses_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config_module.get_default_config()
    cfg["llm"]["enabled"] = True
    cfg["llm"]["summary"]["enabled"] = True
    cfg["llm"]["summary"]["top_k"] = 2
    cfg["llm"]["summary"]["max_tokens"] = 50
    cfg["rerank"]["final_top_k"] = 1

    async def fake_build_corpus(*args, **kwargs):
        return ["dummy"]

    def fake_build_bm25(*args, **kwargs):
        return None, None

    def fake_search_topk(**kwargs):
        return [
            {"text": "one"},
            {"text": "two"},
            {"text": "three"},
        ], {"level": "high", "score": 1.0, "spread": 0.5, "stability": 1.0}

    captured = {}

    class StubLLM:
        def __init__(self, _cfg):
            self.last_summary_debug = None
            self.last_hyde_debug = None

        def generate_hypothetical_answer(self, _query):
            return None

        def generate_hypothetical_answers(self, _query):
            return []

        def generate_summary(self, _query, chunks, max_tokens=None):
            captured["chunks"] = chunks
            captured["max_tokens"] = max_tokens
            return "summary"

    monkeypatch.setattr(main, "build_corpus", fake_build_corpus)
    monkeypatch.setattr(main, "build_bm25", fake_build_bm25)
    monkeypatch.setattr(main, "search_topk", fake_search_topk)
    monkeypatch.setattr(main, "update_cache_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("lightweight_rag.llm.LLMClient", StubLLM)

    output = await main.run_rag_pipeline_with_summary(cfg, "query")

    assert output["summary"] == "summary"
    assert output["results"] == [{"text": "one"}]
    assert captured["chunks"] == ["one", "two"]
    assert captured["max_tokens"] == 50
