#!/usr/bin/env python3
"""Tests for pipeline-level contracts (config immutability, async API, etc.)."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.config import get_default_config


class TestConfigImmutabilityContract:
    """P2-7: _run_rag_pipeline_internal must not mutate the caller's config dict."""

    def _build_llm_enabled_config(self) -> dict:
        """Return a minimal config with LLM enabled so the hyde_queries path is exercised."""
        cfg = get_default_config()
        cfg["llm"]["enabled"] = True
        cfg["llm"]["hyde_queries_before"] = "sentinel"
        return cfg

    def test_caller_config_not_mutated_after_pipeline(self, monkeypatch):
        """The caller's config dict must be identical before and after run_rag_pipeline."""
        from lightweight_rag import main as main_mod

        # Stub out the heavy parts so we don't need real PDFs or LLM access.
        async def fake_build_corpus(*args, **kwargs):
            return []

        monkeypatch.setattr(main_mod, "build_corpus", fake_build_corpus)

        cfg = get_default_config()
        cfg_snapshot = copy.deepcopy(cfg)

        import asyncio

        asyncio.run(main_mod.run_rag_pipeline(cfg, "test query"))

        assert cfg == cfg_snapshot, (
            "run_rag_pipeline mutated the caller's config dict. "
            "Keys that changed: "
            + str({k for k in cfg if cfg.get(k) != cfg_snapshot.get(k)})
        )

    def test_caller_config_hyde_key_not_injected(self, monkeypatch):
        """hyde_queries must not appear in the caller's config after the pipeline runs."""
        from lightweight_rag import main as main_mod
        from lightweight_rag.llm import LLMClient

        async def fake_build_corpus(*args, **kwargs):
            return []

        def fake_hyde(self, query):
            return ["passage one", "passage two", "passage three"]

        monkeypatch.setattr(main_mod, "build_corpus", fake_build_corpus)
        monkeypatch.setattr(LLMClient, "generate_hypothetical_answers", fake_hyde)

        cfg = get_default_config()
        cfg["llm"]["enabled"] = True

        import asyncio

        asyncio.run(main_mod.run_rag_pipeline(cfg, "test query"))

        assert "hyde_queries" not in cfg.get("llm", {}), (
            "run_rag_pipeline leaked hyde_queries into the caller's config dict."
        )
