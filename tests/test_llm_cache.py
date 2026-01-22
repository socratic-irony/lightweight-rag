#!/usr/bin/env python3
"""Tests for LLM caching behavior."""

from typing import Any, Dict

import pytest

from lightweight_rag.llm import LLMClient


class _DummyResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return {"choices": [{"message": {"content": self._content}}]}


def test_llm_summary_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "DummyClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, *args: Any, **kwargs: Any) -> _DummyResponse:
            calls.append((args, kwargs))
            return _DummyResponse("summary")

    monkeypatch.setattr("lightweight_rag.llm.httpx.Client", DummyClient)

    cfg = {
        "enabled": True,
        "provider": "openai",
        "base_url": "http://test",
        "model": "test-model",
        "api_key": "test-key",
        "temperature": 0.0,
        "max_tokens": 100,
        "n": 1,
    }

    client = LLMClient(cfg)
    first = client.generate_summary("query", ["a", "b"], max_tokens=50)
    second = client.generate_summary("query", ["a", "b"], max_tokens=50)

    assert first == "summary"
    assert second == "summary"
    assert len(calls) == 1


def test_hyde_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "DummyClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, *args: Any, **kwargs: Any) -> _DummyResponse:
            calls.append((args, kwargs))
            return _DummyResponse("hyde")

    monkeypatch.setattr("lightweight_rag.llm.httpx.Client", DummyClient)

    cfg = {
        "enabled": True,
        "provider": "openai",
        "base_url": "http://test",
        "model": "test-model",
        "api_key": "test-key",
        "temperature": 0.1,
        "max_tokens": 32,
        "n": 1,
    }

    client = LLMClient(cfg)
    first = client.generate_hypothetical_answer("query")
    second = client.generate_hypothetical_answer("query")

    assert first == "hyde"
    assert second == "hyde"
    assert len(calls) == 1
