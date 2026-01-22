"""LLM integration for generative query expansion and summaries."""

import json
import os
from typing import Any, Dict, Optional, List

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


_hyde_cache: Dict[str, str] = {}
_summary_cache: Dict[str, str] = {}


def _cache_key(config: Dict[str, Any], query: str, purpose: str) -> str:
    model = config.get("model", "")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 0)
    n = config.get("n", 1)
    provider = config.get("provider", "")
    return f"{purpose}|{provider}|{model}|{temperature}|{max_tokens}|{n}|{query}"


class LLMClient:
    """Simple client for OpenAI-compatible LLM APIs (including Ollama and OpenRouter)."""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.provider = config.get("provider", "openai")
        self.base_url = config.get("base_url", "http://localhost:11434/v1").rstrip("/")
        
        # Load API key from env var if specified, else use config value, else default
        env_var_name = config.get("api_key_env")
        if env_var_name and os.getenv(env_var_name):
            self.api_key = os.getenv(env_var_name)
        else:
            self.api_key = config.get("api_key", "ollama")
            
        self.model = config.get("model", "llama3.2:1b")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 150)
        self.n = config.get("n", 1)
        self.timeout = 20.0  # Increased timeout for external APIs/multiple generations

    def generate_hypothetical_answer(self, query: str) -> Optional[str]:
        """
        Generate hypothetical answer(s) to the query.
        Returns None if disabled or if the call fails.
        """
        if not self.enabled:
            return None

        cache_key = _cache_key(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "n": self.n,
                "provider": self.provider,
            },
            query,
            "hyde",
        )
        cached = _hyde_cache.get(cache_key)
        if cached:
            return cached

        prompt = (
            f"Write a short, concise hypothetical passage that answers the following question. "
            f"Focus on including relevant technical keywords and terminology. "
            f"Do not explain what you are doing, just write the passage.\n\n"
            f"Question: {query}\n\n"
            f"Passage:"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Add OpenRouter specific headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/lightweight-rag"
            headers["X-Title"] = "Lightweight RAG"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "stream": False,
        }
        
        # Add extra_body for OpenRouter/specific models
        if self.provider == "openrouter":
             payload["extra_body"] = {"reasoning": {"enabled": False}}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                # Collect all choices
                passages = []
                if "choices" in data:
                    for choice in data["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            passages.append(choice["message"]["content"].strip())
                
                if not passages:
                    return None
                    
                # Join them with spaces to form one large expansion context
                result = " ".join(passages)
                _hyde_cache[cache_key] = result
                return result
                
        except Exception as e:
            print(f"LLM generation failed (continuing with standard search): {e}")
            return None

    def generate_summary(self, query: str, chunks: List[str], max_tokens: Optional[int] = None) -> Optional[str]:
        """Generate a concise summary from retrieved chunks."""
        if not self.enabled:
            return None

        summary_max_tokens = max_tokens if max_tokens is not None else max(64, int(self.max_tokens))
        cache_key = _cache_key(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": summary_max_tokens,
                "n": 1,
                "provider": self.provider,
            },
            query + "||" + "|".join(chunks),
            "summary",
        )
        cached = _summary_cache.get(cache_key)
        if cached:
            return cached

        context = "\n\n".join(f"- {chunk}" for chunk in chunks)
        prompt = (
            "You are a research assistant. Summarize the relevant information below and answer the user's question.\n"
            "Be concise and factual. If the answer isn't contained in the context, say so.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/lightweight-rag"
            headers["X-Title"] = "Lightweight RAG"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": summary_max_tokens,
            "n": 1,
            "stream": False,
        }
        if self.provider == "openrouter":
            payload["extra_body"] = {"reasoning": {"enabled": False}}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"].strip()
                    _summary_cache[cache_key] = content
                    return content
        except Exception as e:
            print(f"LLM summary failed (continuing without summary): {e}")
            return None

        return None
