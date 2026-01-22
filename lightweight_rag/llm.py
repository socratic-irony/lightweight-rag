"""LLM integration for generative query expansion and summaries."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file in the lightweight-rag root
_dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_dotenv_path, override=False)


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
        self.referer = config.get("referer")
        self.title = config.get("title")
        self.extra_body = config.get("extra_body", {})
        self.last_hyde_debug: Optional[Dict[str, str]] = None
        self.last_summary_debug: Optional[Dict[str, str]] = None
        self._last_raw_response: Optional[Dict[str, Any]] = None

    def _chat_completion(self, prompt: str, max_tokens: int, n: int) -> Optional[List[str]]:
        """Call the configured LLM provider and return a list of response strings."""
        extra_headers = {}
        if self.referer:
            extra_headers["HTTP-Referer"] = self.referer
        if self.title:
            extra_headers["X-Title"] = self.title

        message_content = (
            [{"type": "text", "text": prompt}] if self.provider == "openrouter" else prompt
        )

        # Prefer OpenAI client if available (matches OpenRouter example)
        last_exc = None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            completion = client.chat.completions.create(
                extra_headers=extra_headers,
                extra_body=self.extra_body or {},
                model=self.model,
                messages=[{"role": "user", "content": message_content}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                n=n,
            )
            raw = completion.model_dump() if hasattr(completion, "model_dump") else None
            choices = getattr(completion, "choices", None)
            if not choices:
                self._last_raw_response = raw
                return None
            passages = []
            for choice in choices:
                if not choice.message:
                    continue
                content = (choice.message.content or "").strip()
                if content:
                    passages.append(content)
                    continue
                reasoning = getattr(choice.message, "reasoning", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    passages.append(reasoning.strip())
                    continue
                reasoning_details = getattr(choice.message, "reasoning_details", None)
                if isinstance(reasoning_details, list):
                    for detail in reasoning_details:
                        summary = detail.get("summary") if isinstance(detail, dict) else None
                        if isinstance(summary, str) and summary.strip():
                            passages.append(summary.strip())
                            break
            self._last_raw_response = raw
            return passages if passages else None
        except Exception as exc:
            last_exc = exc

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **extra_headers,
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "n": n,
            "stream": False,
        }
        if self.extra_body:
            payload.update(self.extra_body)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()
                self._last_raw_response = data
                passages = []
                if "choices" in data:
                    for choice in data["choices"]:
                        message = choice.get("message", {})
                        content = (message.get("content") or "").strip()
                        if content:
                            passages.append(content)
                            continue
                        reasoning = message.get("reasoning")
                        if isinstance(reasoning, str) and reasoning.strip():
                            passages.append(reasoning.strip())
                            continue
                        reasoning_details = message.get("reasoning_details")
                        if isinstance(reasoning_details, list):
                            for detail in reasoning_details:
                                summary = detail.get("summary") if isinstance(detail, dict) else None
                                if isinstance(summary, str) and summary.strip():
                                    passages.append(summary.strip())
                                    break
                return passages if passages else None
        except Exception as e:
            if last_exc is not None:
                raise last_exc
            raise e

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
            self.last_hyde_debug = {
                "prompt": prompt,
                "summary": cached,
                "error": "",
                "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                if self._last_raw_response
                else "",
            }
            return cached

        prompt = (
            f"Write a short, concise hypothetical passage that answers the following question. "
            f"Focus on including relevant technical keywords and terminology. "
            f"Do not explain what you are doing, just write the passage.\n\n"
            f"Question: {query}\n\n"
            f"Passage:"
        )

        try:
            passages = self._chat_completion(prompt, self.max_tokens, self.n)
            if not passages:
                self.last_hyde_debug = {
                    "prompt": prompt,
                    "summary": "",
                    "error": "Empty or missing content in LLM response",
                    "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                    if self._last_raw_response
                    else "",
                }
                return None

            result = " ".join(passages)
            _hyde_cache[cache_key] = result
            self.last_hyde_debug = {
                "prompt": prompt,
                "summary": result,
                "error": "",
                "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                if self._last_raw_response
                else "",
            }
            return result
                
        except Exception as e:
            print(f"LLM generation failed (continuing with standard search): {e}")
            self.last_hyde_debug = {
                "prompt": prompt,
                "summary": "",
                "error": str(e),
                "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                if self._last_raw_response
                else "",
            }
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
        context = "\n\n".join(f"- {chunk}" for chunk in chunks)
        prompt = (
            "You are a research assistant. Provide a concise summary answer using ONLY the provided context and NOTHING ELSE.\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Do NOT use any external knowledge. If the answer is not explicitly contained in the provided context, you MUST state: \"The provided context does not contain enough information to adequately answer the question.\"\n"
            "- NO introductions, NO conclusions, and NO meta-commentary.\n"
            "- STRICTLY NO markdown formatting, NO bolding (**), and NO headings (#).\n"
            "- Return the result as a SINGLE PARAGRAPH only.\n"
            "- 1 to 6 sentences maximum.\n"
            "- Be direct and factual.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        cached = _summary_cache.get(cache_key)
        if cached:
            self.last_summary_debug = {
                "prompt": prompt,
                "summary": cached,
                "error": "",
                "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                if self._last_raw_response
                else "",
            }
            return cached

        try:
            passages = self._chat_completion(prompt, summary_max_tokens, 1)
            if passages:
                # Normalize response: strip linebreaks and multiple spaces to ensure a single paragraph
                content = " ".join(passages[0].split())
                if not content:
                    error_msg = "Empty content in LLM response"
                    if self._last_raw_response:
                        choices = self._last_raw_response.get("choices", [])
                        if choices and choices[0].get("finish_reason") == "length":
                            error_msg = "Token limit reached (reasoning was too long). Increased summary.max_tokens in config."
                    self.last_summary_debug = {
                        "prompt": prompt,
                        "summary": "",
                        "error": error_msg,
                        "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                        if self._last_raw_response
                        else "",
                    }
                    return None
                _summary_cache[cache_key] = content
                self.last_summary_debug = {
                    "prompt": prompt,
                    "summary": content,
                    "error": "",
                    "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                    if self._last_raw_response
                    else "",
                }
                return content
            
            # No passages returned
            error_msg = "Empty or missing content in LLM response"
            if self._last_raw_response:
                choices = self._last_raw_response.get("choices", [])
                if choices and choices[0].get("finish_reason") == "length":
                    error_msg = "Token limit reached (reasoning was too long). Increased summary.max_tokens in config."

            self.last_summary_debug = {
                "prompt": prompt,
                "summary": "",
                "error": error_msg,
                "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                if self._last_raw_response
                else "",
            }
        except Exception as e:
            print(f"LLM summary failed (continuing without summary): {e}")
            self.last_summary_debug = {
                "prompt": prompt,
                "summary": "",
                "error": str(e),
                "raw_response": json.dumps(self._last_raw_response, default=str)[:4000]
                if self._last_raw_response
                else "",
            }
            return None

        return None
