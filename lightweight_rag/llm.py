"""LLM integration for generative query expansion."""

import json
import os
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
                return " ".join(passages)
                
        except Exception as e:
            print(f"LLM generation failed (continuing with standard search): {e}")
            return None
