#!/usr/bin/env python3
"""Configuration system for lightweight-rag with precedence: defaults → file (YAML) → env vars → CLI flags."""

import os
import yaml
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any, Optional


def get_default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        "paths": {
            "pdf_dir": "pdfs",
            "cache_dir": ".raq_cache",
            "crossref_email": None,
        },
        "indexing": {
            "page_split": "page",
            "window_chars": 1000,
            "overlap_chars": 120,
        },
        "bm25": {
            "k1": 1.5,
            "b": 0.75,
            "build_top_k": 300,
            "token_pattern": "[A-Za-z0-9]+",
        },
        "prf": {
            "enabled": False,
            "fb_docs": 6,
            "fb_terms": 10,
            "alpha": 0.6,
        },
        "bonuses": {
            "proximity": {
                "enabled": True,
                "window": 30,
                "weight": 0.2,
            },
            "ngram": {
                "enabled": True,
                "weight": 0.1,
            },
            "patterns": {
                "enabled": True,
                "patterns": [" is a ", " we define ", " we propose ", " we argue ",
                           " consists of ", " stakeholders include ", " method ", " methodology "],
                "weight_per_hit": 0.05,
                "max_hits": 6,
            },
        },
        "diversity": {
            "enabled": True,
            "per_doc_penalty": 0.3,
            "max_per_doc": 2,
        },
        "rerank": {
            "semantic": {
                "enabled": True,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "topn": 120,
            },
            "final_top_k": 8,
        },
        "citations": {
            "crossref": True,
            "openalex": True,
            "unpaywall": False,
            "cache_seconds": 604800,  # 7 days
            "page_offset_from_crossref": True,
        },
        "output": {
            "max_snippet_chars": 900,
            "pretty_json": True,
            "include_scores": True,
        },
        "performance": {
            "api_semaphore_size": 5,      # Max concurrent API calls
            "pdf_thread_workers": None,   # None = auto (num_cores), or set manually
            "deterministic": True,        # Enable deterministic tie-breaking
            "numpy_seed": 42,            # Seed for numpy random operations
        },
    }


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from file with fallback to defaults."""
    cfg = get_default_config()
    
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                file_cfg = yaml.safe_load(f) or {}
            # Deep merge file config into defaults
            cfg = merge_configs(cfg, file_cfg)
        except Exception as e:
            print(f"Warning: Failed to load config file {path}: {e}")
    
    return cfg


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def apply_env_vars(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    # Environment variables use underscore-separated naming
    # e.g., RAG_PATHS_PDF_DIR for cfg["paths"]["pdf_dir"]
    
    env_mappings = {
        "RAG_PATHS_PDF_DIR": ["paths", "pdf_dir"],
        "RAG_PATHS_CACHE_DIR": ["paths", "cache_dir"],
        "RAG_PATHS_CROSSREF_EMAIL": ["paths", "crossref_email"],
        "RAG_BM25_K1": ["bm25", "k1"],
        "RAG_BM25_B": ["bm25", "b"],
        "RAG_PRF_ENABLED": ["prf", "enabled"],
        "RAG_PRF_FB_DOCS": ["prf", "fb_docs"],
        "RAG_PRF_FB_TERMS": ["prf", "fb_terms"],
        "RAG_RERANK_FINAL_TOP_K": ["rerank", "final_top_k"],
    }
    
    for env_var, path in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # Type conversion based on the original value
            current = cfg
            for key in path[:-1]:
                current = current.setdefault(key, {})
            
            original_type = type(current.get(path[-1]))
            if original_type == bool:
                value = value.lower() in ("true", "1", "yes", "on")
            elif original_type == int:
                value = int(value)
            elif original_type == float:
                value = float(value)
            
            current[path[-1]] = value
    
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], cli: Namespace) -> Dict[str, Any]:
    """Apply CLI argument overrides to config."""
    # Map CLI args to config paths
    cli_mappings = {
        "pdf_dir": ["paths", "pdf_dir"],
        "k": ["rerank", "final_top_k"],
        "rm3": ["prf", "enabled"],
        "fb_docs": ["prf", "fb_docs"],
        "fb_terms": ["prf", "fb_terms"],
        "alpha": ["prf", "alpha"],
        "no_prox": ["bonuses", "proximity", "enabled"],  # inverted logic
        "prox_window": ["bonuses", "proximity", "window"],
        "prox_lambda": ["bonuses", "proximity", "weight"],
        "ngram_lambda": ["bonuses", "ngram", "weight"],
        "no_diversity": ["diversity", "enabled"],  # inverted logic
        "div_lambda": ["diversity", "per_doc_penalty"],
        "max_per_doc": ["diversity", "max_per_doc"],
        "semantic_rerank": ["rerank", "semantic", "enabled"],
        "semantic_topn": ["rerank", "semantic", "topn"],
    }
    
    for cli_arg, path in cli_mappings.items():
        if hasattr(cli, cli_arg) and getattr(cli, cli_arg) is not None:
            value = getattr(cli, cli_arg)
            
            # Handle inverted boolean logic
            if cli_arg in ("no_prox", "no_diversity"):
                value = not value
            
            # Navigate to the right place in config
            current = cfg
            for key in path[:-1]:
                current = current.setdefault(key, {})
            current[path[-1]] = value
    
    return cfg


def load_full_config(config_path: Optional[str] = None, cli_args: Optional[Namespace] = None) -> Dict[str, Any]:
    """Load configuration with full precedence: defaults → file → env → CLI."""
    config_path = config_path or "config.yaml"
    
    # Step 1: Load from file (includes defaults)
    cfg = load_config(config_path)
    
    # Step 2: Apply environment variables
    cfg = apply_env_vars(cfg)
    
    # Step 3: Apply CLI overrides
    if cli_args:
        cfg = apply_cli_overrides(cfg, cli_args)
    
    return cfg