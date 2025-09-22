"""Lightweight RAG package for PDF search and retrieval."""

__version__ = "0.1.0"
__author__ = "Lightweight RAG Contributors"

# Core functionality exports
from .main import query_pdfs, run_rag_pipeline
from .config import get_default_config, load_config, merge_configs

__all__ = [
    "query_pdfs", 
    "run_rag_pipeline",
    "get_default_config",
    "load_config", 
    "merge_configs"
]