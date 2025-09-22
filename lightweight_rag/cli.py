"""Command line interface for the lightweight RAG system."""

import argparse
from pathlib import Path
from typing import Dict, Any

from .config import load_full_config


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Quick RAG over PDFs with BM25 (caching, RM3 PRF, proximity+ngram bonuses, diversity, optional semantic rerank, Crossref page-offset citations)."
    )
    
    # Config system
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to config file (default: config.yaml)")
    
    # Core options (can override config)
    parser.add_argument("--pdf_dir", type=str, help="Folder of PDFs")
    parser.add_argument("--k", type=int, help="Top-k chunks to return")
    parser.add_argument("--query", type=str, default=None, 
                       help="Query; if omitted, will prompt")

    # RM3
    parser.add_argument("--rm3", action="store_true", 
                       help="Enable RM3 pseudo-relevance feedback query expansion")
    parser.add_argument("--fb_docs", type=int, help="Feedback documents for RM3")
    parser.add_argument("--fb_terms", type=int, help="Feedback terms for RM3")
    parser.add_argument("--alpha", type=float, help="Mixing weight for original query in RM3")

    # Proximity & n-grams
    parser.add_argument("--no_prox", action="store_true", 
                       help="Disable proximity bonus (enabled by default)")
    parser.add_argument("--prox_window", type=int, help="Token window size for proximity bonus")
    parser.add_argument("--prox_lambda", type=float, help="Scaling weight for proximity bonus")
    parser.add_argument("--ngram_lambda", type=float, help="Weight for n-gram (bi/tri) bonus")

    # Diversity controls
    parser.add_argument("--no_diversity", action="store_true", 
                       help="Disable diversity bonus (allow many hits from same document)")
    parser.add_argument("--div_lambda", type=float, 
                       help="Diversity penalty per additional hit from the same document")
    parser.add_argument("--max_per_doc", type=int, help="Maximum results from the same document")

    # Semantic rerank
    parser.add_argument("--semantic_rerank", action="store_true", 
                       help="Enable CPU-only sentence-transformers rerank of candidates")
    parser.add_argument("--semantic_topn", type=int, 
                       help="Number of top candidates to rerank semantically")

    return parser


def setup_directories(cfg: Dict[str, Any]) -> None:
    """Ensure required directories exist."""
    pdf_dir = Path(cfg["paths"]["pdf_dir"])
    pdf_dir.mkdir(exist_ok=True)
    
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(exist_ok=True)


def get_query_input(args) -> str:
    """Get query from args or user input."""
    return args.query or input("Enter your query: ").strip()


def parse_args_and_load_config() -> tuple[Dict[str, Any], str]:
    """Parse command line arguments and load configuration."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration with full precedence
    cfg = load_full_config(args.config, args)
    
    # Setup directories
    setup_directories(cfg)
    
    # Get query
    query = get_query_input(args)
    
    return cfg, query