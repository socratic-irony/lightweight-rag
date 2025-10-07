#!/usr/bin/env python3
"""
Lightweight RAG - Modular version
Command-line interface for PDF search and retrieval using BM25 + optional semantic reranking.
"""

import os
import platform
import asyncio
import json
import time
from dotenv import load_dotenv

from lightweight_rag.cli import parse_args_and_load_config
from lightweight_rag.main import run_rag_pipeline

# On macOS, setting this environment variable can help avoid a multiprocessing-related error.
if platform.system() == "Darwin":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


async def main():
    """Main entry point for the CLI application."""
    load_dotenv()
    
    # Parse arguments and load configuration
    cfg, query = parse_args_and_load_config()
    
    # Run the RAG pipeline
    start_time = time.perf_counter()
    results = await run_rag_pipeline(cfg, query)
    total_time = time.perf_counter() - start_time
    
    # Display results
    print("\n=== Top Results ===")
    if cfg["output"]["pretty_json"]:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(results, ensure_ascii=False))

    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    print("Starting RAG pipeline...")
    asyncio.run(main())
