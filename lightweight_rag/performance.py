"""Performance improvements and robustness utilities."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

# Optional numpy import for seeding
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def seed_numpy(seed: Optional[int] = None) -> None:
    """Seed numpy random state for deterministic behavior."""
    if HAS_NUMPY and seed is not None:
        np.random.seed(seed)


def get_optimal_worker_count() -> int:
    """Get optimal number of workers for concurrent processing."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def create_api_semaphore(max_concurrent: int = 5) -> asyncio.Semaphore:
    """Create semaphore for limiting concurrent API calls."""
    return asyncio.Semaphore(max_concurrent)


async def process_with_semaphore(
    semaphore: asyncio.Semaphore, coro_func: Callable, *args, **kwargs
) -> Any:
    """Process async function with semaphore limiting."""
    async with semaphore:
        return await coro_func(*args, **kwargs)


def process_with_thread_pool(
    func: Callable, items: List[Any], max_workers: Optional[int] = None
) -> List[Any]:
    """Process items concurrently using ThreadPoolExecutor."""
    if max_workers is None:
        max_workers = get_optimal_worker_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    return results


def deterministic_sort_key(item: Dict[str, Any]) -> tuple:
    """
    Generate deterministic sort key for tie-breaking.

    When scores are tied, prefer:
    1. Earlier pages (lower page number)
    2. Lexicographic file order
    3. Source position/chunk order
    """
    # Extract relevant fields for sorting
    score = item.get("score", 0.0)

    # Handle both old flat format and new nested format
    if "source" in item and isinstance(item["source"], dict):
        # New nested format: {'source': {'page': 1, 'file': 'a.pdf', 'doi': '...'}}
        source_info = item["source"]
        page = source_info.get("page", 0)
        source_file = source_info.get("file", "")
        doc_id = source_info.get("doi", "")  # Use DOI as doc_id for nested format
    else:
        # Old flat format: {'page': 1, 'source': 'a.pdf', 'doc_id': '1'}
        page = item.get("page", 0)
        source_file = str(item.get("source", ""))
        doc_id = str(item.get("doc_id", ""))

    # Return tuple for sorting (higher score first, then deterministic tie-breaking)
    return (-score, page, source_file, doc_id)


def sort_results_deterministically(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort results with deterministic tie-breaking."""
    return sorted(results, key=deterministic_sort_key)
