"""Citation and DOI handling with Crossref integration."""

import re
import asyncio
from typing import Optional, List
from urllib.parse import quote

import httpx

from .models import DocMeta
from .index import (
    load_doi_cache, cache_doi_metadata, is_doi_cache_fresh, 
    get_cached_doi_metadata
)
from .performance import create_api_semaphore, process_with_semaphore


async def crossref_meta_for_doi(client: httpx.AsyncClient, doi: str) -> Optional[DocMeta]:
    """Fetch document metadata from Crossref API."""
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    try:
        r = await client.get(url, timeout=20)
        r.raise_for_status()
        item = r.json().get("message", {})
        
        title = (item.get("title") or [""])[0] or None
        
        year = None
        for k in ("published-print", "published-online", "issued"):
            if item.get(k, {}).get("date-parts"):
                year = item[k]["date-parts"][0][0]
                break
        
        # Parse page range like "300-314" to derive start_page offset
        start_page: Optional[int] = None
        page_range = item.get("page")
        if isinstance(page_range, str):
            m = re.match(r"\s*(\d+)", page_range)
            if m:
                try:
                    start_page = int(m.group(1))
                except ValueError:
                    start_page = None
        
        authors: List[str] = []
        for a in item.get("author", []) or []:
            fam = (a.get("family") or "").strip()
            giv = (a.get("given") or "").strip()
            if fam and giv:
                authors.append(f"{fam}, {giv}")
            elif fam:
                authors.append(fam)
        
        return DocMeta(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            source="",  # Will be filled by caller
            start_page=start_page
        )
    
    except Exception as e:
        print(f"Crossref lookup failed for {doi}: {e}")
        return None


def author_date_citation(meta: DocMeta, page: Optional[int]) -> str:
    """Format an author-date citation string."""
    if not meta.authors:
        au = "Unknown"
    elif len(meta.authors) == 1:
        # Extract surname only
        au = meta.authors[0].split(",")[0].strip()
    else:
        first_surname = meta.authors[0].split(",")[0].strip()
        au = f"{first_surname} et al."
    
    yr = f"{meta.year}" if meta.year else "n.d."
    
    # Calculate actual page in original document
    actual_page = page
    if meta.start_page is not None and page is not None:
        actual_page = meta.start_page + (page - 1)
    
    if actual_page:
        return f"({au}, {yr}, p. {actual_page})"
    else:
        return f"({au}, {yr})"


async def crossref_meta_for_doi_cached(
    client: httpx.AsyncClient, 
    doi: str,
    cache_seconds: int = 604800,
    semaphore: Optional[asyncio.Semaphore] = None
) -> Optional[DocMeta]:
    """
    Fetch document metadata from Crossref API with caching and semaphore limiting.
    """
    # Check cache first
    if is_doi_cache_fresh(doi, cache_seconds):
        cached_data = get_cached_doi_metadata(doi)
        if cached_data and "crossref" in cached_data:
            crossref_data = cached_data["crossref"]
            return DocMeta(
                title=crossref_data.get("title"),
                authors=crossref_data.get("authors", []),
                year=crossref_data.get("year"),
                doi=doi,
                source="",
                start_page=crossref_data.get("start_page")
            )
    
    # If not in cache or stale, fetch from API
    if semaphore:
        return await process_with_semaphore(
            semaphore, _fetch_crossref_uncached, client, doi, cache_seconds
        )
    else:
        return await _fetch_crossref_uncached(client, doi, cache_seconds)


async def _fetch_crossref_uncached(
    client: httpx.AsyncClient, 
    doi: str, 
    cache_seconds: int
) -> Optional[DocMeta]:
    """Internal function to fetch from Crossref API and cache result."""
    meta = await crossref_meta_for_doi(client, doi)
    
    # Cache the result
    if meta:
        crossref_data = {
            "title": meta.title,
            "authors": meta.authors,
            "year": meta.year,
            "start_page": meta.start_page
        }
        cache_doi_metadata(doi, crossref_data=crossref_data)
    
    return meta


async def batch_crossref_lookup(
    client: httpx.AsyncClient,
    dois: List[str],
    cache_seconds: int = 604800,
    max_concurrent: int = 5
) -> List[Optional[DocMeta]]:
    """
    Batch lookup DOIs from Crossref with concurrent processing and caching.
    """
    semaphore = create_api_semaphore(max_concurrent)
    
    tasks = [
        crossref_meta_for_doi_cached(
            client, doi, cache_seconds, semaphore
        ) for doi in dois
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to None
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"DOI lookup failed: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)
    
    return processed_results