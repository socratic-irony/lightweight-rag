"""Citation and DOI handling with Crossref, OpenAlex, and Unpaywall integration."""

import re
import asyncio
from typing import Optional, List, Dict, Any

import httpx

from .models import DocMeta
from .index import (
    load_doi_cache, cache_doi_metadata, is_doi_cache_fresh, 
    get_cached_doi_metadata
)
from .performance import create_api_semaphore, process_with_semaphore


def _print_quiet(message: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Print message unless quiet mode is enabled."""
    if config is None or not config.get("_quiet_mode", False):
        print(message)


async def crossref_meta_for_doi(client: httpx.AsyncClient, doi: str) -> Optional[DocMeta]:
    """Fetch document metadata from Crossref API."""
    url = f"https://api.crossref.org/works/{doi}"
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


async def openalex_meta_for_doi(client: httpx.AsyncClient, doi: str) -> Optional[dict]:
    """Fetch document metadata from OpenAlex API."""
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    try:
        r = await client.get(url, timeout=20)
        r.raise_for_status()
        work = r.json()
        
        # Extract venue information
        venue = None
        if work.get("host_venue") and work["host_venue"].get("display_name"):
            venue = work["host_venue"]["display_name"]
        elif work.get("primary_location") and work["primary_location"].get("source"):
            source_info = work["primary_location"]["source"]
            if source_info and source_info.get("display_name"):
                venue = source_info["display_name"]
        
        # Extract publisher
        publisher = None
        if work.get("host_venue") and work["host_venue"].get("publisher"):
            publisher = work["host_venue"]["publisher"]
        
        # Extract concept tags (limit to high-level concepts)
        concepts = []
        for concept in work.get("concepts", []):
            if concept.get("level", 0) <= 2:  # Only high-level concepts
                concept_name = concept.get("display_name")
                if concept_name and concept.get("score", 0) >= 0.3:  # Only confident matches
                    concepts.append(concept_name)
        
        # Extract open access URL
        oa_url = None
        if work.get("open_access") and work["open_access"].get("oa_url"):
            oa_url = work["open_access"]["oa_url"]
        elif work.get("primary_location") and work["primary_location"].get("pdf_url"):
            oa_url = work["primary_location"]["pdf_url"]
        
        return {
            "venue": venue,
            "publisher": publisher, 
            "concepts": concepts,
            "oa_url": oa_url
        }
    
    except Exception as e:
        print(f"OpenAlex lookup failed for {doi}: {e}")
        return None


async def unpaywall_meta_for_doi(client: httpx.AsyncClient, doi: str, email: str = "union-farmers0n@icloud.com") -> Optional[dict]:
    """Fetch document metadata from Unpaywall API."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        r = await client.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        
        # Extract the best OA location
        oa_url = None
        if data.get("best_oa_location") and data["best_oa_location"].get("url_for_pdf"):
            oa_url = data["best_oa_location"]["url_for_pdf"]
        
        return {
            "oa_url": oa_url,
            "is_oa": data.get("is_oa", False)
        }
    
    except Exception as e:
        print(f"Unpaywall lookup failed for {doi}: {e}")
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


async def enriched_meta_for_doi_cached(
    client: httpx.AsyncClient, 
    doi: str,
    cache_seconds: int = 604800,
    semaphore: Optional[asyncio.Semaphore] = None,
    use_crossref: bool = True,
    use_openalex: bool = True, 
    use_unpaywall: bool = False,
    unpaywall_email: str = "union-farmers0n@icloud.com"
) -> Optional[DocMeta]:
    """
    Fetch enriched document metadata from multiple APIs with caching.
    
    This follows the roadmap's data flow:
    1. Crossref for basic metadata + start_page
    2. OpenAlex for venue, publisher, concepts, OA URL 
    3. Unpaywall for verified OA links (optional)
    """
    # Check cache first
    if is_doi_cache_fresh(doi, cache_seconds):
        cached_data = get_cached_doi_metadata(doi)
        if cached_data:
            return _build_docmeta_from_cache(doi, cached_data)
    
    # If not in cache or stale, fetch from APIs
    if semaphore:
        return await process_with_semaphore(
            semaphore, _fetch_enriched_uncached, client, doi, cache_seconds,
            use_crossref, use_openalex, use_unpaywall, unpaywall_email
        )
    else:
        return await _fetch_enriched_uncached(
            client, doi, cache_seconds, use_crossref, use_openalex, use_unpaywall, unpaywall_email
        )


def _build_docmeta_from_cache(doi: str, cached_data: dict) -> Optional[DocMeta]:
    """Build DocMeta from cached data combining all sources."""
    # Start with Crossref data as base
    crossref_data = cached_data.get("crossref", {})
    openalex_data = cached_data.get("openalex", {})
    unpaywall_data = cached_data.get("unpaywall", {})
    
    return DocMeta(
        title=crossref_data.get("title"),
        authors=crossref_data.get("authors", []),
        year=crossref_data.get("year"),
        doi=doi,
        source="",
        start_page=crossref_data.get("start_page"),
        venue=openalex_data.get("venue"),
        publisher=openalex_data.get("publisher"),
        concepts=openalex_data.get("concepts"),
        oa_url=unpaywall_data.get("oa_url") or openalex_data.get("oa_url")  # Prefer Unpaywall
    )


async def _fetch_enriched_uncached(
    client: httpx.AsyncClient, 
    doi: str, 
    cache_seconds: int,
    use_crossref: bool,
    use_openalex: bool,
    use_unpaywall: bool,
    unpaywall_email: str
) -> Optional[DocMeta]:
    """Internal function to fetch from multiple APIs and cache enriched result."""
    crossref_data = {}
    openalex_data = {}
    unpaywall_data = {}
    
    # Fetch from enabled APIs
    if use_crossref:
        crossref_meta = await crossref_meta_for_doi(client, doi)
        if crossref_meta:
            crossref_data = {
                "title": crossref_meta.title,
                "authors": crossref_meta.authors,
                "year": crossref_meta.year,
                "start_page": crossref_meta.start_page
            }
    
    if use_openalex:
        openalex_result = await openalex_meta_for_doi(client, doi)
        if openalex_result:
            openalex_data = openalex_result
    
    if use_unpaywall:
        unpaywall_result = await unpaywall_meta_for_doi(client, doi, unpaywall_email)
        if unpaywall_result:
            unpaywall_data = unpaywall_result
    
    # Cache all results
    cache_doi_metadata(
        doi, 
        crossref_data=crossref_data if crossref_data else None,
        openalex_data=openalex_data if openalex_data else None, 
        unpaywall_data=unpaywall_data if unpaywall_data else None
    )
    
    # Build combined DocMeta
    if not any([crossref_data, openalex_data, unpaywall_data]):
        return None
        
    return DocMeta(
        title=crossref_data.get("title"),
        authors=crossref_data.get("authors", []),
        year=crossref_data.get("year"),
        doi=doi,
        source="",
        start_page=crossref_data.get("start_page"),
        venue=openalex_data.get("venue"),
        publisher=openalex_data.get("publisher"),
        concepts=openalex_data.get("concepts"),
        oa_url=unpaywall_data.get("oa_url") or openalex_data.get("oa_url")
    )


async def crossref_meta_for_doi_cached(
    client: httpx.AsyncClient, 
    doi: str,
    cache_seconds: int = 604800,
    semaphore: Optional[asyncio.Semaphore] = None
) -> Optional[DocMeta]:
    """
    Fetch document metadata from Crossref API with caching and semaphore limiting.
    This is kept for backward compatibility - use enriched_meta_for_doi_cached for full functionality.
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
    This is kept for backward compatibility - use batch_enriched_lookup for full functionality.
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


async def batch_enriched_lookup(
    client: httpx.AsyncClient,
    dois: List[str],
    cache_seconds: int = 604800,
    max_concurrent: int = 5,
    use_crossref: bool = True,
    use_openalex: bool = True, 
    use_unpaywall: bool = False,
    unpaywall_email: str = "union-farmers0n@icloud.com"
) -> List[Optional[DocMeta]]:
    """
    Batch lookup DOIs from multiple APIs with concurrent processing and caching.
    """
    semaphore = create_api_semaphore(max_concurrent)
    
    tasks = [
        enriched_meta_for_doi_cached(
            client, doi, cache_seconds, semaphore, 
            use_crossref, use_openalex, use_unpaywall, unpaywall_email
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