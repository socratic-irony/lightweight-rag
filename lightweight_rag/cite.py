"""Citation and DOI handling with Crossref integration."""

import re
from typing import Optional, List
from urllib.parse import quote

import httpx

from .models import DocMeta


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