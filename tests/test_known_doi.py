#!/usr/bin/env python3
"""Test with known working DOIs to verify our URL format is correct."""

import asyncio
import httpx


async def test_known_working_doi():
    """Test with a known working DOI."""
    # Using a well-known Nature paper DOI
    working_doi = "10.1038/nature12373"  # Famous Nature paper 
    
    async with httpx.AsyncClient() as client:
        print(f"Testing with known working DOI: {working_doi}\n")
        
        # Test Crossref 
        crossref_url = f"https://api.crossref.org/works/{working_doi}"
        print(f"Crossref URL: {crossref_url}")
        try:
            r = await client.get(crossref_url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                title = (data.get("message", {}).get("title") or [""])[0] or "No title"
                print(f"✓ Crossref: SUCCESS - '{title[:50]}...'")
            else:
                print(f"✗ Crossref: FAILED - HTTP {r.status_code}")
        except Exception as e:
            print(f"✗ Crossref: ERROR - {e}")
        
        # Test OpenAlex  
        openalex_url = f"https://api.openalex.org/works/https://doi.org/{working_doi}"
        print(f"OpenAlex URL: {openalex_url}")
        try:
            r = await client.get(openalex_url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                title = data.get("title", "No title")
                print(f"✓ OpenAlex: SUCCESS - '{title[:50]}...'")
            else:
                print(f"✗ OpenAlex: FAILED - HTTP {r.status_code}")
        except Exception as e:
            print(f"✗ OpenAlex: ERROR - {e}")
        
        # Test Unpaywall
        unpaywall_url = f"https://api.unpaywall.org/v2/{working_doi}?email=union-farmers0n@icloud.com"
        print(f"Unpaywall URL: {unpaywall_url}")
        try:
            r = await client.get(unpaywall_url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                is_oa = data.get("is_oa", False)
                print(f"✓ Unpaywall: SUCCESS - Open Access: {is_oa}")
            else:
                print(f"✗ Unpaywall: FAILED - HTTP {r.status_code}")
        except Exception as e:
            print(f"✗ Unpaywall: ERROR - {e}")


if __name__ == "__main__":
    asyncio.run(test_known_working_doi())