"""PDF text extraction and document processing."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

from .models import DocMeta, Chunk, find_doi_in_text
# Import moved inside function to avoid circular dependency


# -------------------------
# PDF Text Extraction
# -------------------------

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from each page of a PDF."""
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages.append({
                "page_number": page_num + 1,  # 1-based
                "text": text
            })
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return pages


async def build_corpus(pdf_dir: Path, max_workers: Optional[int] = None, 
                     cache_seconds: int = 604800, 
                     max_concurrent_api: int = 5,
                     citation_config: Optional[dict] = None) -> List[Chunk]:
    """Build corpus by extracting text from all PDFs in directory."""
    import httpx
    from .performance import process_with_thread_pool, get_optimal_worker_count
    from .index import (
        load_manifest, save_manifest, load_corpus_from_cache, 
        save_corpus_to_cache, manifest_for_dir_with_text_hash, manifest_for_dir
    )
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return []

    # Check if we can load from cache
    cached_corpus = load_corpus_from_cache()
    cached_manifest = load_manifest()
    current_manifest = manifest_for_dir(pdf_dir)
    
    if cached_corpus and cached_manifest == current_manifest:
        print(f"Loaded {len(cached_corpus)} chunks from cache")
        return cached_corpus

    print(f"Processing {len(pdf_files)} PDFs...")
    
    # Extract text from PDFs (potentially in parallel)
    if max_workers is None:
        max_workers = get_optimal_worker_count()
    
    if max_workers > 1 and len(pdf_files) > 1:
        print(f"Using {max_workers} workers for PDF processing")
        pdf_data_list = process_with_thread_pool(
            extract_pdf_pages, [str(f) for f in pdf_files], max_workers
        )
    else:
        pdf_data_list = [extract_pdf_pages(str(f)) for f in tqdm(pdf_files, desc="Processing PDFs")]
    
    # Collect all DOIs for batch processing
    dois_to_fetch = []
    pdf_metadata = []
    
    for i, (pdf_file, pages) in enumerate(zip(pdf_files, pdf_data_list)):
        if not pages:
            pdf_metadata.append(None)
            continue
            
        # Look for DOI in first 2 pages
        first_pages_text = " ".join([p["text"] for p in pages[:2]])
        doi = find_doi_in_text(first_pages_text)
        
        meta = DocMeta(
            title=None,
            authors=[],
            year=None,
            doi=doi,
            source=os.path.basename(str(pdf_file))
        )
        
        pdf_metadata.append((meta, pages))
        if doi:
            dois_to_fetch.append(doi)
    
    # Batch fetch metadata for all DOIs
    doi_meta_map = {}
    if dois_to_fetch:
        print(f"Fetching metadata for {len(dois_to_fetch)} DOIs...")
        
        # Use configuration or defaults
        if citation_config is None:
            citation_config = {
                "crossref": True,
                "openalex": True, 
                "unpaywall": False,
                "unpaywall_email": None
            }
        
        from .cite import batch_enriched_lookup
        
        async with httpx.AsyncClient() as client:
            enriched_results = await batch_enriched_lookup(
                client, dois_to_fetch, cache_seconds, max_concurrent_api,
                use_crossref=citation_config.get("crossref", True),
                use_openalex=citation_config.get("openalex", True),
                use_unpaywall=citation_config.get("unpaywall", False),
                unpaywall_email=citation_config.get("unpaywall_email", "anonymous@example.com")
            )
            
            for doi, enriched_meta in zip(dois_to_fetch, enriched_results):
                if enriched_meta:
                    doi_meta_map[doi] = enriched_meta

    # Build corpus with fetched metadata
    corpus = []
    doc_id = 0
    
    for pdf_file, pdf_data in zip(pdf_files, pdf_metadata):
        if pdf_data is None:
            continue
            
        meta, pages = pdf_data
        
        # Update metadata with fetched info
        if meta.doi and meta.doi in doi_meta_map:
            enriched_meta = doi_meta_map[meta.doi]
            meta.title = enriched_meta.title
            meta.authors = enriched_meta.authors
            meta.year = enriched_meta.year
            meta.start_page = enriched_meta.start_page
            meta.venue = enriched_meta.venue
            meta.publisher = enriched_meta.publisher
            meta.concepts = enriched_meta.concepts
            meta.oa_url = enriched_meta.oa_url

        # Create chunks for each page
        for page_data in pages:
            text = page_data["text"].strip()
            if text:  # Skip empty pages
                chunk = Chunk(
                    doc_id=doc_id,
                    source=os.path.basename(str(pdf_file)),
                    page=page_data["page_number"],
                    text=text,
                    meta=meta
                )
                corpus.append(chunk)

        doc_id += 1
    
    # Cache the results
    save_corpus_to_cache(corpus)
    enhanced_manifest = manifest_for_dir_with_text_hash(pdf_dir, corpus)
    save_manifest(enhanced_manifest)

    print(f"Extracted {len(corpus)} chunks from {len(pdf_files)} PDFs")
    return corpus