"""PDF text extraction and document processing."""

import os
import glob
import sys
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

from .models import DocMeta, Chunk, find_doi_in_text
# Import moved inside function to avoid circular dependency


def _print_if_not_quiet(message: str, config: Dict[str, Any] = None):
    """Print message unless quiet mode is enabled."""
    if config is None or not config.get("_quiet_mode", False):
        print(message, file=sys.stderr if config and config.get("_quiet_mode") else sys.stdout)


def is_text_quality_good(text: str, min_readable_ratio: float = 0.7) -> bool:
    """
    Check if extracted text has good quality (not garbled/encoded).
    
    Args:
        text: Text to check
        min_readable_ratio: Minimum ratio of readable characters required
        
    Returns:
        True if text quality is acceptable, False otherwise
    """
    if not text or len(text.strip()) < 10:
        return False
    
    # Count control characters (excluding common whitespace)
    control_chars = 0
    printable_chars = 0
    
    for char in text:
        if ord(char) < 32 and char not in '\t\n\r':  # Control characters except tab, newline, carriage return
            control_chars += 1
        elif char.isprintable() or char.isspace():
            printable_chars += 1
    
    total_chars = len(text)
    if total_chars == 0:
        return False
    
    # Calculate ratios
    control_ratio = control_chars / total_chars
    printable_ratio = printable_chars / total_chars
    
    # Reject if too many control characters or too few printable characters
    if control_ratio > 0.05:  # More than 5% control characters is suspicious (was 10%)
        return False
    
    if printable_ratio < min_readable_ratio:  # Less than 70% printable characters
        return False
    
    # Check for excessive repeated patterns that might indicate encoding issues
    # Look for sequences of 5+ identical characters (excluding spaces)
    repeated_pattern = re.findall(r'(.)\1{4,}', text)
    non_space_repeats = [p for p in repeated_pattern if p not in ' \t\n\r']
    if len(non_space_repeats) > 3:  # Too many repeated patterns
        return False
    
    # Check for reasonable character distribution
    # Text should have some common letters
    common_chars = set('etaoinshrdlucmfwypvbgkjqxz ETAOINSHRDLUCMFWYPVBGKJQXZ')
    text_chars = set(text.lower())
    if len(text_chars & common_chars) < 5:  # Should have at least 5 common characters
        return False
    
    return True


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text from PDF extraction
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove null bytes and other problematic control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Advanced text normalization for PDF-specific issues
    text = normalize_text(text)
    
    return text


def normalize_text(s: str) -> str:
    """
    Advanced text normalization for PDF extraction issues.
    
    Handles:
    - Soft hyphens (U+00AD)
    - Hard hyphenation across line breaks  
    - Line break normalization
    - Unicode normalization (NFKC)
    - Whitespace normalization
    """
    HARD_HYPH = re.compile(r"(\w)-\n(\w)")   # word-break hyphens
    SOFT_HYPH = "\u00AD"                     # soft hyphen
    
    s = s.replace(SOFT_HYPH, "")             # Remove soft hyphens
    s = HARD_HYPH.sub(r"\1\2", s)           # De-hyphenate line breaks
    s = s.replace("\n", " ")                 # Convert newlines to spaces
    s = unicodedata.normalize("NFKC", s)     # Normalize unicode
    s = re.sub(r"\s+", " ", s).strip()       # Clean excessive whitespace
    return s


# -------------------------
# PDF Text Extraction
# -------------------------

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from each page of a PDF with quality validation."""
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Try different extraction methods if quality is poor
            text = page.get_text()
            
            # Clean and validate text quality
            text = clean_text(text)
            
            if not is_text_quality_good(text):
                # Try alternative extraction method
                try:
                    # Try extracting as textpage with different options
                    textpage = page.get_textpage()
                    alt_text = textpage.extractText()
                    alt_text = clean_text(alt_text)
                    
                    if is_text_quality_good(alt_text):
                        text = alt_text
                    else:
                        # Try blocks extraction
                        blocks = page.get_text("blocks")
                        block_text = " ".join([block[4] for block in blocks if isinstance(block[4], str)])
                        block_text = clean_text(block_text)
                        
                        if is_text_quality_good(block_text):
                            text = block_text
                        else:
                            # Log warning about poor text quality
                            print(f"Warning: Poor text quality detected in {pdf_path}, page {page_num + 1}")
                            # Still include the page but mark it
                            text = f"[TEXT_QUALITY_WARNING] {text}"
                            
                except Exception as e:
                    print(f"Alternative extraction failed for {pdf_path}, page {page_num + 1}: {e}")
            
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
                unpaywall_email=citation_config.get("unpaywall_email", "union-farmers0n@icloud.com")
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
            
            # Skip empty pages or pages with text quality warnings
            if not text:
                continue
                
            # Check if this page has a text quality warning
            if text.startswith("[TEXT_QUALITY_WARNING]"):
                print(f"Skipping page {page_data['page_number']} of {os.path.basename(str(pdf_file))} due to poor text quality")
                continue
            
            # Final quality check before adding to corpus
            if not is_text_quality_good(text):
                print(f"Skipping page {page_data['page_number']} of {os.path.basename(str(pdf_file))} due to failed quality check")
                continue
            
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