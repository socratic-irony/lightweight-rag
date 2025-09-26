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
from .io_biblio import load_biblio_index
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


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using regex patterns.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    # Simple sentence boundaries - avoid complex lookbehind patterns
    # First replace common abbreviations to protect them
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|vs|etc|i\.e|e\.g|cf|al)\.',
                  lambda m: m.group(0).replace('.', '~DOT~'), text)
    
    # Split on sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Restore the dots in abbreviations
    sentences = [s.replace('~DOT~', '.').strip() for s in sentences]
    
    # Filter out empty sentences and very short ones
    sentences = [s for s in sentences if s and len(s.strip()) > 10]
    
    return sentences


def create_sliding_windows(text: str, window_chars: int = 300, overlap_chars: int = 50) -> List[str]:
    """
    Create sliding windows of text with overlap.
    
    Args:
        text: Input text
        window_chars: Size of each window in characters
        overlap_chars: Number of characters to overlap between windows
        
    Returns:
        List of text windows
    """
    if len(text) <= window_chars:
        return [text]
    
    windows = []
    start = 0
    
    while start < len(text):
        end = start + window_chars
        
        # If this would be the last window and it's very small, merge with previous
        if end >= len(text):
            window = text[start:]
        else:
            window = text[start:end]
            
            # Try to end on sentence boundary if possible
            last_sentence_end = max(
                window.rfind('.'),
                window.rfind('!'),
                window.rfind('?')
            )
            
            if last_sentence_end > window_chars * 0.7:  # At least 70% of window used
                window = text[start:start + last_sentence_end + 1]
                end = start + last_sentence_end + 1
        
        windows.append(window.strip())
        
        # Stop if we've reached the end
        if end >= len(text):
            break
            
        # Move start forward by (window_size - overlap)
        start = end - overlap_chars
        
        # Ensure we don't go backwards
        if start <= len(windows[-1]) - window_chars + overlap_chars:
            break
    
    return [w for w in windows if len(w.strip()) > 20]  # Filter very short windows


def chunk_text(text: str, doc_title: str = "", chunking_config: dict = None) -> List[str]:
    """
    Chunk text according to configuration.
    
    Args:
        text: Input text to chunk
        doc_title: Document title for context enrichment
        chunking_config: Configuration for chunking strategy
        
    Returns:
        List of text chunks
    """
    if not chunking_config:
        chunking_config = {"page_split": "page"}
    
    page_split = chunking_config.get("page_split", "page")
    
    if page_split == "sentence":
        sentences = split_into_sentences(text)
        # Add document context to each sentence if available
        if doc_title:
            sentences = [f"{doc_title} | {sent}" for sent in sentences]
        return sentences
    
    elif page_split == "sliding":
        window_chars = chunking_config.get("window_chars", 300)
        overlap_chars = chunking_config.get("overlap_chars", 50)
        windows = create_sliding_windows(text, window_chars, overlap_chars)
        # Add document context to each window if available
        if doc_title:
            windows = [f"{doc_title} | {window}" for window in windows]
        return windows
    
    else:  # page_split == "page" or default
        # Single chunk with optional context
        if doc_title:
            return [f"{doc_title} | {text}"]
        return [text]


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
                     citation_config: Optional[dict] = None,
                     chunking_config: Optional[dict] = None) -> List[Chunk]:
    """Build corpus by extracting text from all PDFs in directory."""
    import httpx
    from .performance import process_with_thread_pool, get_optimal_worker_count
    from .index import (
        load_manifest, save_manifest, load_corpus_from_cache, 
        save_corpus_to_cache, manifest_for_dir_with_text_hash, manifest_for_dir,
        detect_changed_files, filter_corpus_by_files, CACHE_DIR
    )
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return []

    # Check cache and detect changes for incremental processing
    cached_corpus = load_corpus_from_cache()
    cached_manifest = load_manifest()
    
    # Detect which files have changed
    new_files, changed_files, removed_files = detect_changed_files(pdf_dir, cached_manifest)
    files_to_process = new_files + changed_files
    
    # If no files need processing, return cached corpus (filtered for removed files)
    if not files_to_process and cached_corpus:
        if removed_files:
            # Filter out chunks from removed files
            keep_files = [f for f in pdf_files if f not in removed_files]
            filtered_corpus = filter_corpus_by_files(cached_corpus, keep_files)
            print(f"Removed {len(removed_files)} files from cache, {len(filtered_corpus)} chunks remaining")
            # Update cache with filtered corpus
            save_corpus_to_cache(filtered_corpus)
            current_manifest = manifest_for_dir_with_text_hash(pdf_dir, filtered_corpus)
            save_manifest(current_manifest)
            return filtered_corpus
        else:
            print(f"Loaded {len(cached_corpus)} chunks from cache (no changes detected)")
            return cached_corpus
    
    # Log what we're processing
    if files_to_process:
        print(f"Processing {len(files_to_process)} changed/new PDFs (out of {len(pdf_files)} total)...")
        if new_files:
            print(f"  - {len(new_files)} new files")
        if changed_files:
            print(f"  - {len(changed_files)} changed files")
        if removed_files:
            print(f"  - {len(removed_files)} removed files")
    else:
        print(f"Processing {len(pdf_files)} PDFs (initial build)...")
    
    # Only process files that need processing, not all files
    files_to_process_list = list(files_to_process) if files_to_process else pdf_files
    
    # Extract text from PDFs that need processing (potentially in parallel)
    if max_workers is None:
        max_workers = get_optimal_worker_count()
    
    if files_to_process_list:
        if max_workers > 1 and len(files_to_process_list) > 1:
            print(f"Using {max_workers} workers for PDF processing")
            pdf_data_list = process_with_thread_pool(
                extract_pdf_pages, [str(f) for f in files_to_process_list], max_workers
            )
        else:
            pdf_data_list = [extract_pdf_pages(str(f)) for f in tqdm(files_to_process_list, desc="Processing PDFs")]
    else:
        pdf_data_list = []
    
    # Load bibliography index if present
    biblio_map = {}
    prefer_biblio = True
    drop_unknown = False
    include_pandoc_cite = False
    if citation_config:
        idx_path = citation_config.get("bibliography_index_path")
        if idx_path:
            try:
                biblio_map = load_biblio_index(idx_path)
            except Exception:
                biblio_map = {}
        prefer_biblio = citation_config.get("prefer_bibliography", True)
        drop_unknown = citation_config.get("drop_unknown", False)
        include_pandoc_cite = citation_config.get("include_pandoc_cite", False)

    # Diagnostics counters
    total_pdfs = len(files_to_process_list) if files_to_process_list else len(pdf_files)
    matched_via_index = 0
    matched_via_doi = 0
    dropped_unknown = 0

    # Collect all DOIs for batch processing
    dois_to_fetch = []
    pdf_metadata = []
    
    for i, (pdf_file, pages) in enumerate(zip(files_to_process_list, pdf_data_list)):
        if not pages:
            pdf_metadata.append(None)
            continue
            
        pdf_basename = os.path.basename(str(pdf_file))
        pdf_key = pdf_basename.lower()

        # If we have a bibliography entry for this PDF, prefer it
        biblio = biblio_map.get(pdf_key)
        if biblio and prefer_biblio:
            authors_fmt = []
            for a in (biblio.authors or []):
                fam = (a.get("family") or "").strip()
                giv = (a.get("given") or "").strip()
                if fam and giv:
                    authors_fmt.append(f"{fam}, {giv}")
                elif fam:
                    authors_fmt.append(fam)
            meta = DocMeta(
                title=biblio.title,
                authors=authors_fmt,
                year=biblio.year,
                doi=biblio.doi,
                source=pdf_basename,
                start_page=biblio.start_page,
                end_page=biblio.end_page,
                citekey=biblio.citekey
            )
            matched_via_index += 1
            doi = biblio.doi
        else:
            # Look for DOI in first 2 pages
            first_pages_text = " ".join([p["text"] for p in pages[:2]])
            doi = find_doi_in_text(first_pages_text)
            meta = DocMeta(
                title=None,
                authors=[],
                year=None,
                doi=doi,
                source=pdf_basename
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
                unpaywall_email=citation_config.get("unpaywall_email", "REDACTED")
            )
            
            for doi, enriched_meta in zip(dois_to_fetch, enriched_results):
                if enriched_meta:
                    doi_meta_map[doi] = enriched_meta

    # Build corpus with fetched metadata from processed files only
    new_corpus_chunks = []
    doc_id = 0
    
    for pdf_file, pdf_data in zip(files_to_process_list, pdf_metadata):
        if pdf_data is None:
            continue
            
        meta, pages = pdf_data
        
        # Update metadata with fetched info (only if missing and doi lookup available)
        if meta.doi and meta.doi in doi_meta_map:
            enriched_meta = doi_meta_map[meta.doi]
            # Only overwrite empty fields so biblio data remains authoritative
            if not meta.title:
                meta.title = enriched_meta.title
            if not meta.authors:
                meta.authors = enriched_meta.authors
            if not meta.year:
                meta.year = enriched_meta.year
            if meta.start_page is None:
                meta.start_page = enriched_meta.start_page
            # Non-citation extras
            meta.venue = meta.venue or enriched_meta.venue
            meta.publisher = meta.publisher or enriched_meta.publisher
            meta.concepts = meta.concepts or enriched_meta.concepts
            meta.oa_url = meta.oa_url or enriched_meta.oa_url
            matched_via_doi += 1

        # Create chunks for each page using configured chunking strategy
        doc_title = meta.title or os.path.basename(str(pdf_file)).replace('.pdf', '')
        
        for page_data in pages:
            text = page_data["text"].strip()
            
            # Skip empty pages or pages with text quality warnings
            if not text:
                continue
                
            # Check if this page has a text quality warning
            if text.startswith("[TEXT_QUALITY_WARNING]"):
                print(f"Skipping page {page_data['page_number']} of {os.path.basename(str(pdf_file))} due to poor text quality")
                continue
            
            # Drop based on missing author/year if configured (once per document)
            if drop_unknown and (not meta.authors or meta.year is None):
                # Mark entire document as dropped and stop processing its pages
                dropped_unknown += 1
                # Skip to next document (break inner page loop)
                break

            # Final quality check before adding to corpus
            if not is_text_quality_good(text):
                print(f"Skipping page {page_data['page_number']} of {os.path.basename(str(pdf_file))} due to failed quality check")
                continue
            
            # Apply chunking strategy
            text_chunks = chunk_text(text, doc_title, chunking_config)
            
            # Create Chunk objects for each text chunk
            for chunk_idx, chunk_content in enumerate(text_chunks):
                chunk = Chunk(
                    doc_id=doc_id,
                    source=os.path.basename(str(pdf_file)),
                    page=page_data["page_number"],
                    text=chunk_content,
                    meta=meta
                )
                new_corpus_chunks.append(chunk)

        doc_id += 1
    
    # Merge with cached corpus chunks from unchanged files
    final_corpus = []
    
    if cached_corpus and files_to_process:
        # Keep chunks from files that weren't processed (unchanged files)
        unchanged_files = [f for f in pdf_files if f not in files_to_process and f not in removed_files]
        cached_chunks_to_keep = filter_corpus_by_files(cached_corpus, unchanged_files)
        final_corpus.extend(cached_chunks_to_keep)
        print(f"Keeping {len(cached_chunks_to_keep)} chunks from {len(unchanged_files)} unchanged files")
    
    # Add newly processed chunks
    final_corpus.extend(new_corpus_chunks)
    
    # Cache the results
    save_corpus_to_cache(final_corpus)
    enhanced_manifest = manifest_for_dir_with_text_hash(pdf_dir, final_corpus)
    save_manifest(enhanced_manifest)

    processed_files_count = len(files_to_process_list) if files_to_process else len(pdf_files)
    print(f"Extracted {len(new_corpus_chunks)} chunks from {processed_files_count} processed PDFs")
    print(f"Total corpus: {len(final_corpus)} chunks from {len(pdf_files)} PDFs")
    # Diagnostics summary
    diagnostics = {
        "total_pdfs": total_pdfs,
        "processed_pdfs": processed_files_count,
        "matched_via_index": matched_via_index,
        "matched_via_doi": matched_via_doi,
        "dropped_unknown": dropped_unknown,
        "new_files": len(new_files) if 'new_files' in locals() and new_files is not None else None,
        "changed_files": len(changed_files) if 'changed_files' in locals() and changed_files is not None else None,
        "removed_files": len(removed_files) if 'removed_files' in locals() and removed_files is not None else None,
        "biblio_index_present": bool(biblio_map)
    }
    try:
        import json
        diag_path = CACHE_DIR / "warmup_diagnostics.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f)
    except Exception as e:
        print(f"Failed to write diagnostics: {e}")

    if biblio_map:
        print(f"Bibliography index matched: {matched_via_index}/{total_pdfs}")
    if matched_via_doi:
        print(f"DOI lookups matched: {matched_via_doi}")
    if drop_unknown:
        print(f"Dropped unknown (author/year): {dropped_unknown}")
    return final_corpus
