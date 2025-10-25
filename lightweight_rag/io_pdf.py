"""PDF text extraction and document processing."""

import os
import re
import sys
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

from .io_biblio import load_biblio_index
from .models import Chunk, DocMeta, find_doi_in_text

# Import moved inside function to avoid circular dependency

# NOTE: PyMuPDF / MuPDF is known to have thread-safety issues when opening/closing
# multiple documents concurrently in different threads on some platforms / versions.
# To prevent intermittent native crashes (EXC_BAD_ACCESS in libmupdf.dylib
# pdf_minimize_document), we serialize PDF open/close + page extraction with a
# global lock. This preserves stability when the orchestrator requests multiple
# worker threads.
_FITZ_LOCK = threading.Lock()


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
        if (
            ord(char) < 32 and char not in "\t\n\r"
        ):  # Control characters except tab, newline, carriage return
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
    repeated_pattern = re.findall(r"(.)\1{4,}", text)
    non_space_repeats = [p for p in repeated_pattern if p not in " \t\n\r"]
    if len(non_space_repeats) > 3:  # Too many repeated patterns
        return False

    # Check for reasonable character distribution
    # Text should have some common letters
    common_chars = set("etaoinshrdlucmfwypvbgkjqxz ETAOINSHRDLUCMFWYPVBGKJQXZ")
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
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

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
    HARD_HYPH = re.compile(r"(\w)-\n(\w)")  # word-break hyphens
    SOFT_HYPH = "\u00ad"  # soft hyphen

    s = s.replace(SOFT_HYPH, "")  # Remove soft hyphens
    s = HARD_HYPH.sub(r"\1\2", s)  # De-hyphenate line breaks
    s = s.replace("\n", " ")  # Convert newlines to spaces
    s = unicodedata.normalize("NFKC", s)  # Normalize unicode
    s = re.sub(r"\s+", " ", s).strip()  # Clean excessive whitespace
    return s


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using regex patterns, preserving sentence-ending punctuation.

    Args:
        text: Input text to split

    Returns:
        List of sentences with their ending punctuation
    """
    # Simple sentence boundaries - avoid complex lookbehind patterns
    # First replace common abbreviations to protect them
    text = re.sub(
        r"\b(Dr|Mr|Mrs|Ms|Prof|vs|etc|i\.e|e\.g|cf|al)\.",
        lambda m: m.group(0).replace(".", "~DOT~"),
        text,
    )

    # Split on sentence endings, keeping the punctuation with each sentence
    # Use lookahead to split after punctuation + whitespace
    sentences = re.split(r"([.!?]+)\s+", text)

    # Rejoin sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            sentence = sentence.replace("~DOT~", ".").strip()
            if sentence and len(sentence.strip()) > 10:
                result.append(sentence)
    
    # Handle the last sentence if it doesn't have trailing punctuation
    if len(sentences) % 2 == 1:
        last = sentences[-1].replace("~DOT~", ".").strip()
        if last and len(last.strip()) > 10:
            result.append(last)

    return result


def create_sliding_windows(
    text: str, window_chars: int = 300, overlap_chars: int = 50
) -> List[str]:
    """
    Create sliding windows of text with overlap, respecting sentence boundaries.

    Args:
        text: Input text
        window_chars: Size of each window in characters
        overlap_chars: Number of characters to overlap between windows

    Returns:
        List of text windows
    """
    if len(text) <= window_chars:
        return [text]

    # Split text into sentences to respect sentence boundaries
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # If we only got one "sentence" and it's very long, it likely means
    # the text doesn't have proper sentence boundaries. In this case,
    # fall back to word-based chunking to avoid returning a single huge chunk.
    if len(sentences) == 1 and len(sentences[0]) > window_chars:
        # Fall back to word-based chunking for non-sentence text
        words = text.split()
        if not words:
            return []

        windows: List[str] = []
        current_words: List[str] = []
        current_length = 0
        index = 0

        while index < len(words):
            word = words[index]
            word_length = len(word)
            additional_length = word_length if not current_words else word_length + 1

            if current_length + additional_length <= window_chars or not current_words:
                current_words.append(word)
                current_length += additional_length
                index += 1
            else:
                window_text = " ".join(current_words).strip()
                if window_text:
                    windows.append(window_text)

                if overlap_chars > 0 and current_words:
                    overlap_words: List[str] = []
                    overlap_length = 0
                    j = len(current_words) - 1
                    while j >= 0 and overlap_length < overlap_chars:
                        token = current_words[j]
                        token_length = len(token) if not overlap_words else len(token) + 1
                        overlap_length += token_length
                        overlap_words.insert(0, token)
                        j -= 1

                    overlap_total_length = sum(len(tok) for tok in overlap_words) + max(
                        len(overlap_words) - 1, 0
                    )
                    next_word_length = word_length if not overlap_words else word_length + 1

                    if overlap_total_length + next_word_length > window_chars and overlap_words:
                        current_words = []
                        current_length = 0
                    else:
                        current_words = overlap_words
                        current_length = sum(len(tok) for tok in current_words) + max(
                            len(current_words) - 1, 0
                        )
                else:
                    current_words = []
                    current_length = 0

        if current_words:
            window_text = " ".join(current_words).strip()
            if window_text:
                windows.append(window_text)

        return [w for w in windows if len(w.strip()) > 20]

    # Sentence-based chunking (the primary/correct path)
    windows: List[str] = []
    current_sentences: List[str] = []
    current_length = 0
    index = 0

    while index < len(sentences):
        sentence = sentences[index]
        sentence_length = len(sentence)
        # Include a space when appending additional sentences
        additional_length = sentence_length if not current_sentences else sentence_length + 1

        if current_length + additional_length <= window_chars or not current_sentences:
            current_sentences.append(sentence)
            current_length += additional_length
            index += 1
        else:
            window_text = " ".join(current_sentences).strip()
            if window_text:
                windows.append(window_text)

            if overlap_chars > 0 and current_sentences:
                overlap_sentences: List[str] = []
                overlap_length = 0
                j = len(current_sentences) - 1
                while j >= 0 and overlap_length < overlap_chars:
                    sent = current_sentences[j]
                    sent_length = len(sent) if not overlap_sentences else len(sent) + 1
                    overlap_length += sent_length
                    overlap_sentences.insert(0, sent)
                    j -= 1

                # Check if next sentence would fit with overlap sentences
                # If not (e.g., sentence is too long), skip overlap to avoid infinite loop
                overlap_total_length = sum(len(s) for s in overlap_sentences) + max(
                    len(overlap_sentences) - 1, 0
                )
                next_sentence_length = sentence_length if not overlap_sentences else sentence_length + 1

                if overlap_total_length + next_sentence_length > window_chars and overlap_sentences:
                    # Sentence won't fit even after overlap, reset completely
                    current_sentences = []
                    current_length = 0
                else:
                    current_sentences = overlap_sentences
                    current_length = sum(len(s) for s in current_sentences) + max(
                        len(current_sentences) - 1, 0
                    )
            else:
                current_sentences = []
                current_length = 0

    # Append any remaining sentences as the final window
    if current_sentences:
        window_text = " ".join(current_sentences).strip()
        if window_text:
            windows.append(window_text)

    return [w for w in windows if len(w.strip()) > 20]


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
    """Extract text from each page of a PDF with quality validation.

    Thread-safety: guarded by a global lock to avoid MuPDF native crashes when
    multiple documents are processed concurrently in different threads.
    """
    pages: List[Dict[str, Any]] = []
    try:
        with _FITZ_LOCK:
            # Use context manager to ensure deterministic close ordering
            with fitz.open(pdf_path) as doc:
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
                                block_text = " ".join(
                                    [block[4] for block in blocks if isinstance(block[4], str)]
                                )
                                block_text = clean_text(block_text)

                                if is_text_quality_good(block_text):
                                    text = block_text
                                else:
                                    # Log warning about poor text quality
                                    print(
                                        f"Warning: Poor text quality detected in {pdf_path}, page {page_num + 1}"
                                    )
                                    # Still include the page but mark it
                                    text = f"[TEXT_QUALITY_WARNING] {text}"

                        except Exception as e:
                            print(
                                f"Alternative extraction failed for {pdf_path}, page {page_num + 1}: {e}"
                            )

                    pages.append(
                        {
                            "page_number": page_num + 1,  # 1-based
                            "text": text,
                        }
                    )
                    # Help GC release references deterministically per iteration
                    del page
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return pages


def _discover_pdf_files(pdf_dir: Path) -> List[Path]:
    """Discover all PDF files in directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of Path objects for PDF files
    """
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
    return pdf_files


def _enrich_chunk_with_bibliography(chunk: Chunk, biblio_map: dict, doi_map: dict) -> tuple[Chunk, bool]:
    """Enrich a single chunk with bibliography metadata.
    
    Args:
        chunk: Chunk to enrich
        biblio_map: Bibliography index by filename
        doi_map: Bibliography index by DOI
        
    Returns:
        Tuple of (enriched_chunk, was_enriched)
    """
    meta = chunk.meta
    enriched = False
    
    key = os.path.basename(meta.source).lower()
    entry = biblio_map.get(key)
    if not entry and meta.doi:
        entry = doi_map.get((meta.doi or "").lower())
    
    if entry:
        authors_fmt = []
        for a in entry.authors or []:
            fam = (a.get("family") or "").strip()
            giv = (a.get("given") or "").strip()
            if fam and giv:
                authors_fmt.append(f"{fam}, {giv}")
            elif fam:
                authors_fmt.append(fam)
        
        if not meta.title:
            meta.title = entry.title
        if not meta.authors:
            meta.authors = authors_fmt
        if not meta.year and hasattr(entry, "year"):
            meta.year = entry.year
        if not meta.doi and entry.doi:
            meta.doi = entry.doi
        if meta.start_page is None and entry.start_page is not None:
            meta.start_page = entry.start_page
        if meta.end_page is None and entry.end_page is not None:
            meta.end_page = entry.end_page
        if not meta.citekey and entry.citekey:
            meta.citekey = entry.citekey
        enriched = True
    
    return chunk, enriched


def _try_load_cache(
    pdf_files: List[Path],
    pdf_dir: Path,
    citation_config: Optional[dict],
    new_files: List[Path],
    changed_files: List[Path],
    removed_files: List[Path],
    cached_corpus: Optional[List[Chunk]],
    chunking_config_hash: Optional[str],
) -> Optional[List[Chunk]]:
    """Try to load and enrich cached corpus if no files need processing.
    
    Args:
        pdf_files: All PDF files in directory
        pdf_dir: Directory containing PDFs
        citation_config: Configuration for citation enrichment
        new_files: List of new files detected
        changed_files: List of changed files detected
        removed_files: List of removed files detected
        cached_corpus: Previously cached corpus
        chunking_config_hash: Hash of the active chunking configuration

    Returns:
        Enriched corpus if cache is valid, None if reprocessing needed
    """
    from .index import (
        filter_corpus_by_files,
        manifest_for_dir_with_text_hash,
        save_corpus_to_cache,
        save_manifest,
        CACHE_DIR,
        CHUNKING_HASH_KEY,
    )
    
    files_to_process = new_files + changed_files
    
    # If no files need processing, enrich cached corpus with bibliography (and filter) and return
    if not files_to_process and cached_corpus:
        print(f"Loaded {len(cached_corpus)} chunks from cache (no changes detected)")
        
        # Load bibliography maps if configured
        biblio_map = {}
        doi_map = {}
        prefer_biblio = True
        drop_unknown = False
        if citation_config:
            idx_path = citation_config.get("bibliography_index_path")
            if idx_path:
                try:
                    biblio_map = load_biblio_index(idx_path)
                    from .io_biblio import load_biblio_index_by_doi as _load_doi
                    doi_map = _load_doi(idx_path)
                except Exception:
                    biblio_map = {}
                    doi_map = {}
            prefer_biblio = citation_config.get("prefer_bibliography", True)
            drop_unknown = citation_config.get("drop_unknown", False)
        
        # Track per-document enrichment and drops
        matched_docs = set()
        dropped_docs = set()
        
        # Enrich cached corpus
        updated_corpus: List[Chunk] = []
        for chunk in cached_corpus:
            if prefer_biblio and biblio_map:
                chunk, enriched = _enrich_chunk_with_bibliography(chunk, biblio_map, doi_map)
                if enriched:
                    try:
                        matched_docs.add(os.path.basename(str(chunk.meta.source)).lower())
                    except Exception:
                        pass
            
            # Drop if missing required metadata
            if drop_unknown and (not chunk.meta.authors or chunk.meta.year is None):
                try:
                    dropped_docs.add(os.path.basename(str(chunk.meta.source)).lower())
                except Exception:
                    pass
                continue
            
            updated_corpus.append(chunk)
        
        # Filter out removed files if any
        if removed_files:
            keep_files = [f for f in pdf_files if f not in removed_files]
            updated_corpus = filter_corpus_by_files(updated_corpus, keep_files)
            print(
                f"Removed {len(removed_files)} files from cache, {len(updated_corpus)} chunks remaining"
            )
        
        # Persist cache and manifest, write diagnostics
        save_corpus_to_cache(updated_corpus)
        metadata = {CHUNKING_HASH_KEY: chunking_config_hash} if chunking_config_hash else None
        current_manifest = manifest_for_dir_with_text_hash(
            pdf_dir,
            updated_corpus,
            metadata,
        )
        save_manifest(current_manifest)
        
        diagnostics = {
            "total_pdfs": len(pdf_files),
            "processed_pdfs": 0,
            "matched_via_index": len(matched_docs) if biblio_map else 0,
            "matched_via_doi": 0,
            "dropped_unknown": len(dropped_docs) if drop_unknown else 0,
            "new_files": len(new_files),
            "changed_files": len(changed_files),
            "removed_files": len(removed_files),
            "biblio_index_present": bool(biblio_map),
        }
        try:
            import json
            from .index import CACHE_DIR as _C
            diag_path = _C / "warmup_diagnostics.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics, f)
        except Exception as e:
            print(f"Failed to write diagnostics: {e}")
        
        return updated_corpus
    
    return None


async def _enrich_citations_parallel(
    dois_to_fetch: List[str],
    citation_config: Optional[dict],
    cache_seconds: int,
    max_concurrent_api: int,
) -> Dict[str, DocMeta]:
    """Batch fetch and enrich metadata for DOIs in parallel.
    
    Args:
        dois_to_fetch: List of DOIs to fetch metadata for
        citation_config: Configuration for citation sources
        cache_seconds: Cache duration in seconds
        max_concurrent_api: Maximum concurrent API requests
        
    Returns:
        Dictionary mapping DOI to enriched metadata
    """
    import httpx
    from .cite import batch_enriched_lookup
    
    doi_meta_map = {}
    if not dois_to_fetch:
        return doi_meta_map
    
    print(f"Fetching metadata for {len(dois_to_fetch)} DOIs...")

    # Use configuration or defaults
    if citation_config is None:
        citation_config = {
            "crossref": True,
            "openalex": True,
            "unpaywall": False,
            "unpaywall_email": None,
        }

    async with httpx.AsyncClient() as client:
        enriched_results = await batch_enriched_lookup(
            client,
            dois_to_fetch,
            cache_seconds,
            max_concurrent_api,
            use_crossref=citation_config.get("crossref", True),
            use_openalex=citation_config.get("openalex", True),
            use_unpaywall=citation_config.get("unpaywall", False),
            unpaywall_email=citation_config.get("unpaywall_email", "REDACTED"),
        )

        for doi, enriched_meta in zip(dois_to_fetch, enriched_results):
            if enriched_meta:
                doi_meta_map[doi] = enriched_meta
    
    return doi_meta_map


def _get_metadata_from_bibliography(
    pdf_basename: str,
    pages: List[Dict[str, Any]],
    biblio_map: dict,
    citation_config: Optional[dict],
) -> tuple[DocMeta, int]:
    """Get metadata from bibliography index or extract DOI.
    
    Args:
        pdf_basename: PDF filename
        pages: Extracted page data
        biblio_map: Bibliography index
        citation_config: Citation configuration
        
    Returns:
        Tuple of (DocMeta, matched_via_index_count)
    """
    pdf_key = pdf_basename.lower()
    prefer_biblio = citation_config.get("prefer_bibliography", True) if citation_config else True
    
    # Check bibliography by filename
    biblio = biblio_map.get(pdf_key)
    if biblio and prefer_biblio:
        authors_fmt = []
        for a in biblio.authors or []:
            fam = (a.get("family") or "").strip()
            giv = (a.get("given") or "").strip()
            if fam and giv:
                authors_fmt.append(f"{fam}, {giv}")
            elif fam:
                authors_fmt.append(fam)
        return DocMeta(
            title=biblio.title,
            authors=authors_fmt,
            year=biblio.year,
            doi=biblio.doi,
            source=pdf_basename,
            start_page=biblio.start_page,
            end_page=biblio.end_page,
            citekey=biblio.citekey,
        ), 1
    
    # Look for DOI in first 2 pages
    first_pages_text = " ".join([p["text"] for p in pages[:2]])
    doi = find_doi_in_text(first_pages_text)
    
    # Try DOI lookup in bibliography
    if prefer_biblio and (not biblio) and doi and citation_config:
        from .io_biblio import load_biblio_index_by_doi as _load_doi
        doi_map = _load_doi(citation_config.get("bibliography_index_path")) if citation_config else {}
        by_doi = doi_map.get(doi.lower()) if doi else None
        if by_doi:
            authors_fmt = []
            for a in by_doi.authors or []:
                fam = (a.get("family") or "").strip()
                giv = (a.get("given") or "").strip()
                if fam and giv:
                    authors_fmt.append(f"{fam}, {giv}")
                elif fam:
                    authors_fmt.append(fam)
            return DocMeta(
                title=by_doi.title,
                authors=authors_fmt,
                year=by_doi.year,
                doi=doi,
                source=pdf_basename,
                start_page=by_doi.start_page,
                end_page=by_doi.end_page,
                citekey=by_doi.citekey,
            ), 1
    
    return DocMeta(title=None, authors=[], year=None, doi=doi, source=pdf_basename), 0


def _create_chunks_from_pages(
    pages: List[Dict[str, Any]],
    meta: DocMeta,
    pdf_basename: str,
    doc_id: int,
    doc_title: str,
    chunking_config: Optional[dict],
    drop_unknown: bool,
) -> tuple[List[Chunk], bool]:
    """Create chunks from PDF pages.
    
    Args:
        pages: Extracted page data
        meta: Document metadata
        pdf_basename: PDF filename
        doc_id: Document ID
        doc_title: Document title for context
        chunking_config: Chunking configuration
        drop_unknown: Whether to drop docs with missing metadata
        
    Returns:
        Tuple of (chunks_list, was_dropped)
    """
    chunks = []
    
    for page_data in pages:
        text = page_data["text"].strip()

        # Skip empty pages or pages with text quality warnings
        if not text or text.startswith("[TEXT_QUALITY_WARNING]"):
            if text.startswith("[TEXT_QUALITY_WARNING]"):
                print(
                    f"Skipping page {page_data['page_number']} of {pdf_basename} due to poor text quality"
                )
            continue

        # Drop based on missing author/year if configured (once per document)
        if drop_unknown and (not meta.authors or meta.year is None):
            return [], True

        # Final quality check before adding to corpus
        if not is_text_quality_good(text):
            print(
                f"Skipping page {page_data['page_number']} of {pdf_basename} due to failed quality check"
            )
            continue

        # Apply chunking strategy
        text_chunks = chunk_text(text, doc_title, chunking_config)

        # Create Chunk objects for each text chunk
        for chunk_idx, chunk_content in enumerate(text_chunks):
            chunk = Chunk(
                doc_id=doc_id,
                source=pdf_basename,
                page=page_data["page_number"],
                text=chunk_content,
                meta=meta,
            )
            chunks.append(chunk)
    
    return chunks, False


def _extract_and_chunk_pdfs(
    files_to_process_list: List[Path],
    pdf_data_list: List[List[Dict[str, Any]]],
    citation_config: Optional[dict],
    chunking_config: Optional[dict],
    doi_meta_map: Dict[str, DocMeta],
    pdf_dir: Path,
) -> tuple[List[Chunk], int, int, int]:
    """Extract text from PDFs and create chunks with metadata enrichment.
    
    Args:
        files_to_process_list: List of PDF files to process
        pdf_data_list: Extracted page data from PDFs
        citation_config: Configuration for citation enrichment
        chunking_config: Configuration for text chunking
        doi_meta_map: Pre-fetched DOI metadata
        pdf_dir: Directory containing PDFs
        
    Returns:
        Tuple of (corpus_chunks, matched_via_index, matched_via_doi, dropped_unknown)
    """
    from .index import CACHE_DIR
    
    # Load bibliography index if present
    biblio_map = {}
    drop_unknown = False
    if citation_config:
        idx_path = citation_config.get("bibliography_index_path")
        if idx_path:
            try:
                biblio_map = load_biblio_index(idx_path)
            except Exception:
                biblio_map = {}
        drop_unknown = citation_config.get("drop_unknown", False)

    matched_via_index = 0
    matched_via_doi = 0
    dropped_unknown_count = 0
    new_corpus_chunks = []
    doc_id = 0
    total_pdfs = len(files_to_process_list)

    for i, (pdf_file, pdf_data) in enumerate(zip(files_to_process_list, pdf_data_list)):
        if pdf_data is None:
            continue

        pdf_basename = os.path.basename(str(pdf_file))
        pages = pdf_data

        # Get metadata from bibliography or extract DOI
        meta, biblio_matched = _get_metadata_from_bibliography(
            pdf_basename, pages, biblio_map, citation_config
        )
        matched_via_index += biblio_matched

        # Update metadata with DOI lookup results
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

        # Create chunks for each page
        doc_title = meta.title or pdf_basename.replace(".pdf", "")
        chunks, was_dropped = _create_chunks_from_pages(
            pages, meta, pdf_basename, doc_id, doc_title, chunking_config, drop_unknown
        )
        
        if was_dropped:
            dropped_unknown_count += 1
        else:
            new_corpus_chunks.extend(chunks)

        # After finishing this PDF, update warmup diagnostics for UI polling
        try:
            import json as _json
            diag_path = CACHE_DIR / "warmup_diagnostics.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_partial = {
                "total_pdfs": total_pdfs,
                "processed_pdfs": i + 1,
                "matched_via_index": matched_via_index,
                "matched_via_doi": matched_via_doi,
                "dropped_unknown": dropped_unknown_count,
                "biblio_index_present": bool(biblio_map),
            }
            with open(diag_path, "w", encoding="utf-8") as _f:
                _json.dump(diagnostics_partial, _f)
        except Exception:
            pass

        doc_id += 1

    return new_corpus_chunks, matched_via_index, matched_via_doi, dropped_unknown_count


def _save_corpus_cache(
    corpus: List[Chunk],
    pdf_dir: Path,
    pdf_files: List[Path],
    matched_via_index: int,
    matched_via_doi: int,
    dropped_unknown: int,
    new_files: List[Path],
    changed_files: List[Path],
    removed_files: List[Path],
    biblio_map: dict,
    chunking_config_hash: Optional[str],
) -> None:
    """Save corpus to cache with manifest and diagnostics.
    
    Args:
        corpus: Corpus chunks to save
        pdf_dir: Directory containing PDFs
        pdf_files: All PDF files in directory
        matched_via_index: Count of docs matched via bibliography index
        matched_via_doi: Count of docs matched via DOI lookup
        dropped_unknown: Count of docs dropped due to missing metadata
        new_files: List of new files
        changed_files: List of changed files
        removed_files: List of removed files
        biblio_map: Bibliography index map
        chunking_config_hash: Hash of the chunking configuration used for this corpus
    """
    from .index import (
        manifest_for_dir_with_text_hash,
        save_corpus_to_cache,
        save_manifest,
        CACHE_DIR,
        CHUNKING_HASH_KEY,
    )
    
    # Cache the results
    save_corpus_to_cache(corpus)
    metadata = {CHUNKING_HASH_KEY: chunking_config_hash} if chunking_config_hash else None
    enhanced_manifest = manifest_for_dir_with_text_hash(pdf_dir, corpus, metadata)
    save_manifest(enhanced_manifest)

    # Write diagnostics
    diagnostics = {
        "total_pdfs": len(pdf_files),
        "processed_pdfs": len(new_files) + len(changed_files) if new_files or changed_files else len(pdf_files),
        "matched_via_index": matched_via_index,
        "matched_via_doi": matched_via_doi,
        "dropped_unknown": dropped_unknown,
        "new_files": len(new_files) if new_files is not None else None,
        "changed_files": len(changed_files) if changed_files is not None else None,
        "removed_files": len(removed_files) if removed_files is not None else None,
        "biblio_index_present": bool(biblio_map),
    }
    try:
        import json

        diag_path = CACHE_DIR / "warmup_diagnostics.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f)
    except Exception as e:
        print(f"Failed to write diagnostics: {e}")


def _collect_dois_from_pdfs(
    files_to_process_list: List[Path],
    pdf_data_list: List[List[Dict[str, Any]]],
    citation_config: Optional[dict],
) -> List[str]:
    """Collect DOIs from PDF pages for batch lookup.
    
    Args:
        files_to_process_list: List of PDF files to process
        pdf_data_list: Extracted page data from PDFs
        citation_config: Configuration for citation enrichment
        
    Returns:
        List of DOIs found in PDFs
    """
    # Load bibliography index if present
    biblio_map = {}
    prefer_biblio = True
    if citation_config:
        idx_path = citation_config.get("bibliography_index_path")
        if idx_path:
            try:
                biblio_map = load_biblio_index(idx_path)
            except Exception:
                biblio_map = {}
        prefer_biblio = citation_config.get("prefer_bibliography", True)
    
    dois_to_fetch = []
    for pdf_file, pages in zip(files_to_process_list, pdf_data_list):
        if not pages:
            continue
            
        pdf_basename = os.path.basename(str(pdf_file))
        pdf_key = pdf_basename.lower()
        
        # Check if we have bibliography entry with DOI
        biblio = biblio_map.get(pdf_key)
        if biblio and prefer_biblio and biblio.doi:
            dois_to_fetch.append(biblio.doi)
        else:
            # Look for DOI in first 2 pages
            first_pages_text = " ".join([p["text"] for p in pages[:2]])
            doi = find_doi_in_text(first_pages_text)
            if doi:
                dois_to_fetch.append(doi)
    
    return dois_to_fetch


def _log_processing_status(
    files_to_process: List[Path],
    new_files: List[Path],
    changed_files: List[Path],
    removed_files: List[Path],
    pdf_files: List[Path],
) -> None:
    """Log what files are being processed.
    
    Args:
        files_to_process: Files that need processing
        new_files: New files detected
        changed_files: Changed files detected
        removed_files: Removed files detected
        pdf_files: All PDF files in directory
    """
    if files_to_process:
        print(
            f"Processing {len(files_to_process)} changed/new PDFs (out of {len(pdf_files)} total)..."
        )
        if new_files:
            print(f"  - {len(new_files)} new files")
        if changed_files:
            print(f"  - {len(changed_files)} changed files")
        if removed_files:
            print(f"  - {len(removed_files)} removed files")
    else:
        print(f"Processing {len(pdf_files)} PDFs (initial build)...")


def _extract_pdf_text_parallel(
    files_to_process_list: List[Path],
    max_workers: Optional[int],
) -> List[List[Dict[str, Any]]]:
    """Extract text from PDFs in parallel.
    
    Args:
        files_to_process_list: List of PDF files to process
        max_workers: Maximum number of worker threads
        
    Returns:
        List of extracted page data for each PDF
    """
    from .performance import get_optimal_worker_count, process_with_thread_pool
    
    if not files_to_process_list:
        return []
    
    if max_workers is None:
        max_workers = get_optimal_worker_count()

    if max_workers > 1 and len(files_to_process_list) > 1:
        print(f"Using {max_workers} workers for PDF processing")
        return process_with_thread_pool(
            extract_pdf_pages, [str(f) for f in files_to_process_list], max_workers
        )
    else:
        return [
            extract_pdf_pages(str(f))
            for f in tqdm(files_to_process_list, desc="Processing PDFs")
        ]


def _merge_with_cached_corpus(
    cached_corpus: Optional[List[Chunk]],
    files_to_process: List[Path],
    pdf_files: List[Path],
    removed_files: List[Path],
) -> List[Chunk]:
    """Merge new chunks with cached chunks from unchanged files.
    
    Args:
        cached_corpus: Previously cached corpus
        files_to_process: Files that were processed
        pdf_files: All PDF files in directory
        removed_files: Removed files
        
    Returns:
        List of chunks from unchanged files
    """
    from .index import filter_corpus_by_files
    
    if not cached_corpus or not files_to_process:
        return []
    
    # Keep chunks from files that weren't processed (unchanged files)
    unchanged_files = [
        f for f in pdf_files if f not in files_to_process and f not in removed_files
    ]
    cached_chunks_to_keep = filter_corpus_by_files(cached_corpus, unchanged_files)
    print(
        f"Keeping {len(cached_chunks_to_keep)} chunks from {len(unchanged_files)} unchanged files"
    )
    return cached_chunks_to_keep


def _print_final_stats(
    new_corpus_chunks: List[Chunk],
    final_corpus: List[Chunk],
    files_to_process: List[Path],
    pdf_files: List[Path],
    matched_via_index: int,
    matched_via_doi: int,
    dropped_unknown: int,
    biblio_map: dict,
    citation_config: Optional[dict],
) -> None:
    """Print final processing statistics.
    
    Args:
        new_corpus_chunks: Newly created chunks
        final_corpus: Final merged corpus
        files_to_process: Files that were processed
        pdf_files: All PDF files in directory
        matched_via_index: Count of docs matched via bibliography
        matched_via_doi: Count of docs matched via DOI lookup
        dropped_unknown: Count of docs dropped
        biblio_map: Bibliography index map
        citation_config: Citation configuration
    """
    processed_files_count = len(files_to_process) if files_to_process else len(pdf_files)
    print(f"Extracted {len(new_corpus_chunks)} chunks from {processed_files_count} processed PDFs")
    print(f"Total corpus: {len(final_corpus)} chunks from {len(pdf_files)} PDFs")

    if biblio_map:
        print(f"Bibliography index matched: {matched_via_index}/{processed_files_count}")
    if matched_via_doi:
        print(f"DOI lookups matched: {matched_via_doi}")
    if citation_config and citation_config.get("drop_unknown", False):
        print(f"Dropped unknown (author/year): {dropped_unknown}")


async def build_corpus(
    pdf_dir: Path,
    max_workers: Optional[int] = None,
    cache_seconds: int = 604800,
    max_concurrent_api: int = 5,
    citation_config: Optional[dict] = None,
    chunking_config: Optional[dict] = None,
) -> List[Chunk]:
    """Build corpus by extracting text from all PDFs in directory."""
    from .index import (
        CHUNKING_HASH_KEY,
        MANIFEST_META_KEY,
        compute_chunking_config_hash,
        detect_changed_files,
        load_corpus_from_cache,
        load_manifest,
    )

    # Discover PDF files
    pdf_files = _discover_pdf_files(pdf_dir)
    if not pdf_files:
        return []

    # Check cache and detect changes
    cached_corpus = load_corpus_from_cache()
    cached_manifest = load_manifest()

    chunking_config_hash = compute_chunking_config_hash(chunking_config)
    cached_config_hash = None
    if isinstance(cached_manifest, dict):
        meta_section = cached_manifest.get(MANIFEST_META_KEY)
        if isinstance(meta_section, dict):
            cached_config_hash = meta_section.get(CHUNKING_HASH_KEY)

    config_changed = cached_manifest is not None and cached_config_hash != chunking_config_hash

    new_files, changed_files, removed_files = detect_changed_files(pdf_dir, cached_manifest)

    if config_changed:
        if cached_config_hash is None:
            print("Chunking configuration hash missing in cache; forcing rebuild.")
        else:
            print("Chunking configuration changed; forcing corpus rebuild.")
        new_files = []
        changed_files = list(pdf_files)

    # Try to use cached corpus if no processing needed
    cached_result = _try_load_cache(
        pdf_files, pdf_dir, citation_config,
        new_files, changed_files, removed_files, cached_corpus,
        chunking_config_hash,
    )
    if cached_result is not None:
        return cached_result
    
    # Determine which files to process
    files_to_process = new_files + changed_files
    _log_processing_status(files_to_process, new_files, changed_files, removed_files, pdf_files)
    files_to_process_list = list(files_to_process) if files_to_process else pdf_files

    # Extract text from PDFs in parallel
    pdf_data_list = _extract_pdf_text_parallel(files_to_process_list, max_workers)

    # Collect DOIs and fetch metadata in parallel
    dois_to_fetch = _collect_dois_from_pdfs(files_to_process_list, pdf_data_list, citation_config)
    doi_meta_map = await _enrich_citations_parallel(
        dois_to_fetch, citation_config, cache_seconds, max_concurrent_api
    )

    # Extract and chunk PDFs with enriched metadata
    new_corpus_chunks, matched_via_index, matched_via_doi, dropped_unknown = _extract_and_chunk_pdfs(
        files_to_process_list, pdf_data_list, citation_config,
        chunking_config, doi_meta_map, pdf_dir
    )

    # Merge with cached corpus chunks from unchanged files
    cached_chunks = _merge_with_cached_corpus(
        cached_corpus, files_to_process, pdf_files, removed_files
    )
    final_corpus = cached_chunks + new_corpus_chunks

    # Load bibliography map for diagnostics
    biblio_map = {}
    if citation_config:
        idx_path = citation_config.get("bibliography_index_path")
        if idx_path:
            try:
                biblio_map = load_biblio_index(idx_path)
            except Exception:
                pass

    # Save corpus cache with diagnostics
    _save_corpus_cache(
        final_corpus, pdf_dir, pdf_files,
        matched_via_index, matched_via_doi, dropped_unknown,
        new_files, changed_files, removed_files, biblio_map,
        chunking_config_hash,
    )

    # Print final statistics
    _print_final_stats(
        new_corpus_chunks, final_corpus, files_to_process, pdf_files,
        matched_via_index, matched_via_doi, dropped_unknown,
        biblio_map, citation_config
    )

    return final_corpus
