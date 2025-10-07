"""Tokenization, BM25 indexing, and caching functionality."""

import glob
import gzip
import hashlib
import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from .models import STOP, Chunk

# -------------------------
# Global cache paths (updated by main)
# -------------------------
CACHE_DIR = Path(".raq_cache")
# Don't create the directory immediately - let update_cache_paths handle it
CORPUS_CACHE = CACHE_DIR / "corpus.jsonl.gz"
BM25_CACHE = CACHE_DIR / "bm25.pkl.gz"
MANIFEST_CACHE = CACHE_DIR / "manifest.json"
TOKENIZED_CACHE = CACHE_DIR / "tokenized.pkl.gz"
DOI_CACHE = CACHE_DIR / "dois.json"

MANIFEST_META_KEY = "__meta__"
CHUNKING_HASH_KEY = "chunking_config_hash"


# -------------------------
# Tokenization
# -------------------------


def tokenize(s: str, pattern: str = r"[A-Za-z0-9]+") -> List[str]:
    """Tokenize text using regex pattern."""
    return [t.lower() for t in re.findall(pattern, s) if t.lower() not in STOP]


# -------------------------
# BM25 Building
# -------------------------


def build_bm25(
    corpus: List[Chunk], token_pattern: str = r"[A-Za-z0-9]+", k1: float = 1.5, b: float = 0.75
) -> Tuple[BM25Okapi, List[List[str]]]:
    """Build BM25 index from corpus, with caching."""
    cached_result = load_bm25_from_cache()

    # Validate cached BM25 matches current corpus size
    if cached_result is not None:
        bm25, tokenized = cached_result
        if len(tokenized) == len(corpus):
            return cached_result
        else:
            print(
                f"BM25 cache mismatch: cached {len(tokenized)} chunks, corpus has {len(corpus)} chunks. Rebuilding..."
            )

    print("Building BM25 index...")
    tokenized = [tokenize(chunk.text, token_pattern) for chunk in corpus]
    bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    save_bm25_to_cache(bm25, tokenized)
    print(f"BM25 index built and cached for {len(corpus)} chunks (k1={k1}, b={b})")
    return bm25, tokenized


# -------------------------
# Caching functions
# -------------------------


def update_cache_paths(cache_dir: Path) -> None:
    """Update global cache paths based on config."""
    global CACHE_DIR, CORPUS_CACHE, BM25_CACHE, MANIFEST_CACHE, TOKENIZED_CACHE, DOI_CACHE
    CACHE_DIR = cache_dir
    CACHE_DIR.mkdir(exist_ok=True)
    CORPUS_CACHE = CACHE_DIR / "corpus.jsonl.gz"
    BM25_CACHE = CACHE_DIR / "bm25.pkl.gz"
    MANIFEST_CACHE = CACHE_DIR / "manifest.json"
    TOKENIZED_CACHE = CACHE_DIR / "tokenized.pkl.gz"
    DOI_CACHE = CACHE_DIR / "dois.json"


def manifest_for_dir(pdf_dir: Path) -> Dict[str, Any]:
    """Generate manifest of PDF directory for cache invalidation."""
    files = sorted(glob.glob(str(pdf_dir / "*.pdf")))
    return {
        "files": [
            {"path": f, "mtime": os.path.getmtime(f), "size": os.path.getsize(f)} for f in files
        ]
    }


def load_manifest() -> Optional[Dict[str, Any]]:
    """Load cached manifest."""
    if not MANIFEST_CACHE.exists():
        return None
    try:
        with open(MANIFEST_CACHE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Save manifest to cache."""
    # Ensure parent directory exists
    MANIFEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CACHE, "w") as f:
        json.dump(manifest, f)


def load_corpus_from_cache() -> Optional[List[Chunk]]:
    """Load corpus from cache."""
    if not CORPUS_CACHE.exists():
        return None
    try:
        corpus = []
        with gzip.open(CORPUS_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Reconstruct Chunk and DocMeta objects
                from .models import DocMeta

                meta = DocMeta(**data["meta"])
                chunk = Chunk(
                    doc_id=data["doc_id"],
                    source=data["source"],
                    page=data["page"],
                    text=data["text"],
                    meta=meta,
                )
                corpus.append(chunk)
        return corpus
    except Exception:
        return None


def save_corpus_to_cache(corpus: List[Chunk]) -> None:
    """Save corpus to cache."""
    try:
        # Ensure parent directory exists
        CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CORPUS_CACHE, "wt", encoding="utf-8") as f:
            for chunk in corpus:
                data = {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "text": chunk.text,
                    "meta": {
                        "title": chunk.meta.title,
                        "authors": chunk.meta.authors,
                        "year": chunk.meta.year,
                        "doi": chunk.meta.doi,
                        "source": chunk.meta.source,
                        "start_page": chunk.meta.start_page,
                    },
                }
                f.write(json.dumps(data) + "\n")
    except Exception:
        raise


def load_bm25_from_cache() -> Optional[Tuple[BM25Okapi, List[List[str]]]]:
    """Load BM25 index from cache."""
    if not BM25_CACHE.exists() or not TOKENIZED_CACHE.exists():
        return None
    try:
        with gzip.open(TOKENIZED_CACHE, "rb") as f:
            tokenized = pickle.load(f)
        with gzip.open(BM25_CACHE, "rb") as f:
            bm25 = pickle.load(f)
        if not isinstance(bm25, BM25Okapi):
            bm25 = BM25Okapi(tokenized)
        return bm25, tokenized
    except Exception:
        return None


def save_bm25_to_cache(bm25: BM25Okapi, tokenized: List[List[str]]) -> None:
    """Save BM25 index to cache."""
    # Ensure parent directory exists
    TOKENIZED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BM25_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(TOKENIZED_CACHE, "wb") as f:
        pickle.dump(tokenized, f)
    with gzip.open(BM25_CACHE, "wb") as f:
        pickle.dump(bm25, f)


# -------------------------
# DOI metadata caching
# -------------------------


def load_doi_cache() -> Dict[str, Any]:
    """Load DOI metadata cache."""
    if not DOI_CACHE.exists():
        return {}
    try:
        with open(DOI_CACHE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_doi_cache(cache_data: Dict[str, Any]) -> None:
    """Save DOI metadata cache."""
    with open(DOI_CACHE, "w") as f:
        json.dump(cache_data, f, indent=2)


def is_doi_cache_fresh(doi: str, cache_seconds: int) -> bool:
    """Check if DOI cache entry is still fresh."""
    cache_data = load_doi_cache()
    if doi not in cache_data:
        return False

    entry = cache_data[doi]
    if "updated_at" not in entry:
        return False

    try:
        updated_at = datetime.fromisoformat(entry["updated_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - updated_at).total_seconds()
        return age < cache_seconds
    except (ValueError, KeyError):
        return False


def cache_doi_metadata(
    doi: str,
    crossref_data: Optional[Dict[str, Any]] = None,
    openalex_data: Optional[Dict[str, Any]] = None,
    unpaywall_data: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Cache DOI metadata with timestamp."""
    cache_data = load_doi_cache()

    entry = cache_data.get(doi, {})
    if crossref_data:
        entry["crossref"] = crossref_data
    if openalex_data:
        entry["openalex"] = openalex_data
    if unpaywall_data:
        entry["unpaywall"] = unpaywall_data
    if extra_fields:
        entry.update(extra_fields)

    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    cache_data[doi] = entry

    save_doi_cache(cache_data)


def get_cached_doi_metadata(doi: str) -> Optional[Dict[str, Any]]:
    """Get cached DOI metadata if available."""
    cache_data = load_doi_cache()
    return cache_data.get(doi)


# -------------------------
# Enhanced manifest with text hashing
# -------------------------


def compute_text_hash(text_content: str) -> str:
    """Compute SHA256 hash of text content for change detection."""
    return hashlib.sha256(text_content.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> str:
    """Fallback serializer for non-JSON-native config values."""

    return str(value)


def compute_chunking_config_hash(chunking_config: Optional[Dict[str, Any]]) -> str:
    """Create a stable hash for the chunking portion of the config."""

    normalized = json.dumps(chunking_config or {}, sort_keys=True, default=_json_default)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def manifest_for_dir_with_text_hash(
    pdf_dir: Path,
    corpus: Optional[List[Chunk]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate manifest with file stats, text hashes, and optional metadata."""
    manifest: Dict[str, Any] = {}

    # Create a mapping of file paths to text content from corpus
    text_by_file = {}
    if corpus:
        for chunk in corpus:
            file_path = chunk.source
            if file_path not in text_by_file:
                text_by_file[file_path] = ""
            text_by_file[file_path] += chunk.text

    for pdf_file in pdf_dir.glob("*.pdf"):
        stat = pdf_file.stat()
        file_path = str(pdf_file)

        entry = {"size": stat.st_size, "mtime": stat.st_mtime}

        # Add text hash if we have the content
        if file_path in text_by_file:
            entry["text_hash"] = compute_text_hash(text_by_file[file_path])

        manifest[file_path] = entry

    if metadata:
        manifest[MANIFEST_META_KEY] = metadata

    return manifest


def detect_changed_files(
    pdf_dir: Path, cached_manifest: Optional[Dict[str, Any]]
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Detect which PDF files have changed, are new, or have been removed.

    Returns:
        Tuple of (new_files, changed_files, removed_files)
    """
    current_files = set(pdf_dir.glob("*.pdf"))
    new_files = []
    changed_files = []
    removed_files = []

    if not cached_manifest:
        # No cache exists, all files are new
        return list(current_files), [], []

    # Handle both old and new manifest formats
    if "files" in cached_manifest:
        # Old format: {"files": [{"path": ..., "mtime": ..., "size": ...}]}
        cached_files = {Path(f["path"]) for f in cached_manifest["files"]}
        cached_file_info = {Path(f["path"]): f for f in cached_manifest["files"]}
    else:
        # New format: {"/path/to/file.pdf": {"size": ..., "mtime": ...}, "__meta__": {...}}
        cached_files = set()
        cached_file_info: Dict[Path, Any] = {}
        for key, info in cached_manifest.items():
            key_str = str(key)
            if key_str.startswith("__"):
                continue
            if not key_str.lower().endswith(".pdf"):
                continue
            path_obj = Path(key_str)
            cached_files.add(path_obj)
            cached_file_info[path_obj] = info

    # Find new and changed files
    for pdf_file in current_files:
        if pdf_file not in cached_files:
            new_files.append(pdf_file)
        else:
            # Check if file has changed
            stat = pdf_file.stat()
            cached_info = cached_file_info[pdf_file]

            # Extract mtime and size from cached info (handle both formats)
            if "mtime" in cached_info and "size" in cached_info:
                cached_mtime = cached_info["mtime"]
                cached_size = cached_info["size"]
            else:
                # Should not happen with current code, but handle gracefully
                new_files.append(pdf_file)
                continue

            if stat.st_mtime != cached_mtime or stat.st_size != cached_size:
                changed_files.append(pdf_file)

    # Find removed files
    for cached_file in cached_files:
        if cached_file not in current_files:
            removed_files.append(cached_file)

    return new_files, changed_files, removed_files


def filter_corpus_by_files(corpus: List[Chunk], keep_files: List[Path]) -> List[Chunk]:
    """
    Filter corpus to only include chunks from specified files.

    Args:
        corpus: List of chunks to filter
        keep_files: List of file paths to keep (can be Path objects or strings)

    Returns:
        Filtered list of chunks
    """
    if not corpus:
        return []

    # Convert keep_files to a set of basenames for comparison
    keep_basenames = {Path(f).name for f in keep_files}

    filtered_corpus = []
    for chunk in corpus:
        # chunk.source is typically just the basename (e.g., "document.pdf")
        if chunk.source in keep_basenames:
            filtered_corpus.append(chunk)

    return filtered_corpus
