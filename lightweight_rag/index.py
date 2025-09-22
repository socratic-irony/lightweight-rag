"""Tokenization, BM25 indexing, and caching functionality."""

import re
import os
import json
import pickle
import gzip
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from rank_bm25 import BM25Okapi

from .models import Chunk, STOP


# -------------------------
# Global cache paths (updated by main)
# -------------------------
CACHE_DIR = Path(".raq_cache")
CACHE_DIR.mkdir(exist_ok=True)
CORPUS_CACHE = CACHE_DIR / "corpus.jsonl.gz"
BM25_CACHE = CACHE_DIR / "bm25.pkl.gz"
MANIFEST_CACHE = CACHE_DIR / "manifest.json"
TOKENIZED_CACHE = CACHE_DIR / "tokenized.pkl.gz"
DOI_CACHE = CACHE_DIR / "dois.json"


# -------------------------
# Tokenization
# -------------------------

def tokenize(s: str, pattern: str = r"[A-Za-z0-9]+") -> List[str]:
    """Tokenize text using regex pattern."""
    return [t.lower() for t in re.findall(pattern, s) if t.lower() not in STOP]


# -------------------------
# BM25 Building
# -------------------------

def build_bm25(corpus: List[Chunk], token_pattern: str = r"[A-Za-z0-9]+") -> Tuple[BM25Okapi, List[List[str]]]:
    """Build BM25 index from corpus, with caching."""
    cached_result = load_bm25_from_cache()
    if cached_result is not None:
        return cached_result
    
    print("Building BM25 index...")
    tokenized = [tokenize(chunk.text, token_pattern) for chunk in corpus]
    bm25 = BM25Okapi(tokenized)
    
    save_bm25_to_cache(bm25, tokenized)
    print(f"BM25 index built and cached for {len(corpus)} chunks")
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
    manifest = {}
    for pdf_file in pdf_dir.glob("*.pdf"):
        stat = pdf_file.stat()
        manifest[str(pdf_file)] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime
        }
    return manifest


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
                    meta=meta
                )
                corpus.append(chunk)
        return corpus
    except Exception:
        return None


def save_corpus_to_cache(corpus: List[Chunk]) -> None:
    """Save corpus to cache."""
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
                    "start_page": chunk.meta.start_page
                }
            }
            f.write(json.dumps(data) + "\n")


def load_bm25_from_cache() -> Optional[Tuple[BM25Okapi, List[List[str]]]]:
    """Load BM25 index from cache."""
    if not BM25_CACHE.exists():
        return None
    try:
        with gzip.open(BM25_CACHE, "rb") as f:
            data = pickle.load(f)
            return data["bm25"], data["tokenized"]
    except Exception:
        return None


def save_bm25_to_cache(bm25: BM25Okapi, tokenized: List[List[str]]) -> None:
    """Save BM25 index to cache."""
    data = {"bm25": bm25, "tokenized": tokenized}
    with gzip.open(BM25_CACHE, "wb") as f:
        pickle.dump(data, f)


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


def cache_doi_metadata(doi: str, crossref_data: Optional[Dict[str, Any]] = None, 
                      openalex_data: Optional[Dict[str, Any]] = None,
                      unpaywall_data: Optional[Dict[str, Any]] = None,
                      extra_fields: Optional[Dict[str, Any]] = None) -> None:
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
    return hashlib.sha256(text_content.encode('utf-8')).hexdigest()


def manifest_for_dir_with_text_hash(pdf_dir: Path, corpus: Optional[List[Chunk]] = None) -> Dict[str, Any]:
    """Generate manifest with file stats and text hashes for better cache invalidation."""
    manifest = {}
    
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
        
        entry = {
            "size": stat.st_size,
            "mtime": stat.st_mtime
        }
        
        # Add text hash if we have the content
        if file_path in text_by_file:
            entry["text_hash"] = compute_text_hash(text_by_file[file_path])
        
        manifest[file_path] = entry
    
    return manifest