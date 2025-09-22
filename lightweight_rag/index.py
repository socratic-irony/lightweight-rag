"""Tokenization, BM25 indexing, and caching functionality."""

import re
import os
import json
import pickle
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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


# -------------------------
# Tokenization
# -------------------------

def tokenize(s: str, pattern: str = r"[A-Za-z0-9]+") -> List[str]:
    """Tokenize text using regex pattern."""
    return [t.lower() for t in re.findall(pattern, s) if t.lower() not in STOP]


# -------------------------
# BM25 Building
# -------------------------

def build_bm25(corpus: List[Chunk]) -> Tuple[BM25Okapi, List[List[str]]]:
    """Build BM25 index from corpus, with caching."""
    cached_result = load_bm25_from_cache()
    if cached_result is not None:
        return cached_result
    
    print("Building BM25 index...")
    tokenized = [tokenize(chunk.text) for chunk in corpus]
    bm25 = BM25Okapi(tokenized)
    
    save_bm25_to_cache(bm25, tokenized)
    print(f"BM25 index built and cached for {len(corpus)} chunks")
    return bm25, tokenized


# -------------------------
# Caching functions
# -------------------------

def update_cache_paths(cache_dir: Path) -> None:
    """Update global cache paths based on config."""
    global CACHE_DIR, CORPUS_CACHE, BM25_CACHE, MANIFEST_CACHE, TOKENIZED_CACHE
    CACHE_DIR = cache_dir
    CACHE_DIR.mkdir(exist_ok=True)
    CORPUS_CACHE = CACHE_DIR / "corpus.jsonl.gz"
    BM25_CACHE = CACHE_DIR / "bm25.pkl.gz"
    MANIFEST_CACHE = CACHE_DIR / "manifest.json"
    TOKENIZED_CACHE = CACHE_DIR / "tokenized.pkl.gz"


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