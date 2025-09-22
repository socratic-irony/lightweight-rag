#!/usr/bin/env python3
# raq.py  —  minimal PDF → BM25 → top-k raw chunks with author–date–page
# Features: caching, RM3 PRF, proximity+ngram bonuses, diversity control,
# optional CPU semantic rerank (sentence-transformers), Crossref page-offset citations
# python >= 3.10

import os, re, json, glob, argparse, pickle, gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
import httpx
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from dotenv import load_dotenv
from urllib.parse import quote

from config import load_full_config

# Optional lightweight semantic reranker (CPU)
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:
    np = None
    SentenceTransformer = None

# -------------------------
# Config
# -------------------------
CACHE_DIR = Path(".raq_cache")
CACHE_DIR.mkdir(exist_ok=True)
CORPUS_CACHE = CACHE_DIR / "corpus.jsonl.gz"
BM25_CACHE = CACHE_DIR / "bm25.pkl.gz"
MANIFEST_CACHE = CACHE_DIR / "manifest.json"
TOKENIZED_CACHE = CACHE_DIR / "tokenized.pkl.gz"

# -------------------------
# Utilities & Data Models
# -------------------------

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

@dataclass
class DocMeta:
    title: Optional[str]
    authors: List[str]          # ["Surname, Given", ...]
    year: Optional[int]
    doi: Optional[str]
    source: str                 # file path
    start_page: Optional[int] = None  # page offset if citation has page range (e.g., 300-314)

@dataclass
class Chunk:
    doc_id: int
    source: str
    page: int                   # 1-based PDF page index
    text: str
    meta: DocMeta

# -------------------------
# DOI + Metadata
# -------------------------

def find_doi_in_text(text: str) -> Optional[str]:
    m = DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(0).rstrip("]).,;")
    return doi

async def crossref_meta_for_doi(client: httpx.AsyncClient, doi: str) -> Optional[DocMeta]:
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
        return DocMeta(title=title, authors=authors, year=year, doi=doi, source="", start_page=start_page)
    except Exception:
        return None

# -------------------------
# Citation helpers
# -------------------------

def author_date_citation(meta: DocMeta, page: Optional[int]) -> str:
    year = str(meta.year) if meta.year else "n.d."
    if meta.authors:
        first = meta.authors[0]
        surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
        if len(meta.authors) == 2:
            s2 = (meta.authors[1].split(",")[0].strip() if "," in meta.authors[1] else meta.authors[1].split()[-1])
            name_part = f"{surname} and {s2}"
        else:
            etal = " et al." if len(meta.authors) >= 3 else ""
            name_part = f"{surname}{etal}"
    else:
        name_part = (Path(meta.source).stem if meta.source else "Unknown")

    # Adjust page by start_page offset if provided (e.g., PDF page 4 becomes citation page 303 when start_page=300)
    disp_page: Optional[int] = None
    if page is not None:
        if isinstance(meta.start_page, int):
            disp_page = meta.start_page + (page - 1)
        else:
            disp_page = page

    page_part = f", p. {disp_page}" if disp_page is not None else ""
    return f"{name_part} {year}{page_part}"

# -------------------------
# Tokenization & Helpers
# -------------------------

def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

STOP = set("""
a an and are as at be by for from has have in is it its of on or that the their this to was were with without within into between over under than then thus hence therefore however not nor but if else when where while whom whose which who what why how can may might must shall should will would could do does did done also such many most more some any each per via using used study studies paper papers result results method methods approach approaches technique techniques model models data dataset datasets system systems figure figures table tables appendix references introduction conclusion conclusions
yes no true false
design value values vsd privacy security fairness bias harms trust governance regulation policy stakeholders stakeholder users user participants participants actors actor
""".split())

from collections import Counter

def rm3_expand_query(query: str, bm25: BM25Okapi, tokenized: List[List[str]], corpus: List[Chunk], fb_docs: int = 5, fb_terms: int = 8, alpha: float = 0.6) -> str:
    """Very small RM3-style expansion: pick top fb_docs by BM25, harvest top fb_terms, append to query.
    Approximation: we just append terms (bag-of-words).
    """
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(1, fb_docs)]

    tf = Counter()
    for i in top_idxs:
        tf.update(t for t in tokenized[i] if len(t) >= 3 and t not in STOP)

    fb = [t for t, _ in tf.most_common(fb_terms) if t not in q_tokens]
    if not fb:
        return query
    expanded = query + " " + " ".join(fb)
    return expanded

def window(text: str, maxlen: int = 900) -> str:
    """Return a truncated preview of text up to maxlen characters."""
    text = text.strip().replace("\n", " ")
    if len(text) > maxlen:
        return text[: maxlen - 3] + "..."
    return text

# N-gram (bi/tri) contiguous phrase bonus

def ngram_bonus(text: str, query: str, max_hits: int = 6) -> float:
    t = " ".join(text.lower().split())
    toks = tokenize(query)
    bigrams = [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]
    trigrams = [" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)]
    hits = 0
    for ng in bigrams + trigrams:
        if len(ng) >= 5 and ng in t:
            hits += 1
            if hits >= max_hits:
                break
    return min(hits, max_hits) / max_hits

# -------------------------
# Caching
# -------------------------

def manifest_for_dir(pdf_dir: Path) -> Dict[str, Any]:
    files = sorted(glob.glob(str(pdf_dir / "*.pdf")))
    return {"files": [{"path": f, "mtime": os.path.getmtime(f), "size": os.path.getsize(f)} for f in files]}

def load_manifest() -> Optional[Dict[str, Any]]:
    if MANIFEST_CACHE.exists():
        try:
            return json.loads(MANIFEST_CACHE.read_text())
        except Exception:
            return None
    return None

def save_manifest(m: Dict[str, Any]) -> None:
    MANIFEST_CACHE.write_text(json.dumps(m))

def load_corpus_from_cache() -> Optional[List[Chunk]]:
    if not CORPUS_CACHE.exists():
        return None
    corpus: List[Chunk] = []
    try:
        with gzip.open(CORPUS_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                meta = DocMeta(**rec["meta"])  # type: ignore[arg-type]
                corpus.append(
                    Chunk(
                        doc_id=rec["doc_id"],
                        source=rec["source"],
                        page=rec["page"],
                        text=rec["text"],
                        meta=meta,
                    )
                )
        return corpus
    except Exception:
        return None

def save_corpus_to_cache(corpus: List[Chunk]) -> None:
    with gzip.open(CORPUS_CACHE, "wt", encoding="utf-8") as f:
        for c in corpus:
            rec = {
                "doc_id": c.doc_id,
                "source": c.source,
                "page": c.page,
                "text": c.text,
                "meta": {
                    "title": c.meta.title,
                    "authors": c.meta.authors,
                    "year": c.meta.year,
                    "doi": c.meta.doi,
                    "source": c.meta.source,
                    "start_page": c.meta.start_page,
                },
            }
            f.write(json.dumps(rec) + "\n")

def load_bm25_from_cache() -> Optional[Tuple[BM25Okapi, List[List[str]]]]:
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
    with gzip.open(TOKENIZED_CACHE, "wb") as f:
        pickle.dump(tokenized, f)
    with gzip.open(BM25_CACHE, "wb") as f:
        pickle.dump(bm25, f)

# -------------------------
# Ingestion (pages → chunks)
# -------------------------

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text") or ""
        if txt.strip():
            pages.append({"page": i, "text": txt})
    doc.close()
    return pages

async def build_corpus(pdf_dir: Path) -> List[Chunk]:
    # Check cache validity
    current = manifest_for_dir(pdf_dir)
    cached = load_manifest()
    if cached == current:
        corpus = load_corpus_from_cache()
        if corpus is not None:
            return corpus

    pdf_paths = sorted(glob.glob(str(pdf_dir / "*.pdf")))
    corpus: List[Chunk] = []

    if not pdf_paths:
        print(f"No PDFs found under {pdf_dir}")
        return corpus

    preinfo = []
    for p in tqdm(pdf_paths, desc="Reading PDFs", unit="pdf"):
        pages = extract_pdf_pages(p)
        head_txt = " ".join([pg["text"] for pg in pages[:2]])
        doi = find_doi_in_text(head_txt) or None
        preinfo.append((p, pages, doi))

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for idx, (p, pages, doi) in enumerate(tqdm(preinfo, desc="Metadata lookup", unit="file")):
            meta = DocMeta(title=None, authors=[], year=None, doi=doi, source=p, start_page=None)
            if doi:
                cr = await crossref_meta_for_doi(client, doi)
                if cr:
                    meta.title = cr.title or meta.title
                    meta.authors = cr.authors or meta.authors
                    meta.year = cr.year or meta.year
                    meta.start_page = cr.start_page if getattr(cr, "start_page", None) is not None else meta.start_page
            for pg in pages:
                corpus.append(Chunk(doc_id=idx, source=p, page=pg["page"], text=pg["text"], meta=meta))

    save_corpus_to_cache(corpus)
    save_manifest(current)
    return corpus

# -------------------------
# Proximity & Pattern helpers
# -------------------------

def proximity_bonus(text: str, query_tokens: List[str], window_size: int = 30) -> float:
    """Reward when distinct query tokens co-occur within a short window."""
    if not query_tokens:
        return 0.0
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    positions: Dict[str, List[int]] = {}
    qset = set(query_tokens)
    for i, tok in enumerate(tokens):
        if tok in qset:
            positions.setdefault(tok, []).append(i)
    if len([t for t in positions if positions[t]]) < 2:
        return 0.0
    distinct_terms = [t for t in positions if positions[t]]
    bonus = 0.0
    for i in range(len(distinct_terms)):
        ti = distinct_terms[i]
        for j in range(i + 1, len(distinct_terms)):
            tj = distinct_terms[j]
            pi, pj = 0, 0
            li, lj = positions[ti], positions[tj]
            best = None
            while pi < len(li) and pj < len(lj):
                d = abs(li[pi] - lj[pj])
                best = d if best is None or d < best else best
                if li[pi] < lj[pj]:
                    pi += 1
                else:
                    pj += 1
            if best is not None and best <= window_size:
                bonus += (1.0 - (best / window_size))
    return bonus

ANSWER_PATTERNS = [
    " is a ", " we define ", " we propose ", " we argue ",
    " consists of ", " stakeholders include ", " method ", " methodology ",
]

def pattern_bonus(text: str) -> float:
    tl = text.lower()
    return sum(1 for p in ANSWER_PATTERNS if p in tl) * 0.05

# -------------------------
# BM25 + Rerank + Diversity
# -------------------------

_model = None

def embed_texts(texts: List[str]) -> Optional["np.ndarray"]:
    """Return normalized embeddings or None if sentence-transformers not installed."""
    global _model
    if SentenceTransformer is None or np is None:
        return None
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB CPU-friendly
    arr = np.asarray(_model.encode(texts, normalize_embeddings=True, batch_size=64))
    return arr

def semantic_rerank(query: str, idxs: List[int], corpus: List[Chunk], topk: int) -> List[int]:
    embs = embed_texts([query] + [corpus[i].text for i in idxs])
    if embs is None:
        return idxs[:topk]
    q = embs[0]
    docs = embs[1:]
    sims = docs @ q  # cosine since normalized
    order = np.argsort(-sims)[:topk]
    return [idxs[i] for i in order]

def build_bm25(corpus: List[Chunk]) -> Tuple[BM25Okapi, List[List[str]]]:
    cached = load_bm25_from_cache()
    if cached is not None:
        return cached
    tokenized = [tokenize(c.text) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    save_bm25_to_cache(bm25, tokenized)
    return bm25, tokenized

def search_topk(
    corpus: List[Chunk], bm25: BM25Okapi, tokenized: List[List[str]], query: str, k: int = 8,
    prox_window: int = 30, prox_lambda: float = 0.2,
    ngram_lambda: float = 0.1,
    diversity: bool = True, div_lambda: float = 0.3, max_per_doc: int = 2,
    semantic: bool = False, semantic_topn: int = 80,
):
    q_tokens = tokenize(query)
    base_scores = bm25.get_scores(q_tokens)

    # Bonuses
    scores = list(base_scores)
    for i, c in enumerate(corpus):
        if prox_lambda > 0 and prox_window > 0:
            pb = proximity_bonus(c.text, q_tokens, window_size=prox_window)
            if pb:
                scores[i] += prox_lambda * pb
        if ngram_lambda > 0:
            nb = ngram_bonus(c.text, query)
            if nb:
                scores[i] += ngram_lambda * nb
        scores[i] += pattern_bonus(c.text)

    # Candidate pool
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    cand = order[:max(semantic_topn, k)] if semantic else order[:max(3 * k, 100)]

    # Optional semantic rerank
    if semantic:
        cand = semantic_rerank(query, cand, corpus, topk=max(k, min(len(cand), semantic_topn)))

    # Diversity-aware greedy selection
    selected: List[int] = []
    per_doc: Dict[int, int] = {}
    def doc_key(i: int) -> int:
        return corpus[i].doc_id

    while cand and len(selected) < k:
        best_idx = None
        best_val = None
        for i in cand[: max(5 * k, 200)]:
            base = scores[i]
            if diversity:
                count = per_doc.get(doc_key(i), 0)
                base = base - div_lambda * max(0, count)
            if best_val is None or base > best_val:
                best_val = base
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
        dk = doc_key(best_idx)
        per_doc[dk] = per_doc.get(dk, 0) + 1
        if diversity and per_doc[dk] >= max_per_doc:
            cand = [i for i in cand if doc_key(i) != dk]
        else:
            cand = [i for i in cand if i != best_idx]

    # Build results
    results = []
    for i in selected:
        c = corpus[i]
        results.append({
            "text": window(c.text, 900),
            "citation": author_date_citation(c.meta, c.page),
            "source": {
                "file": c.source,
                "page": c.page,
                "doi": c.meta.doi,
                "title": c.meta.title,
            },
            "score": round(float(scores[i]), 4),
        })
    return results

# -------------------------
# CLI
# -------------------------

async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Quick RAG over PDFs with BM25 (caching, RM3 PRF, proximity+ngram bonuses, diversity, optional semantic rerank, Crossref page-offset citations)."
    )
    # Config system
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")
    
    # Core options (can override config)
    parser.add_argument("--pdf_dir", type=str, help="Folder of PDFs")
    parser.add_argument("--k", type=int, help="Top-k chunks to return")
    parser.add_argument("--query", type=str, default=None, help="Query; if omitted, will prompt")

    # RM3
    parser.add_argument("--rm3", action="store_true", help="Enable RM3 pseudo-relevance feedback query expansion")
    parser.add_argument("--fb_docs", type=int, help="Feedback documents for RM3")
    parser.add_argument("--fb_terms", type=int, help="Feedback terms for RM3")
    parser.add_argument("--alpha", type=float, help="Mixing weight for original query in RM3")

    # Proximity & n-grams
    parser.add_argument("--no_prox", action="store_true", help="Disable proximity bonus (enabled by default)")
    parser.add_argument("--prox_window", type=int, help="Token window size for proximity bonus")
    parser.add_argument("--prox_lambda", type=float, help="Scaling weight for proximity bonus")
    parser.add_argument("--ngram_lambda", type=float, help="Weight for n-gram (bi/tri) bonus")

    # Diversity controls
    parser.add_argument("--no_diversity", action="store_true", help="Disable diversity bonus (allow many hits from same document)")
    parser.add_argument("--div_lambda", type=float, help="Diversity penalty per additional hit from the same document")
    parser.add_argument("--max_per_doc", type=int, help="Maximum results from the same document")

    # Semantic rerank
    parser.add_argument("--semantic_rerank", action="store_true", help="Enable CPU-only sentence-transformers rerank of candidates")
    parser.add_argument("--semantic_topn", type=int, help="Number of top candidates to rerank semantically")

    args = parser.parse_args()
    
    # Load configuration with full precedence
    cfg = load_full_config(args.config, args)

    pdf_dir = Path(cfg["paths"]["pdf_dir"])
    pdf_dir.mkdir(exist_ok=True)

    # Update global cache dir based on config
    global CACHE_DIR, CORPUS_CACHE, BM25_CACHE, MANIFEST_CACHE, TOKENIZED_CACHE
    CACHE_DIR = Path(cfg["paths"]["cache_dir"])
    CACHE_DIR.mkdir(exist_ok=True)
    CORPUS_CACHE = CACHE_DIR / "corpus.jsonl.gz"
    BM25_CACHE = CACHE_DIR / "bm25.pkl.gz"
    MANIFEST_CACHE = CACHE_DIR / "manifest.json"
    TOKENIZED_CACHE = CACHE_DIR / "tokenized.pkl.gz"

    # Build or load cached corpus & BM25
    corpus = await build_corpus(pdf_dir)
    if not corpus:
        return
    bm25, tokenized = build_bm25(corpus)

    # Query
    query = args.query or input("Enter your query: ").strip()
    expanded = query
    if cfg["prf"]["enabled"]:
        expanded = rm3_expand_query(
            query, bm25, tokenized, corpus,
            fb_docs=cfg["prf"]["fb_docs"],
            fb_terms=cfg["prf"]["fb_terms"],
            alpha=cfg["prf"]["alpha"]
        )
        if expanded != query:
            print(f"\n[rm3] expanded → {expanded}")

    results = search_topk(
        corpus, bm25, tokenized, expanded,
        k=cfg["rerank"]["final_top_k"],
        prox_window=(0 if not cfg["bonuses"]["proximity"]["enabled"] else cfg["bonuses"]["proximity"]["window"]),
        prox_lambda=(0.0 if not cfg["bonuses"]["proximity"]["enabled"] else cfg["bonuses"]["proximity"]["weight"]),
        ngram_lambda=cfg["bonuses"]["ngram"]["weight"],
        diversity=cfg["diversity"]["enabled"],
        div_lambda=cfg["diversity"]["per_doc_penalty"],
        max_per_doc=cfg["diversity"]["max_per_doc"],
        semantic=cfg["rerank"]["semantic"]["enabled"],
        semantic_topn=cfg["rerank"]["semantic"]["topn"],
    )

    print("\n=== Top Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("Starting RAG pipeline...")
    import asyncio
    asyncio.run(main())