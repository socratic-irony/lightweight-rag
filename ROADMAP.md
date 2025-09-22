Here’s a focused roadmap with concrete refactors, a config system, and targeted upgrades that keep the project fast, lean, and reliable.

1) Centralize settings in a config

Move all knobs into a single config file with clear precedence: defaults → file (YAML/TOML) → env vars → CLI flags.

Example config.yaml

paths:
  pdf_dir: "pdfs"
  cache_dir: ".raq_cache"
  crossref_email: null           # optional; some APIs accept mailto

indexing:
  page_split: "page"             # "page" | "sentence" | "sliding"
  window_chars: 1000             # used when page_split != "page"
  overlap_chars: 120

bm25:
  k1: 1.5
  b: 0.75
  build_top_k: 300               # candidate pool from lexical stage (pre-rerank)
  token_pattern: "[A-Za-z0-9]+"

prf:                              # RM3
  enabled: false
  fb_docs: 6
  fb_terms: 10

bonuses:
  proximity:
    enabled: true
    window: 30
    weight: 0.2
  ngram:
    enabled: true
    weight: 0.1
  patterns:
    enabled: true
    patterns: [" is a ", " we define ", " we propose ", " we argue ",
               " consists of ", " stakeholders include ", " method ", " methodology "]
    weight_per_hit: 0.05
    max_hits: 6

diversity:
  enabled: true
  per_doc_penalty: 0.3
  max_per_doc: 2

rerank:
  semantic:
    enabled: true
    model: "sentence-transformers/all-MiniLM-L6-v2"
    topn: 120                    # candidates to rerank
  final_top_k: 8                 # number of items to return/print

citations:
  crossref: true
  openalex: true
  unpaywall: false
  cache_seconds: 604800          # 7 days
  page_offset_from_crossref: true
  # future: try page offset from OpenAlex host venue pages if Crossref lacks 'page'

output:
  max_snippet_chars: 900
  pretty_json: true
  include_scores: true

Loader pattern (minimal)

# pip install pyyaml
import os, yaml
from argparse import Namespace

def load_config(path="config.yaml") -> dict:
    cfg = {}  # defaults baked into code if file missing
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    return cfg

def override(cfg: dict, cli: Namespace) -> dict:
    # Apply CLI/env overrides onto cfg; skip empty values
    # Example: if cli.k is set, set cfg["rerank"]["final_top_k"] = cli.k
    return cfg

2) Modularize the codebase

Split the current script into small modules (each ≤200 lines). Keep import times low.

lightweight_rag/
  __init__.py
  config.py         # load/merge config; env and CLI precedence
  io_pdf.py         # extraction (PyMuPDF), optional sentence split, chunk windows
  index.py          # tokenization, BM25 build/load, caches
  scoring.py        # BM25 scoring + bonuses (proximity, n-gram, patterns)
  prf.py            # RM3
  rerank.py         # sentence-transformers (lazy load) + optional char-gram TF-IDF fallback
  diversity.py      # greedy selection across docs
  cite.py           # DOI extraction, Crossref/OpenAlex/Unpaywall, caches
  cli.py            # argument parsing, invokes main()
  main.py           # pipeline orchestration

Benefits: faster cold-start (only import what’s needed), easier testing, and clearer ownership of each concern.

3) Performance and robustness improvements
	•	Tokenization: keep the current regex for speed; expose it in config for experimentation.
	•	Caching:
	•	Add a metadata cache (dois.json) keyed by DOI with Crossref/OpenAlex/Unpaywall payloads and an updated_at timestamp. Respect citations.cache_seconds.
	•	Store per-PDF text hash (e.g., SHA256 of concatenated page text) in the manifest to avoid re-parsing changed PDFs with the same filename/mtime.
	•	Concurrency:
	•	Batch Crossref/OpenAlex calls with a small async semaphore (e.g., 5 in-flight).
	•	Optionally parallelize PDF parsing with concurrent.futures.ThreadPoolExecutor (cap workers to number of cores).
	•	Determinism:
	•	Seed numpy if you add any stochastic rerankers.
	•	When ties occur, prefer earlier pages and then lexicographic file order.

4) Retrieval quality without heavy models
	•	Better PRF (still cheap):
	•	Filter feedback terms by document frequency (drop terms present in >25% of chunks).
	•	Restrict to alphabetic tokens of length ≥3.
	•	Impose a per-term cap to keep expansions short.
	•	Enhanced lexical signals:
	•	Phrase bigram whitelist based on the query (we already add n-grams; consider giving bigrams 2× the weight of unigrams when both present).
	•	Heading/abstract weighting: if you detect “Abstract” or large-font headings (PyMuPDF font size heuristic), multiply those chunk scores by, say, 1.15.
	•	RRF (reciprocal rank fusion) across variants:
	•	Fuse rankings from: (a) raw query, (b) RM3-expanded, (c) phrase-boosted. This is ~20 lines and consistently lifts recall with negligible cost.

5) Citation enrichment (Crossref, OpenAlex, Unpaywall)

Keep it simple and cached:

Data flow
	1.	DOI sniff from first 2 pages (you have this).
	2.	Crossref /works/{doi}
	•	Use message.page to set start_page (you did this).
	•	Fall back to issued or published-online/print for year.
	3.	OpenAlex /works/https://doi.org/DOI
	•	Fill missing fields: venue, publisher, concept tags.
	•	Add canonical OA URL if present (OpenAlex mirrors Unpaywall signals).
	4.	Unpaywall /v2/{doi}?email=... (optional)
	•	Get best_oa_location.url_for_pdf for a verified OA link.

Caching
	•	dois.json:

{
  "10.xxxxx/abc": {
    "crossref": {...}, "openalex": {...}, "unpaywall": {...},
    "start_page": 300, "year": 2018, "authors": ["Friedman, Batya", ...],
    "title": "...", "updated_at": "2025-09-21T12:34:56Z"
  }
}


	•	Refresh if stale by citations.cache_seconds.

6) Configurable chunking

Add a light sentence splitter to enable “sentence” or “sliding” windows:
	•	Fast option: regex split on [.!?] with heuristics to avoid abbreviations.
	•	Optional dependency: syntok (very small, accurate) for sentence boundaries.
	•	Keep metadata: (file, page, window_start_char, window_end_char).
	•	Configure window_chars and overlap_chars in indexing.

7) Testing and evaluation
	•	Unit tests for:
	•	DOI regex and Crossref page-offset logic.
	•	Proximity and n-gram bonuses (deterministic inputs).
	•	Diversity selection invariants (max per doc, monotonicity).
	•	Golden tests:
	•	For a small “fixtures” folder, assert that a fixed query returns specific citations (filenames + pages) under a fixed config.
	•	Perf guardrails:
	•	Add a simple timer report per stage (parse, index, search, rerank, enrich) to catch regressions.

8) Logging/telemetry (minimal)
	•	Structured logs (stdout JSON) at INFO for stage boundaries and counts:
	•	{"stage":"index","chunks":1234,"elapsed_ms":...}
	•	{"stage":"rerank","candidates":120,"elapsed_ms":...}
	•	At DEBUG, optionally dump top-terms from PRF and the bonuses applied (guarded to avoid noise).

9) API surface (optional)

Wrap the pipeline behind a tiny function you can import elsewhere:

def query_pdfs(query: str, cfg: dict) -> List[dict]:
    # returns [{text, citation, source:{file,page,doi,title}, score}, ...]
    ...

This lets you call it from a web UI or your Writer/Revise stage cleanly.
