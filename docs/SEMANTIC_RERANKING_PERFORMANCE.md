# Semantic Reranking Performance Notes

## Overview

This document captures the current execution flow for semantic reranking and highlights why enabling it slows down the pipeline compared to the lexical-only configuration.

## Current Execution Flow

1. The baseline lexical search runs across the full candidate pool and then `build_ranking_runs` constructs additional runs (RM3, heuristic, semantic, robust query).【F:lightweight_rag/fusion.py†L118-L173】
2. When semantic reranking is enabled, the top *N* candidates from the lexical pool (default 80) are passed to `semantic_rerank` so they can be re-scored.【F:lightweight_rag/fusion.py†L142-L164】
3. `semantic_rerank` lazily instantiates a `SentenceTransformer` model, encodes the incoming query, and scores candidate chunks against cached embeddings that are populated asynchronously via a worker pool.【F:lightweight_rag/rerank.py†L129-L218】

## Cost Drivers

- **Cache warm-up**: The first time a candidate chunk appears the system must still perform an encoding pass, but subsequent queries reuse the cached vectors.【F:lightweight_rag/rerank.py†L173-L218】
- **Synchronous query scoring**: Query embeddings are generated inline for each request. While chunk encodings now happen in a background worker pool, query encoding itself still blocks the request thread.【F:lightweight_rag/rerank.py†L203-L218】

## Opportunities for Improvement

1. **Proactively warm the embedding cache** for high-traffic corpora so the first few requests avoid the cold-start cost.【F:lightweight_rag/rerank.py†L173-L218】
2. **Consider a dedicated inference service** if query throughput grows; this would decouple model execution from request handling.
3. **Expose `semantic_topn` as an operational knob** in downstream repos so integrators can cap the semantic workload to match their latency budget.

## Likely Source of the Latency Issue

The previous slowdown stemmed from forcing CPU-only inference and repeatedly encoding every candidate chunk. Those bottlenecks are now addressed in-repo through default device selection, cached chunk embeddings, and a worker pool that populates the cache incrementally.【F:lightweight_rag/rerank.py†L135-L289】 Downstream repos that vendor this project automatically inherit these improvements and only need to consider cache warm-up and `semantic_topn` tuning for their workloads.
