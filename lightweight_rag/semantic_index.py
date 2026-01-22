"""ANN-backed semantic candidate selection with optional FAISS cache."""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from .index import CACHE_DIR, load_manifest
from .rerank import _load_model


def _hash_payload(payload: dict) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cache_key(model_name: str) -> Optional[str]:
    manifest = load_manifest()
    if manifest is None:
        return None
    payload = {"model": model_name, "manifest": manifest}
    return _hash_payload(payload)


def _cache_paths(key: str) -> Tuple[Path, Path, Path]:
    embeddings_path = CACHE_DIR / f"semantic_embeddings_{key}.npy"
    index_path = CACHE_DIR / f"semantic_index_{key}.faiss"
    meta_path = CACHE_DIR / f"semantic_index_{key}.json"
    return embeddings_path, index_path, meta_path


def _load_cached_index(key: str, expected_count: int):
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None, None

    embeddings_path, index_path, meta_path = _cache_paths(key)
    if not embeddings_path.exists() or not index_path.exists() or not meta_path.exists():
        return None, None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("count") != expected_count:
            return None, None

        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] != expected_count:
            return None, None

        index = faiss.read_index(str(index_path))
        return index, embeddings
    except Exception:
        return None, None


def _save_cached_index(key: str, embeddings, index, count: int) -> None:
    embeddings_path, index_path, meta_path = _cache_paths(key)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore

        np.save(embeddings_path, embeddings)
        faiss.write_index(index, str(index_path))
        with open(meta_path, "w") as f:
            json.dump({"count": count}, f)
    except Exception:
        return


def _embed_corpus(texts: List[str], model_name: str, batch_size: int = 64):
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    model = _load_model(model_name)
    if model is None:
        return None

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_emb = model.encode(batch, convert_to_numpy=True)
        norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings.append(batch_emb / norms)

    if not embeddings:
        return None

    return np.vstack(embeddings)


def _build_faiss_index(embeddings):
    try:
        import faiss  # type: ignore
    except Exception:
        return None

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


def ann_select_candidates(
    texts: List[str],
    query: str,
    topn: int,
    model_name: str,
) -> Optional[List[int]]:
    """Return ANN-selected candidate indices using FAISS if available."""
    if not texts or topn <= 0:
        return None

    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore
    except Exception:
        return None

    cache_key = _cache_key(model_name)
    index = None
    embeddings = None
    if cache_key:
        index, embeddings = _load_cached_index(cache_key, len(texts))

    if index is None or embeddings is None:
        embeddings = _embed_corpus(texts, model_name)
        if embeddings is None:
            return None
        index = _build_faiss_index(embeddings)
        if index is None:
            return None
        if cache_key:
            _save_cached_index(cache_key, embeddings, index, len(texts))

    from .rerank import embed_texts

    query_embedding = embed_texts([query], model_name)
    if query_embedding is None:
        return None

    query_vec = query_embedding.astype("float32")
    k = min(topn, len(texts))
    distances, indices = index.search(query_vec, k)
    if indices is None:
        return None
    return [int(i) for i in indices[0] if i >= 0]
