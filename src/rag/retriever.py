"""
Hybrid retrieval + reranking (Member 6).

- Dense: SentenceTransformer embeddings stored in Chroma.
- Sparse: BM25 over the same documents (lazy-loaded for the first query).
- Fusion: Reciprocal Rank Fusion.
- Rerank: cross-encoder for the final top-K.
"""

from __future__ import annotations

import math
from functools import lru_cache

from src.core.config import config
from src.core.state import Chunk


@lru_cache(maxsize=1)
def _get_client_and_collection():
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(config.CHROMA_COLLECTION_CORPUS)
    return client, col


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _build_bm25():
    """Build a BM25 index by reading all docs out of the Chroma collection."""
    from rank_bm25 import BM25Okapi

    _, col = _get_client_and_collection()
    res = col.get(include=["documents", "metadatas"])
    ids = res["ids"]
    docs = res["documents"]
    metas = res["metadatas"] or [{} for _ in ids]
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return bm25, ids, docs, metas


def _dense_search(query: str, k: int) -> list[Chunk]:
    _, col = _get_client_and_collection()
    embedder = _get_embedder()
    qvec = embedder.encode(query).tolist()
    n_results = min(k, max(col.count(), 1))
    res = col.query(query_embeddings=[qvec], n_results=n_results)
    chunks: list[Chunk] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0] or [{}] * len(ids)
    dists = res.get("distances", [[]])[0] or [0.0] * len(ids)
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        score = 1.0 / (1.0 + max(0.0, dist))
        chunks.append(Chunk(id=cid, text=doc, metadata=meta or {}, score=score))
    return chunks


def _bm25_search(query: str, k: int) -> list[Chunk]:
    bm25, ids, docs, metas = _build_bm25()
    if bm25 is None or not ids:
        return []
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:k]
    max_score = max(scores) if scores.size else 1.0
    return [
        Chunk(
            id=ids[i],
            text=docs[i],
            metadata=metas[i] or {},
            score=float(scores[i] / max_score) if max_score else 0.0,
        )
        for i in ranked
    ]


def _rrf_fuse(*lists: list[Chunk], k_const: int = 60) -> list[Chunk]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    objs: dict[str, Chunk] = {}
    for chunks in lists:
        for rank, c in enumerate(chunks):
            scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (k_const + rank + 1)
            objs[c.id] = c
    fused = sorted(objs.values(), key=lambda c: scores[c.id], reverse=True)
    for c in fused:
        c.score = scores[c.id]
    return fused


def invalidate_bm25_cache() -> None:
    """Call this after adding new documents so the BM25 index is rebuilt."""
    _build_bm25.cache_clear()


def hybrid_retrieve(query: str, k: int | None = None) -> list[Chunk]:
    k = k or config.RETRIEVE_K
    dense = _dense_search(query, k)
    sparse = _bm25_search(query, k)
    fused = _rrf_fuse(dense, sparse)[:k]

    # Normalize scores into (0, 1] so the downstream threshold check is sensible
    # when reranking is disabled.
    if fused:
        max_score = max(c.score for c in fused) or 1.0
        for c in fused:
            c.score = c.score / max_score
    return fused


@lru_cache(maxsize=1)
def _get_reranker():
    """Load the cross-encoder reranker."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query: str, chunks: list[Chunk], top_k: int | None = None) -> list[Chunk]:
    top_k = top_k or config.RERANK_K
    if not chunks:
        return []
    try:
        reranker = _get_reranker()
        pairs = [(query, c.text) for c in chunks]
        scores = reranker.predict(pairs).tolist()
    except Exception:  # noqa: BLE001
        return chunks[:top_k]

    for c, s in zip(chunks, scores):
        c.score = 1.0 / (1.0 + math.exp(-s))
    return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]