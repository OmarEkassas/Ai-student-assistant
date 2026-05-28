"""
RAG node — orchestrates query rewriting → hybrid retrieval → (optional) reranking.
"""

from __future__ import annotations

import os

from src.core.config import config
from src.core.state import AgentState
from src.multilingual.translator import translate
from src.rag.retriever import hybrid_retrieve, rerank
from src.utils.tracing import trace_agent

# Set RAG_ENABLE_RERANKER=1 in .env to turn the reranker back on.
_RERANKER_ENABLED = os.getenv("RAG_ENABLE_RERANKER", "0") == "1"


def _build_rewrites(state: AgentState) -> list[str]:
    """
    Only translate the query when it's clearly worth it.
    For first-version speed, we skip translation by default.
    Set RAG_ENABLE_QUERY_TRANSLATION=1 in .env to turn it on.
    """
    queries = [state.text]
    if os.getenv("RAG_ENABLE_QUERY_TRANSLATION", "0") == "1":
        other = "ar" if state.language == "en" else "en"
        translated = translate(state.text, target=other)
        if translated and translated != state.text:
            queries.append(translated)
    return queries


@trace_agent("rag_pipeline")
def rag_pipeline(state: AgentState) -> AgentState:
    """Run the full RAG pipeline: rewrite → retrieve → (optional) rerank."""
    if not state.needs_retrieval:
        return state

    queries = _build_rewrites(state)
    state.rewritten_queries = queries

    # Retrieve for every query variant and merge.
    seen: dict[str, object] = {}
    for q in queries:
        for c in hybrid_retrieve(q, k=config.RETRIEVE_K):
            seen.setdefault(c.id, c)
    state.retrieved_chunks = list(seen.values())

    # Reranking is slow on CPU. By default we just take the top-K from retrieval.
    if _RERANKER_ENABLED:
        state.reranked_chunks = rerank(
            state.text, state.retrieved_chunks, top_k=config.RERANK_K
        )
    else:
        state.reranked_chunks = state.retrieved_chunks[: config.RERANK_K]

    return state