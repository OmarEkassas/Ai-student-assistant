"""Build (or rebuild) the corpus vector index from data/corpus/."""

from __future__ import annotations

from src.rag.ingestion import build_index


if __name__ == "__main__":
    print("🔨 Building corpus index from data/corpus/ …\n")
    n = build_index()
    print(f"\n✅ Done. {n} chunks indexed.")
