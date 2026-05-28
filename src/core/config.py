"""Central config loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
PROMPT_DIR = ROOT_DIR / "prompts"


class Config:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Embeddings / reranker
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # Vector store
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma")
    )
    CHROMA_COLLECTION_CORPUS: str = os.getenv(
        "CHROMA_COLLECTION_CORPUS", "student_corpus"
    )
    CHROMA_COLLECTION_MEMORY: str = os.getenv(
        "CHROMA_COLLECTION_MEMORY", "student_memory"
    )

    # RAG
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    RETRIEVE_K: int = int(os.getenv("RETRIEVE_K", "10"))
    RERANK_K: int = int(os.getenv("RERANK_K", "5"))
    RETRIEVAL_SCORE_THRESHOLD: float = float(
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35")
    )

    # Memory
    MEMORY_BUFFER_SIZE: int = int(os.getenv("MEMORY_BUFFER_SIZE", "10"))
    MEMORY_SUMMARY_THRESHOLD: int = int(
        os.getenv("MEMORY_SUMMARY_THRESHOLD", "15")
    )

    # Guards
    INJECTION_CLASSIFIER: str = os.getenv(
        "INJECTION_CLASSIFIER",
        "protectai/deberta-v3-base-prompt-injection-v2",
    )
    INJECTION_THRESHOLD: float = float(os.getenv("INJECTION_THRESHOLD", "0.7"))

    # Logging
    TRACE_LOG_DIR: str = os.getenv("TRACE_LOG_DIR", str(LOG_DIR))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
