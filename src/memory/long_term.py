"""
Long-term semantic memory (Member 4).

After each successful turn, an LLM extractor pulls durable facts about the
student (e.g. "studies CS", "weak in linear algebra") and persists them in a
ChromaDB collection. At query time we retrieve the most relevant facts.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from src.core.config import config
from src.utils.llm import chat


def _get_collection():
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(config.CHROMA_COLLECTION_MEMORY)


def extract_facts(user_text: str, assistant_text: str) -> list[str]:
    """Use the LLM to pull durable facts about the student from one turn."""
    sys = (
        "Extract durable facts about the student from the conversation turn. "
        "A durable fact is something true across many future conversations "
        "(courses studied, weak/strong topics, preferences, goals). "
        "Ignore one-off questions. "
        'Respond ONLY as JSON: {"facts": ["fact1", "fact2", ...]}. '
        'If there is nothing durable, return {"facts": []}.'
    )
    user = f"USER: {user_text}\nASSISTANT: {assistant_text}"
    try:
        raw = chat(system=sys, user=user, temperature=0.0)
        data = json.loads(raw.strip().strip("`"))
        return [f.strip() for f in data.get("facts", []) if f.strip()]
    except Exception:  # noqa: BLE001
        return []


def remember(session_id: str, facts: list[str]) -> None:
    """Persist facts to the vector store."""
    if not facts:
        return
    col = _get_collection()
    col.add(
        ids=[str(uuid.uuid4()) for _ in facts],
        documents=facts,
        metadatas=[
            {"session_id": session_id, "ts": datetime.utcnow().isoformat()}
            for _ in facts
        ],
    )


def retrieve_relevant(query: str, k: int = 5) -> list[str]:
    """Pull the top-k facts most relevant to the current query."""
    col = _get_collection()
    try:
        res = col.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        return [d for d in docs if d]
    except Exception:  # noqa: BLE001
        return []
