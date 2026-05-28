"""
Cross-lingual translation helper (Bonus B, part 2).

Owner: Member 10.

Used by:
- RAG retriever: translates query to corpus language(s) for hybrid retrieval.
- Output guard: can verify language match.

We use the LLM itself for translation — keeps the dependency list small
and quality is high for AR/EN academic text.
"""

from __future__ import annotations

from src.core.state import Language
from src.utils.llm import chat


def translate(text: str, target: Language) -> str:
    """Translate text to the target language. Returns text unchanged on failure."""
    if not text.strip():
        return text
    sys = (
        "You are a precise translator. Translate the user's text to "
        f"{'Arabic' if target == 'ar' else 'English'}. "
        "Preserve technical terms. Return ONLY the translation."
    )
    try:
        return chat(system=sys, user=text, temperature=0.0).strip()
    except Exception:  # noqa: BLE001
        return text
