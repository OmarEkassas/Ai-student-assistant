"""
Language detector (Bonus B, part 1).

Owner: Member 10.

Tags every query as 'ar' or 'en'. Used downstream by retrieval
(for query-translation), synthesis (for reply language), and prompts.
"""

from __future__ import annotations

from src.core.state import AgentState, Language
from src.utils.tracing import trace_agent


def detect_language(text: str) -> Language:
    """
    Heuristic + langdetect fallback.

    - If any Arabic letter is present → 'ar'.
    - Otherwise let langdetect decide between 'ar' and 'en'.
    """
    if not text or not text.strip():
        return "en"

    # Quick win: any Arabic Unicode block character.
    if any("\u0600" <= ch <= "\u06ff" for ch in text):
        return "ar"

    try:
        from langdetect import detect

        return "ar" if detect(text) == "ar" else "en"
    except Exception:  # noqa: BLE001
        return "en"


@trace_agent("language_detector")
def language_detector(state: AgentState) -> AgentState:
    state.language = detect_language(state.text)
    return state
