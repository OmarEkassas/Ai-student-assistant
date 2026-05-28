"""
Orchestrator / router agent (Member 1).

Classifies intent and decides whether retrieval is needed.
"""

from __future__ import annotations

import json

from src.core.state import AgentState, Intent
from src.utils.llm import chat
from src.utils.prompts import load_prompt, render
from src.utils.tracing import trace_agent

VALID_INTENTS: set[Intent] = {
    "explain_concept",
    "summarize",
    "quiz_me",
    "study_plan",
    "compare",
    "chitchat",
    "general_knowledge",
    "unknown",
}

# Intents that don't require RAG retrieval
NO_RETRIEVAL_INTENTS = {"chitchat", "general_knowledge"}


@trace_agent("orchestrator")
def orchestrator(state: AgentState) -> AgentState:
    tmpl = load_prompt("router", language=state.language)
    prompt = render(tmpl, query=state.text)
    try:
        raw = chat(
            system="You are a precise intent router. Output only JSON.",
            user=prompt,
            temperature=0.0,
        )
        # Strip optional markdown fences
        raw = raw.strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        data = json.loads(raw)
        intent = data.get("intent", "unknown")
        if intent not in VALID_INTENTS:
            intent = "unknown"
        state.intent = intent  # type: ignore[assignment]
        state.needs_retrieval = bool(data.get("needs_retrieval", True))
        if intent in NO_RETRIEVAL_INTENTS:
            state.needs_retrieval = False
    except Exception as exc:  # noqa: BLE001
        state.add_error(f"orchestrator: {exc}")
        state.intent = "unknown"
        state.needs_retrieval = True
    return state
