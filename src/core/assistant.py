"""
Public assistant API.

Single function `ask(...)` that takes user input and returns an
AssistantResponse. Used by the UI and by scripts/eval.py.
"""

from __future__ import annotations

import uuid
from typing import Optional

from src.core.graph import build_graph
from src.core.state import AgentState, AssistantResponse, RawInput
from src.utils.tracing import persist_trace

# Build the graph once at import time.
_graph = build_graph()


def ask(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AssistantResponse:
    """Run one turn through the multi-agent graph."""
    session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"

    init = AgentState(
        session_id=session_id,
        raw_input=RawInput(text=text, image_path=image_path),
    )

    # LangGraph returns a plain dict matching the schema fields.
    final_dict = _graph.invoke(init)
    final = AgentState.model_validate(final_dict)

    persist_trace(final)

    return AssistantResponse(
        answer=final.final_answer or "(no answer produced)",
        language=final.language,
        citations=final.citations,
        trace=final.trace,
        blocked=not (final.input_guard.is_safe and final.output_guard.is_safe),
        block_reason=final.input_guard.reason or final.output_guard.reason,
    )
