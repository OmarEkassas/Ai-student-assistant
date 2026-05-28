"""
Memory node (wraps Members 3 and 4 into a single graph step).
"""

from __future__ import annotations

from src.core.state import AgentState
from src.memory.long_term import retrieve_relevant
from src.memory.short_term import get_session_memory
from src.utils.tracing import trace_agent


@trace_agent("memory_read")
def memory_read(state: AgentState) -> AgentState:
    """Populate state.memory_short_term and state.memory_long_term."""
    short = get_session_memory(state.session_id)
    state.memory_short_term = short.get_context()
    state.memory_long_term = retrieve_relevant(state.text, k=5)
    return state
