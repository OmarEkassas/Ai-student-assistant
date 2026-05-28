"""
LangGraph wiring (Member 1).

This is the central nervous system. Every agent is a node here.
Two early branches handle the guard-blocked paths.

Pipeline:
    raw_input
        ↓
    multimodal_preprocessor       (Bonus A — Member 8)
        ↓
    language_detector             (Bonus B — Member 10)
        ↓
    input_guard                   (Bonus C — Member 9)
        ↓
    [BLOCKED?] ─yes→ refuse_input_node → END
        ↓ no
    orchestrator                  (Member 1)
        ↓
    memory_read                   (Members 3 + 4)
        ↓
    rag_pipeline                  (Members 5 + 6)
        ↓
    synthesis                     (Member 7)
        ↓
    output_guard                  (Bonus D — Member 10)
        ↓
    [BLOCKED?] ─yes→ regenerate_or_refuse → END
        ↓ no
    memory_write                  (Member 4)
        ↓
    END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.orchestrator import orchestrator
from src.agents.synthesis import synthesis
from src.core.state import AgentState
from src.guards.input_guard import input_guard
from src.guards.output_guard import output_guard
from src.memory.long_term import extract_facts, remember
from src.memory.memory_node import memory_read
from src.memory.short_term import get_session_memory
from src.multilingual.detector import language_detector
from src.multimodal.preprocessor import multimodal_preprocessor
from src.utils.tracing import trace_agent


# ─────────────────────────────────────────────────────────────────────────────
# Tail nodes (defined here because they touch multiple subsystems)
# ─────────────────────────────────────────────────────────────────────────────


@trace_agent("refuse_input")
def refuse_input_node(state: AgentState) -> AgentState:
    """Friendly refusal when the input guard fires."""
    if state.language == "ar":
        state.final_answer = (
            "تم رفض الطلب لأنه ربما يحتوي على محاولة تلاعب بالنظام. "
            "من فضلك أعد صياغة سؤالك الأكاديمي بشكل عادي."
        )
    else:
        state.final_answer = (
            "Your request was blocked because it looks like a prompt-injection "
            "attempt. Please rephrase as a normal academic question."
        )
    return state


@trace_agent("refuse_output")
def refuse_output_node(state: AgentState) -> AgentState:
    """Fall back to a safe response when the output guard fires."""
    if state.language == "ar":
        state.final_answer = (
            "الإجابة المقترحة لم تجتز فحوصات الأمان والدقة. "
            "حاول إعادة السؤال بصياغة مختلفة."
        )
    else:
        state.final_answer = (
            "I drafted an answer, but it didn't pass safety/faithfulness checks. "
            "Please try rephrasing your question."
        )
    return state


@trace_agent("memory_write")
def memory_write(state: AgentState) -> AgentState:
    """Persist the turn to short-term memory and extract long-term facts."""
    short = get_session_memory(state.session_id)
    short.add("user", state.text)
    short.add("assistant", state.final_answer)

    facts = extract_facts(state.text, state.final_answer)
    if facts:
        remember(state.session_id, facts)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routers
# ─────────────────────────────────────────────────────────────────────────────


def _route_after_input_guard(state: AgentState) -> str:
    return "blocked" if not state.input_guard.is_safe else "safe"


def _route_after_output_guard(state: AgentState) -> str:
    return "blocked" if not state.output_guard.is_safe else "safe"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────


def build_graph():
    g = StateGraph(AgentState)

    # Pre-processing
    g.add_node("multimodal", multimodal_preprocessor)
    g.add_node("language_detector", language_detector)
    g.add_node("input_guard_node", input_guard)
    g.add_node("refuse_input", refuse_input_node)

    # Core
    g.add_node("orchestrator", orchestrator)
    g.add_node("memory_read", memory_read)
    g.add_node("rag", __import__("src.rag.rag_node", fromlist=["rag_pipeline"]).rag_pipeline)
    g.add_node("synthesis", synthesis)

    # Post-processing
    g.add_node("output_guard_node", output_guard)
    g.add_node("refuse_output", refuse_output_node)
    g.add_node("memory_write", memory_write)

    # Edges
    g.set_entry_point("multimodal")
    g.add_edge("multimodal", "language_detector")
    g.add_edge("language_detector", "input_guard_node")
    g.add_conditional_edges(
        "input_guard_node",
        _route_after_input_guard,
        {"blocked": "refuse_input", "safe": "orchestrator"},
    )
    g.add_edge("refuse_input", END)

    g.add_edge("orchestrator", "memory_read")
    g.add_edge("memory_read", "rag")
    g.add_edge("rag", "synthesis")
    g.add_edge("synthesis", "output_guard_node")

    g.add_conditional_edges(
        "output_guard_node",
        _route_after_output_guard,
        {"blocked": "refuse_output", "safe": "memory_write"},
    )
    g.add_edge("refuse_output", END)
    g.add_edge("memory_write", END)

    return g.compile()
