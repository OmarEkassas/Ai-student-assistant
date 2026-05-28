"""
Shared state schema for the multi-agent graph.

🔒 LOCKED CONTRACT — do not modify without team consensus.
Every agent reads and writes a subset of these fields.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Language = Literal["ar", "en"]
Intent = Literal[
    "explain_concept",
    "summarize",
    "quiz_me",
    "study_plan",
    "compare",
    "chitchat",
    "general_knowledge",
    "unknown",
]


# ─────────────────────────────────────────────────────────────────────────────
# Sub-schemas
# ─────────────────────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """One retrieved document chunk."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0

    @property
    def source_label(self) -> str:
        src = self.metadata.get("source", "unknown")
        page = self.metadata.get("page")
        return f"{src}, p.{page}" if page is not None else src


class Citation(BaseModel):
    """A citation appearing in the final answer."""

    chunk_id: str
    label: str  # e.g. "lecture3.pdf, p.12"
    excerpt: str


class TraceEvent(BaseModel):
    """One entry in the per-session trace log."""

    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0
    input_preview: str = ""
    output_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawInput(BaseModel):
    """User input as it arrives at the system boundary."""

    text: Optional[str] = None
    image_path: Optional[str] = None


class GuardResult(BaseModel):
    """Output of input or output guard agents."""

    is_safe: bool = True
    reason: str = ""
    score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main shared state
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(BaseModel):
    """Single source of truth that flows through the entire graph."""

    # Session identity
    session_id: str
    turn_id: int = 0

    # Input
    raw_input: RawInput = Field(default_factory=RawInput)
    text: str = ""  # cleaned, post-multimodal text
    language: Language = "en"

    # Guards
    input_guard: GuardResult = Field(default_factory=GuardResult)
    output_guard: GuardResult = Field(default_factory=GuardResult)

    # Routing
    intent: Intent = "unknown"
    needs_retrieval: bool = True

    # Memory
    memory_short_term: list[str] = Field(default_factory=list)
    memory_long_term: list[str] = Field(default_factory=list)

    # RAG
    rewritten_queries: list[str] = Field(default_factory=list)
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    reranked_chunks: list[Chunk] = Field(default_factory=list)

    # Generation
    draft_answer: str = ""
    final_answer: str = ""
    citations: list[Citation] = Field(default_factory=list)

    # Observability
    trace: list[TraceEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def add_trace(self, event: TraceEvent) -> None:
        self.trace.append(event)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Final response (sent back to UI)
# ─────────────────────────────────────────────────────────────────────────────


class AssistantResponse(BaseModel):
    """What the UI shows the student."""

    answer: str
    language: Language
    citations: list[Citation]
    trace: list[TraceEvent]
    blocked: bool = False
    block_reason: str = ""
