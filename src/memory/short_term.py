"""
Short-term memory (Member 3).

Sliding-window conversation buffer + on-demand LLM summary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.core.config import config
from src.utils.llm import chat
from src.utils.prompts import load_prompt, render


@dataclass
class Turn:
    role: str  # 'user' | 'assistant'
    text: str


@dataclass
class ShortTermMemory:
    """Per-session sliding window of recent turns."""

    buffer: deque[Turn] = field(default_factory=lambda: deque(maxlen=config.MEMORY_BUFFER_SIZE))
    summary: str = ""

    def add(self, role: str, text: str) -> None:
        self.buffer.append(Turn(role=role, text=text))
        # Once we exceed threshold, fold old turns into a summary.
        if len(self.buffer) >= config.MEMORY_SUMMARY_THRESHOLD:
            self._summarize_oldest_half()

    def get_recent(self, n: int | None = None) -> list[Turn]:
        n = n or config.MEMORY_BUFFER_SIZE
        return list(self.buffer)[-n:]

    def get_context(self) -> list[str]:
        """Combine summary + recent turns into a list of strings for prompts."""
        out: list[str] = []
        if self.summary:
            out.append(f"[Earlier summary] {self.summary}")
        out.extend(f"{t.role}: {t.text}" for t in self.buffer)
        return out

    def _summarize_oldest_half(self) -> None:
        half = max(1, len(self.buffer) // 2)
        old_turns = [self.buffer.popleft() for _ in range(half)]
        history = "\n".join(f"{t.role}: {t.text}" for t in old_turns)
        try:
            tmpl = load_prompt("memory_summary", language="en")
            prompt = render(tmpl, history=history)
            summary = chat(system="You compress conversations into facts.", user=prompt).strip()
            self.summary = f"{self.summary} {summary}".strip()
        except Exception:  # noqa: BLE001
            # On failure, keep the raw text so we don't lose context.
            self.summary = f"{self.summary}\n{history}".strip()


# Module-level session store. Keyed by session_id.
_sessions: dict[str, ShortTermMemory] = {}


def get_session_memory(session_id: str) -> ShortTermMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ShortTermMemory()
    return _sessions[session_id]
