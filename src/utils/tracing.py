"""
Trace logger. Wraps every agent so we get latency + I/O for the UI panel
and the final report. Member 2 owns extensions to this file.
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any, Callable

from src.core.config import config
from src.core.state import AgentState, TraceEvent


def _preview(value: Any, n: int = 200) -> str:
    s = str(value)
    return s if len(s) <= n else s[:n] + "..."


def trace_agent(agent_name: str) -> Callable:
    """
    Decorator: any function `(state: AgentState) -> AgentState` becomes
    automatically traced.

    Usage:
        @trace_agent("input_guard")
        def input_guard(state: AgentState) -> AgentState:
            ...
    """

    def decorator(fn: Callable[[AgentState], AgentState]) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> AgentState:
            start = time.perf_counter()
            input_preview = _preview(state.text)
            try:
                new_state = fn(state)
            except Exception as exc:  # noqa: BLE001
                latency = (time.perf_counter() - start) * 1000
                state.add_error(f"[{agent_name}] {exc}")
                state.add_trace(
                    TraceEvent(
                        agent_name=agent_name,
                        latency_ms=latency,
                        input_preview=input_preview,
                        output_preview=f"ERROR: {exc}",
                        metadata={"error": True},
                    )
                )
                return state

            latency = (time.perf_counter() - start) * 1000
            output_preview = _preview(new_state.final_answer or new_state.draft_answer)
            new_state.add_trace(
                TraceEvent(
                    agent_name=agent_name,
                    latency_ms=latency,
                    input_preview=input_preview,
                    output_preview=output_preview,
                )
            )
            return new_state

        return wrapper

    return decorator


def persist_trace(state: AgentState) -> None:
    """Append the trace events for this turn to a JSON lines file."""
    Path(config.TRACE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    fpath = Path(config.TRACE_LOG_DIR) / f"{state.session_id}.jsonl"
    with open(fpath, "a", encoding="utf-8") as f:
        for ev in state.trace:
            f.write(ev.model_dump_json() + "\n")
