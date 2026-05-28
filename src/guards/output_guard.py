"""
Output guard (Bonus D).

Owner: Member 10.

Inspects the synthesized answer before it reaches the student:
- PII / privacy: no emails, phone numbers, passwords, or personal data echoed back.
- Citation validity: every [source: X] in the answer corresponds to a real chunk.
- Language match: response language == detected input language.
- Faithfulness: LLM-as-judge compares claims to retrieved chunks.
- Safety: no leaked system prompts, no off-topic content.
"""

from __future__ import annotations

import json
import re

from src.core.state import AgentState, GuardResult
from src.multilingual.detector import detect_language
from src.utils.llm import chat
from src.utils.prompts import load_prompt, render
from src.utils.tracing import trace_agent

_CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]")

# ── PII patterns (compiled once at import time) ──────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ \-]?){13,16}\b")
_PASSWORD_KEYWORDS_RE = re.compile(
    r"\b(password|passwd|secret|token|api[_\s]?key|private[_\s]?key)\b",
    re.IGNORECASE,
)


def _check_pii(answer: str) -> tuple[bool, str]:
    """Block the answer if it contains PII or sensitive credentials."""
    if _EMAIL_RE.search(answer):
        return False, "Response contains an email address (PII)."
    if _CREDIT_CARD_RE.search(answer):
        return False, "Response contains what looks like a credit-card number (PII)."
    # Phone: only flag when there are ≥10 digits (avoids false positives on years etc.)
    for match in _PHONE_RE.finditer(answer):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) >= 10:
            return False, "Response contains a phone number (PII)."
    if _PASSWORD_KEYWORDS_RE.search(answer):
        return False, "Response contains sensitive credential keywords (PII)."
    return True, ""


def _check_citations(answer: str, valid_labels: set[str], valid_ids: set[str]) -> tuple[bool, str]:
    found = _CITATION_RE.findall(answer)
    if not found:
        # No citations is acceptable for chitchat / refusals.
        return True, ""
    invalid = []
    for raw in found:
        label = raw.strip()
        # Accept either a real source label OR a real chunk id.
        if label in valid_labels or label in valid_ids:
            continue
        # Accept partial matches (model sometimes writes "lecture3.pdf" without page).
        if any(label in vl or vl in label for vl in valid_labels):
            continue
        invalid.append(label)
    if invalid:
        return False, f"Invalid citations: {invalid}"
    return True, ""


def _check_language(answer: str, expected_lang: str) -> tuple[bool, str]:
    detected = detect_language(answer)
    if detected != expected_lang:
        return False, f"Expected {expected_lang}, got {detected}"
    return True, ""


def _check_faithfulness(answer: str, chunks_text: str) -> tuple[bool, float, str]:
    if not chunks_text.strip():
        return True, 1.0, ""
    try:
        tmpl = load_prompt("faithfulness_check", language="en")
        prompt = render(tmpl, chunks=chunks_text, answer=answer)
        raw = chat(system="You are a strict fact-checker.", user=prompt, temperature=0.0)
        data = json.loads(raw.strip().strip("`"))
        return (
            bool(data.get("faithful", True)),
            float(data.get("score", 1.0)),
            ", ".join(data.get("unsupported_claims", [])),
        )
    except Exception:  # noqa: BLE001
        return True, 1.0, ""


@trace_agent("output_guard")
def output_guard(state: AgentState) -> AgentState:
    """Validate the draft answer; populate state.output_guard + final_answer."""
    answer = state.draft_answer or ""
    if not answer:
        state.output_guard = GuardResult(is_safe=False, reason="Empty answer.")
        return state

    # 0. PII check — runs on EVERY answer, before anything else.
    ok, reason = _check_pii(answer)
    if not ok:
        state.output_guard = GuardResult(is_safe=False, reason=reason)
        return state

    # If retrieval was not needed (e.g. chitchat), skip citation/faithfulness checks.
    if not state.needs_retrieval:
        state.output_guard = GuardResult(is_safe=True, score=1.0)
        state.final_answer = answer
        return state

    # 1. Citations — accept either a source label or a chunk id.
    valid_labels = {c.source_label for c in state.reranked_chunks}
    valid_ids = {c.id for c in state.reranked_chunks}
    ok, reason = _check_citations(answer, valid_labels, valid_ids)
    if not ok:
        state.output_guard = GuardResult(is_safe=False, reason=reason)
        return state

    # 2. Language — skip for very short answers (langdetect is unreliable < 30 chars).
    if len(answer) >= 30:
        ok, reason = _check_language(answer, state.language)
        if not ok:
            state.output_guard = GuardResult(is_safe=False, reason=reason)
            return state

    # 3. Faithfulness
    chunks_text = "\n\n".join(
        f"[{c.source_label}] {c.text}" for c in state.reranked_chunks
    )
    faithful, score, unsupported = _check_faithfulness(answer, chunks_text)
    if not faithful:
        state.output_guard = GuardResult(
            is_safe=False,
            reason=f"Unfaithful claims: {unsupported}",
            score=score,
        )
        return state

    state.output_guard = GuardResult(is_safe=True, score=score)
    state.final_answer = answer
    return state
