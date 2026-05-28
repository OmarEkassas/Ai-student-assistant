"""
Synthesis / response agent (Member 7).

Generates the final grounded answer using only the retrieved chunks
and the memory context. Emits inline [source: <label>] citations.

Owns the "I don't know" gate: when retrieval is weak, returns a graceful
refusal instead of letting the LLM invent content.

Also handles general_knowledge intent: answers from LLM's own knowledge
without requiring course documents.
"""

from __future__ import annotations

import re

from src.core.config import config
from src.core.state import AgentState, Citation
from src.utils.llm import chat
from src.utils.prompts import load_prompt, render
from src.utils.tracing import trace_agent

_CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]")


def _format_chunks(state: AgentState) -> str:
    if not state.reranked_chunks:
        return "(no relevant material was retrieved)"
    return "\n\n".join(
        f"[{c.source_label}] (chunk_id={c.id})\n{c.text}"
        for c in state.reranked_chunks
    )


def _format_memory(state: AgentState) -> str:
    parts: list[str] = []
    if state.memory_long_term:
        parts.append(
            "Durable facts about the student:\n- "
            + "\n- ".join(state.memory_long_term)
        )
    if state.memory_short_term:
        parts.append("Recent conversation:\n" + "\n".join(state.memory_short_term))
    return "\n\n".join(parts) if parts else "(no prior context)"


def _retrieval_too_weak(state: AgentState) -> bool:
    if not state.needs_retrieval:
        return False
    if not state.reranked_chunks:
        return True
    top = state.reranked_chunks[0].score
    return top < config.RETRIEVAL_SCORE_THRESHOLD


def _graceful_refusal(state: AgentState) -> str:
    if state.language == "ar":
        return (
            "آسف، لكن المواد الدراسية المتاحة لا تحتوي على معلومات كافية "
            "للإجابة على هذا السؤال. حاول إعادة صياغة السؤال أو أضف مادة مرجعية ذات صلة."
        )
    return (
        "I'm sorry — the available course material doesn't contain enough "
        "information to answer this question confidently. "
        "Try rephrasing or add relevant reference material."
    )


def _is_general_knowledge(state: AgentState) -> bool:
    """Check if this question should be answered from general LLM knowledge."""
    return getattr(state, "intent", None) == "general_knowledge"


def _answer_general_knowledge(state: AgentState) -> str:
    """Answer using the LLM's general knowledge when no retrieval is needed."""
    if state.language == "ar":
        sys_prompt = (
            "أنت مساعد طلاب ذكي. أجب على السؤال من معرفتك العامة بشكل دقيق وتعليمي. "
            "أجب باللغة العربية دائماً."
        )
    else:
        sys_prompt = (
            "You are a knowledgeable AI student assistant. "
            "Answer the question using your general knowledge in a precise and educational way. "
            "Respond in English."
        )
    memory_ctx = _format_memory(state)
    user_prompt = f"Student context (memory):\n{memory_ctx}\n\nQuestion:\n{state.text}"
    return chat(system=sys_prompt, user=user_prompt, temperature=0.3)


def _is_chitchat(state: AgentState) -> bool:
    """Check if this is casual small talk that needs a warm, friendly reply."""
    return getattr(state, "intent", None) == "chitchat"


def _answer_chitchat(state: AgentState) -> str:
    """Reply to greetings and small talk in a friendly, warm tone."""
    if state.language == "ar":
        sys_prompt = (
            "أنت مساعد طلاب ودود ومحادث. الطالب يكلمك بشكل غير رسمي أو يحييك. "
            "رد بشكل ودي وطبيعي بالعربية، كأنك صاحبه. "
            "يمكنك ذكر أنك هنا لمساعدته في دراسته إذا ناسب. "
            "لا تكن رسمياً أو جامداً، كن طبيعياً ومرحاً."
        )
    else:
        sys_prompt = (
            "You are a friendly AI student assistant having a casual chat. "
            "The student is greeting you or making small talk. "
            "Reply in a warm, natural, conversational way. "
            "You can briefly mention you're here to help with their studies if relevant. "
            "Keep it short and friendly."
        )
    return chat(system=sys_prompt, user=state.text or "مرحبا", temperature=0.7)


@trace_agent("synthesis")
def synthesis(state: AgentState) -> AgentState:
    # Chitchat path — warm, friendly reply, no retrieval needed
    if _is_chitchat(state):
        answer = _answer_chitchat(state)
        state.draft_answer = answer.strip()
        state.citations = []
        return state

    # General knowledge path — answer from LLM's own knowledge, no RAG needed
    if _is_general_knowledge(state):
        answer = _answer_general_knowledge(state)
        state.draft_answer = answer.strip()
        state.citations = []
        return state

    # Graceful refusal path
    if _retrieval_too_weak(state):
        state.draft_answer = _graceful_refusal(state)
        state.citations = []
        return state

    tmpl = load_prompt("synthesis", language=state.language)
    prompt = render(
        tmpl,
        memory=_format_memory(state),
        chunks=_format_chunks(state),
        query=state.text,
    )

    sys_prompt = (
        "You are a careful, well-cited AI student assistant. "
        f"You MUST respond in {'Arabic' if state.language == 'ar' else 'English'} "
        "and never return an empty answer. If the retrieved material is in another "
        "language than the question, translate the relevant content."
    )
    answer = chat(system=sys_prompt, user=prompt, temperature=0.2)
    state.draft_answer = answer.strip()

    # Retry once if the model returned empty text.
    if not state.draft_answer:
        answer = chat(
            system=sys_prompt + " Be concise but never empty.",
            user=prompt,
            temperature=0.4,
        )
        state.draft_answer = answer.strip()

    # Extract citations referenced in the draft.
    label_to_chunk = {c.source_label: c for c in state.reranked_chunks}
    seen: set[str] = set()
    citations: list[Citation] = []
    for label in _CITATION_RE.findall(state.draft_answer):
        label = label.strip()
        if label in seen or label not in label_to_chunk:
            continue
        chunk = label_to_chunk[label]
        citations.append(
            Citation(
                chunk_id=chunk.id,
                label=label,
                excerpt=(chunk.text[:240] + "...") if len(chunk.text) > 240 else chunk.text,
            )
        )
        seen.add(label)
    state.citations = citations
    return state
