"""
Gradio UI (Member 11).

Three columns:
  1. Chat (text + image)
  2. Citations panel
  3. Trace panel (which agents fired, with latency)

This is the visible deliverable. Spend time making it polished.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import gradio as gr

from src.core.assistant import ask
from src.core.config import DATA_DIR
from src.rag.ingestion import add_pdf_to_index
from src.rag.retriever import invalidate_bm25_cache


def _format_citations(citations) -> str:
    if not citations:
        return "_No citations._"
    return "\n\n".join(
        f"**[{c.label}]**\n\n> {c.excerpt}" for c in citations
    )


def _format_trace(trace) -> str:
    if not trace:
        return "_No trace yet._"
    rows = ["| # | Agent | Latency | Output preview |", "|---|---|---|---|"]
    for i, ev in enumerate(trace, 1):
        preview = ev.output_preview.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {i} | `{ev.agent_name}` | {ev.latency_ms:.0f} ms | {preview[:80]} |")
    return "\n".join(rows)


def _ensure_session(state):
    if state is None or not state.get("session_id"):
        return {"session_id": f"sess-{uuid.uuid4().hex[:8]}"}
    return state


def handle_pdf_upload(pdf_file) -> str:
    """Copy an uploaded PDF into the corpus folder and index it immediately."""
    if pdf_file is None:
        return "⚠️ No file selected."
    try:
        src = Path(pdf_file)
        if src.suffix.lower() != ".pdf":
            return "⚠️ Only PDF files are supported."
        dest_dir = DATA_DIR / "corpus"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        n = add_pdf_to_index(dest)
        invalidate_bm25_cache()
        return f"✅ **{src.name}** indexed successfully — {n} chunks added."
    except Exception as exc:  # noqa: BLE001
        return f"❌ Failed to index PDF: {exc}"


def handle_turn(message, history, session_state):
    session_state = _ensure_session(session_state)

    response = ask(
        text=message,
        session_id=session_state["session_id"],
    )

    if response.blocked:
        bot_msg = f"🛑 **{response.answer}**\n\n_Reason: {response.block_reason}_"
    else:
        bot_msg = response.answer

    history = history + [
        {"role": "user", "content": message or ""},
        {"role": "assistant", "content": bot_msg},
    ]
    return (
        history,
        _format_citations(response.citations),
        _format_trace(response.trace),
        session_state,
        None,    # clear textbox
    )


def reset_session(_history, _citations, _trace, _session_state):
    return [], "_No citations._", "_No trace yet._", {"session_id": f"sess-{uuid.uuid4().hex[:8]}"}


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Student Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎓 AI Student Assistant\n"
            "Multi-agent RAG system • Arabic + English • Multimodal • Guard-protected"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=520,
                    show_label=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask anything about your course material…",
                        scale=4,
                        show_label=False,
                        autofocus=True,
                    )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    reset_btn = gr.Button("New session")

            with gr.Column(scale=2):
                with gr.Tab("📚 Citations"):
                    citations_view = gr.Markdown("_No citations._")
                with gr.Tab("🔍 Agent trace"):
                    trace_view = gr.Markdown("_No trace yet._")
                with gr.Tab("📄 Upload PDF"):
                    gr.Markdown(
                        "Upload a new PDF to add it to the knowledge base instantly. "
                        "Re-uploading the same file replaces its previous chunks."
                    )
                    pdf_upload = gr.File(
                        label="Select PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("📥 Index PDF", variant="secondary")
                    upload_status = gr.Markdown("")

                    upload_btn.click(
                        handle_pdf_upload,
                        inputs=[pdf_upload],
                        outputs=[upload_status],
                    )

        session_state = gr.State({"session_id": f"sess-{uuid.uuid4().hex[:8]}"})

        send_btn.click(
            handle_turn,
            inputs=[msg, chatbot, session_state],
            outputs=[chatbot, citations_view, trace_view, session_state, msg],
        )
        msg.submit(
            handle_turn,
            inputs=[msg, chatbot, session_state],
            outputs=[chatbot, citations_view, trace_view, session_state, msg],
        )
        reset_btn.click(
            reset_session,
            inputs=[chatbot, citations_view, trace_view, session_state],
            outputs=[chatbot, citations_view, trace_view, session_state],
        )

    return demo
