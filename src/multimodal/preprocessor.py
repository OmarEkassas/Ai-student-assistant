"""
Multimodal preprocessor (Bonus A).

Owner: Member 8.

Accepts an optional image alongside the user's text. Runs OCR + (optional)
vision captioning, and concatenates everything into a single text field
in the AgentState that the rest of the pipeline can treat as a normal query.
"""

from __future__ import annotations

from pathlib import Path

from src.core.state import AgentState
from src.utils.tracing import trace_agent


def _ocr_image(image_path: str) -> str:
    """Run Tesseract OCR with Arabic + English language packs."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)
        # Requires `tesseract-ocr-ara` and `tesseract-ocr-eng` system packages.
        text = pytesseract.image_to_string(img, lang="ara+eng")
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        return f"[OCR failed: {exc}]"


def _caption_image(image_path: str) -> str:
    """
    Generate a short caption describing the image.

    TODO Member 8:
        Plug in a VLM here (LLaVA via Ollama, Qwen2-VL, or GPT-4o-mini
        with image input). Return one short paragraph describing the figure.
    """
    return f"[image caption stub for {Path(image_path).name}]"


@trace_agent("multimodal_preprocessor")
def multimodal_preprocessor(state: AgentState) -> AgentState:
    """Merge text + image content into a single query string."""
    parts: list[str] = []

    if state.raw_input.text:
        parts.append(state.raw_input.text.strip())

    if state.raw_input.image_path:
        ocr = _ocr_image(state.raw_input.image_path)
        caption = _caption_image(state.raw_input.image_path)
        if ocr:
            parts.append(f"[Text extracted from image]:\n{ocr}")
        if caption:
            parts.append(f"[Image description]:\n{caption}")

    state.text = "\n\n".join(parts).strip()
    return state
