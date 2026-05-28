"""
Corpus ingestion (Member 5).

Loads documents from data/corpus/, chunks them, embeds each chunk,
and stores them in ChromaDB with rich metadata.

Re-runnable: drops and rebuilds the collection on each invocation.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from src.core.config import DATA_DIR, config


def _ocr_page_via_groq(png_bytes: bytes, page_num: int, source_name: str) -> str:
    """Send a page image to Groq vision API and extract its text.
    Uses llama-4-scout which supports image input and handles Arabic + English.
    No local OCR installation required.
    """
    import base64
    import urllib.request
    import json

    api_key = config.GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")

    b64 = base64.b64encode(png_bytes).decode()
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text from this page exactly as it appears. "
                            "Preserve Arabic and English text, headings, bullet points, "
                            "and paragraph breaks. Output only the extracted text, nothing else."
                        ),
                    },
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0,
    }

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())

    return result["choices"][0]["message"]["content"].strip()


def _load_pdf(path: Path) -> list[tuple[str, dict]]:
    """Return a list of (text, metadata) per page.

    Strategy:
    1. Use PyMuPDF (fitz) to extract embedded text — fast, no dependencies.
    2. For image-only pages (scanned PDFs), rasterize with PyMuPDF and send
       the page image to the Groq vision API for OCR.
       No poppler, no Tesseract, no local OCR tooling required.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is not installed. Run: pip install pymupdf")

    out: list[tuple[str, dict]] = []
    doc = fitz.open(str(path))

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            out.append((text, {"source": path.name, "page": i + 1}))
        else:
            # Scanned / image-only page: use Groq vision for OCR
            try:
                pix = page.get_pixmap(dpi=200)
                png_bytes = pix.tobytes("png")
                ocr_text = _ocr_page_via_groq(png_bytes, i + 1, path.name)
                if ocr_text:
                    out.append((ocr_text, {"source": path.name, "page": i + 1, "ocr": True}))
                    print(f"    OCR via Groq: page {i + 1} of {path.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! OCR failed for page {i + 1} of {path.name}: {exc}")

    doc.close()
    out.sort(key=lambda x: x[1].get("page", 0))
    return out


def _load_text(path: Path) -> list[tuple[str, dict]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, {"source": path.name, "page": 1})] if text.strip() else []


def _load_docx(path: Path) -> list[tuple[str, dict]]:
    try:
        from docx import Document

        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(text, {"source": path.name, "page": 1})] if text.strip() else []
    except Exception:  # noqa: BLE001
        return []


LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
    ".docx": _load_docx,
}


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Recursive character splitting via LangChain (handles edge cases well)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_text(text)


def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def _detect_chunk_language(text: str) -> str:
    if any("\u0600" <= ch <= "\u06ff" for ch in text):
        return "ar"
    return "en"


def iter_corpus_files(corpus_dir: Path) -> Iterable[Path]:
    for f in corpus_dir.rglob("*"):
        if f.suffix.lower() in LOADERS:
            yield f


def add_pdf_to_index(pdf_path: Path) -> int:
    """
    Incrementally add a single PDF to the existing corpus collection.
    Skips chunks already indexed from the same source (by filename).
    Returns the number of new chunks added.
    """
    import chromadb
    from chromadb.config import Settings

    Path(config.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(config.CHROMA_COLLECTION_CORPUS)

    # Remove any previously indexed chunks for this file so re-uploads are clean.
    try:
        existing = col.get(where={"source": pdf_path.name})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:  # noqa: BLE001
        pass

    embedder = _embedder()
    pages = _load_pdf(pdf_path)
    total = 0

    for text, meta in pages:
        for chunk in _chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            meta_full = {**meta, "language": _detect_chunk_language(chunk)}
            col.add(
                ids=[str(uuid.uuid4())],
                documents=[chunk],
                metadatas=[meta_full],
                embeddings=[embedder.encode(chunk).tolist()],
            )
            total += 1

    return total


def build_index() -> int:
    """Build (or rebuild) the corpus collection. Returns number of chunks."""
    import chromadb
    from chromadb.config import Settings

    corpus_dir = DATA_DIR / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    Path(config.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    # Fresh rebuild
    try:
        client.delete_collection(config.CHROMA_COLLECTION_CORPUS)
    except Exception:  # noqa: BLE001
        pass
    col = client.create_collection(config.CHROMA_COLLECTION_CORPUS)

    embedder = _embedder()
    total = 0

    for f in iter_corpus_files(corpus_dir):
        loader = LOADERS[f.suffix.lower()]
        try:
            pages = loader(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Failed to load {f.name}: {exc}")
            continue

        for text, meta in pages:
            for chunk in _chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                meta_full = {
                    **meta,
                    "language": _detect_chunk_language(chunk),
                }
                col.add(
                    ids=[str(uuid.uuid4())],
                    documents=[chunk],
                    metadatas=[meta_full],
                    embeddings=[embedder.encode(chunk).tolist()],
                )
                total += 1

        print(f"  ✓ Indexed {f.name}")

    print(f"\nTotal chunks indexed: {total}")
    return total
