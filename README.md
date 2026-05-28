# 🎓 AI Student Assistant

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A multi-agent AI system that answers academic questions over lecture material in **Arabic and English**, with retrieval grounding, multimodal input, memory, and prompt-injection defenses.

> Built as a course project for **Deep Generative Models**, Faculty of Artificial Intelligence.

---

## ✨ Features

- 🧠 **RAG pipeline** — hybrid retrieval over course PDFs/slides, with reranking and citations
- 🖼️ **Multimodal input** — text + image (OCR + vision captioning)
- 🌍 **Bilingual** — Arabic and English, with automatic language detection
- 🛡️ **Prompt-injection guard** — input sanitization via a classifier
- ✅ **Output guardrails** — faithfulness checks and citation enforcement
- 💬 **Memory** — short-term buffer + long-term semantic store
- 📊 **Tracing** — every agent call is logged for evaluation and debugging
- 🖥️ **Gradio UI** — interactive web interface

---

## 🏗️ Architecture

```
User ─► Multimodal Preprocessor ─► Language Detector ─► Input Guard
     ─► Orchestrator (+ Memory) ─► RAG Pipeline
     ─► Synthesis Agent ─► Output Guard ─► User
```

Every agent call is traced by the logger and surfaced in the UI.

---

## 📁 Project structure

```
ai_student_assistant/
├── app.py                    # Entry point — launches the UI
├── requirements.txt
├── .env.example              # Copy to .env and fill in your keys
├── src/
│   ├── core/                 # State schema, graph wiring, config
│   ├── agents/               # Orchestrator, synthesis agent
│   ├── memory/               # Short-term + long-term memory
│   ├── rag/                  # Ingestion, retrieval, reranking
│   ├── guards/               # Input guard (injection), output guard
│   ├── multimodal/           # OCR + vision captioning
│   ├── multilingual/         # Language detection + translation
│   ├── ui/                   # Gradio app
│   └── utils/                # Logging, tracing, prompt loading
├── prompts/
│   ├── en/                   # English prompt templates
│   └── ar/                   # Arabic prompt templates
├── scripts/
│   ├── build_index.py        # Ingest corpus into vector store
│   └── eval.py               # Retrieval + faithfulness evaluation
├── tests/                    # Unit and integration tests
├── data/
│   ├── corpus/               # Drop lecture PDFs here
│   ├── chroma/                ⚙ vector DB (generated, gitignored)
│   └── uploads/               ⚙ runtime image uploads (gitignored)
├── logs/                     # Trace logs (gitignored)
└── docs/                     # Additional documentation
```

---

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/ai_student_assistant.git
cd ai_student_assistant
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI or Groq API key
```

### 3. Add your corpus

Place lecture PDFs, slides, and notes in `data/corpus/`. A few sample PDFs are included.

### 4. Build the vector index

```bash
python scripts/build_index.py
```

### 5. Run the app

```bash
python app.py
```

Open <http://localhost:7860> in your browser.

---

## 🧪 Development

### Run tests

```bash
pytest tests/
```

### Evaluation

```bash
python scripts/eval.py
```

Produces `logs/evaluation_report.json` with retrieval recall@5, latency, and faithfulness scores.

---

## ⚙️ Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` / `GROQ_API_KEY` | LLM provider key (pick one) | `sk-...` / `gsk_...` |
| `LLM_MODEL` | Chat model name | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Sentence-Transformers model | `BAAI/bge-m3` |
| `RERANKER_MODEL` | Reranker model | `BAAI/bge-reranker-v2-m3` |
| `RETRIEVE_K` / `RERANK_K` | Retrieval candidates / final top-k | `10` / `5` |
| `INJECTION_THRESHOLD` | Confidence cutoff for the input guard | `0.7` |

---

## 👥 Team responsibilities

| Member | Owns | Files |
|---|---|---|
| 1 | Orchestrator + graph wiring | `src/core/graph.py`, `src/agents/orchestrator.py` |
| 2 | Tracing + inter-agent contracts | `src/utils/tracing.py`, `src/core/state.py` |
| 3 | Short-term memory | `src/memory/short_term.py` |
| 4 | Long-term memory | `src/memory/long_term.py` |
| 5 | Ingestion + indexing | `src/rag/ingestion.py`, `scripts/build_index.py` |
| 6 | Retrieval quality | `src/rag/retriever.py` |
| 7 | Grounding + citations | `src/agents/synthesis.py`, `prompts/` |
| 8 | Multimodal (Bonus A) | `src/multimodal/` |
| 9 | Input guard (Bonus C) | `src/guards/input_guard.py` |
| 10 | Multilingual (B) + output guard (D) | `src/multilingual/`, `src/guards/output_guard.py` |
| 11 | UI + evaluation + report | `src/ui/`, `scripts/eval.py` |

---

## 📝 License

Released under the [MIT License](LICENSE).

Academic project — Faculty of Artificial Intelligence, Deep Generative Models course.
