# Local RAG Assistant

Offline Q&A assistant powered by **Microsoft Foundry Local** — all inference runs entirely on your machine with zero internet dependency.

## Architecture

```
Documents (.txt/.md/.pdf)
    │
    ▼
┌──────────┐    ┌──────────────────┐    ┌─────────────┐
│  Chunker │───▶│ Foundry Local    │───▶│   SQLite     │
│          │    │ Embedding Model  │    │ Vector Store │
└──────────┘    └──────────────────┘    └──────┬──────┘
                                               │
User Query ──▶ Embed Query ──▶ Cosine Search ──┘
                                    │
                              Top-K Chunks
                                    │
                                    ▼
                          ┌──────────────────┐
                          │ Foundry Local    │──▶ 💬 Answer
                          │ Chat Model      │
                          └──────────────────┘
```

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Windows** (for WinML acceleration) or any OS with CPU/GPU support

### 2. Install

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The first time you run an embedding or chat command, Foundry Local will download the required models (~500 MB each). This is a one-time operation.

### 3. Add Documents

Drop your `.txt`, `.md`, or `.pdf` files into the `documents/` folder.

### 4. Ingest

```bash
python main.py ingest
```

### 5. Ask Questions

```bash
# Single question
python main.py ask "What is this document about?"

# Interactive chat
python main.py chat
```

### 6. Web UI (Gradio)

```bash
python app.py
# Opens http://localhost:7860
```

## CLI Reference

| Command | Description |
|---|---|
| `python main.py ingest` | Ingest all documents from `./documents/` |
| `python main.py ingest -f` | Force re-ingest (overwrites existing) |
| `python main.py ingest -d /path/to/docs` | Ingest from a custom directory |
| `python main.py ask "question"` | Ask a single question |
| `python main.py ask "question" -k 10` | Retrieve 10 context chunks |
| `python main.py chat` | Interactive chat REPL |
| `python main.py status` | Show database statistics |
| `python main.py clear` | Clear all ingested data |

## Configuration

All settings are in [`src/config.py`](src/config.py):

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `qwen3-embedding-0.6b` | Foundry Local embedding model |
| `CHAT_MODEL` | `phi-3.5-mini` | Foundry Local chat model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Context chunks retrieved per query |

## Project Structure

```
local-rag-assistant/
├── documents/           # Drop your files here
├── data/
│   └── rag.db           # SQLite database (auto-created)
├── src/
│   ├── config.py        # Configuration constants
│   ├── database.py      # SQLite vector store
│   ├── embeddings.py    # Embedding generation (Foundry Local)
│   ├── ingestion.py     # Document loading & chunking
│   ├── retrieval.py     # Cosine similarity search
│   └── generation.py    # LLM answer generation
├── main.py              # CLI entry point
├── app.py               # Gradio web UI
├── requirements.txt
└── README.md
```

## Privacy

All data stays on your machine:
- Models are downloaded once and cached locally by Foundry Local
- No API keys or cloud accounts required
- No telemetry or data exfiltration
