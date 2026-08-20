"""
config.py — Central configuration for the Local RAG Assistant.

All tuneable constants live here so they can be adjusted in one place
without touching business logic.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (resolved relative to the repository root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "rag.db"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# ---------------------------------------------------------------------------
# Foundry Local — model aliases
# ---------------------------------------------------------------------------
APP_NAME = "local-rag-assistant"
EMBEDDING_MODEL = "qwen3-embedding-0.6b-generic-cpu:1"
CHAT_MODEL = "qwen2.5-1.5b-instruct-generic-cpu:4"

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
TOP_K = 5               # number of context chunks to retrieve

# ---------------------------------------------------------------------------
# Supported file extensions for ingestion
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
