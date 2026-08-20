"""
ingestion.py — Document loading, chunking, and embedding pipeline.

Supports .txt, .md, and .pdf files.  Each file is read, split into
overlapping character-based chunks, embedded via Foundry Local, and
persisted in the SQLite vector store.
"""

from pathlib import Path

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
)
from src.database import VectorDatabase
from src.embeddings import generate_embeddings_batch


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------
def load_text_file(path: Path) -> str:
    """Read a plain-text or Markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF ingestion.  "
            "Install it with:  pip install PyMuPDF"
        )
    text_parts: list[str] = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def load_file(path: Path) -> str:
    """Dispatch to the correct reader based on file extension."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return load_text_file(path)
    elif ext == ".pdf":
        return load_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping chunks of roughly *chunk_size* characters.

    The splitter tries to break at paragraph or sentence boundaries when
    possible so that chunks remain semantically coherent.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    min_step = max(50, chunk_size - overlap)

    while start < len(text):
        if len(text) - start <= chunk_size:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        end = start + chunk_size
        # Search for a natural boundary only in the upper portion of the chunk (60% to 100%)
        search_start = start + int(chunk_size * 0.6)

        break_pos = -1
        # 1. Try paragraph break
        pos = text.rfind("\n\n", search_start, end)
        if pos != -1:
            break_pos = pos + 2
        else:
            # 2. Try sentence break
            pos = text.rfind(". ", search_start, end)
            if pos != -1:
                break_pos = pos + 2
            else:
                # 3. Try newline
                pos = text.rfind("\n", search_start, end)
                if pos != -1:
                    break_pos = pos + 1
                else:
                    # 4. Try space
                    pos = text.rfind(" ", search_start, end)
                    if pos != -1:
                        break_pos = pos + 1

        if break_pos != -1:
            end = break_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance ensuring forward progress
        next_start = end - overlap
        if next_start <= start:
            next_start = start + min_step
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Ingestion orchestration
# ---------------------------------------------------------------------------
def ingest_file(
    path: Path,
    db: VectorDatabase | None = None,
    *,
    force: bool = False,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    Ingest a single file: read → chunk → embed → store.

    Returns the number of chunks created.
    """
    own_db = db is None
    if own_db:
        db = VectorDatabase(DB_PATH)

    try:
        filename = path.name

        if not force and db.file_already_ingested(filename):
            print(f"  >  Skipping (already ingested): {filename}")
            return 0

        # If re-ingesting, remove old data first
        if force:
            db.delete_file(filename)

        print(f"  - Reading: {filename}")
        text = load_file(path)
        if not text.strip():
            print(f"  !  Empty file, skipping: {filename}")
            return 0

        print(f"  *  Chunking ({chunk_size} chars, {overlap} overlap)...")
        chunks = chunk_text(text, chunk_size, overlap)
        print(f"     -> {len(chunks)} chunk(s)")

        print(f"  * Generating embeddings...")
        embeddings = generate_embeddings_batch(chunks)

        print(f"  * Storing in database...")
        db.store_chunks_batch(filename, chunks, embeddings)

        print(f"  + Done: {filename} ({len(chunks)} chunks)")
        return len(chunks)

    finally:
        if own_db:
            db.close()


def ingest_directory(
    dir_path: Path | None = None,
    *,
    force: bool = False,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict:
    """
    Walk *dir_path* and ingest every supported file.

    Returns a summary dict with counts of files processed and chunks created.
    """
    if dir_path is None:
        dir_path = DOCUMENTS_DIR

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {dir_path}")

    db = VectorDatabase(DB_PATH)
    files_processed = 0
    files_skipped = 0
    total_chunks = 0

    try:
        supported_files = sorted(
            f
            for f in dir_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not supported_files:
            print(f"No supported files found in {dir_path}")
            return {
                "files_processed": 0,
                "files_skipped": 0,
                "total_chunks": 0,
            }

        print(f"Found {len(supported_files)} file(s) to process\n")

        for filepath in supported_files:
            chunks = ingest_file(
                filepath,
                db,
                force=force,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if chunks > 0:
                files_processed += 1
                total_chunks += chunks
            else:
                files_skipped += 1
            print()  # blank line between files

    finally:
        db.close()

    summary = {
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "total_chunks": total_chunks,
    }
    print("─" * 50)
    print(f"Ingestion complete: {files_processed} file(s), {total_chunks} chunk(s)")
    if files_skipped:
        print(f"  ({files_skipped} file(s) skipped — already ingested)")
    return summary
