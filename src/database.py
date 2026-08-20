"""
database.py — SQLite vector store for the Local RAG Assistant.

Embeddings are stored as float32 BLOBs.  Similarity search is performed
in Python (NumPy) after fetching all vectors — this is efficient for
datasets up to ~100 k chunks and avoids any native-extension dependency.
"""

import sqlite3
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class VectorDatabase:
    """Lightweight SQLite wrapper for storing and querying document embeddings."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy, reusable connection (one per VectorDatabase instance)."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create the documents table if it doesn't exist yet."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL,
                content     TEXT    NOT NULL,
                embedding   BLOB    NOT NULL,
                created_at  TEXT    NOT NULL,
                UNIQUE(source_file, chunk_index)
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_file ON documents(source_file)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def store_chunk(
        self,
        source_file: str,
        chunk_index: int,
        content: str,
        embedding: list[float] | np.ndarray,
    ) -> None:
        """Persist a single chunk with its embedding vector."""
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (source_file, chunk_index, content, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_file, chunk_index, content, blob, now),
        )
        self.conn.commit()

    def store_chunks_batch(
        self,
        source_file: str,
        chunks: list[str],
        embeddings: list[list[float]] | list[np.ndarray],
    ) -> None:
        """Persist multiple chunks in a single transaction (much faster)."""
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                source_file,
                idx,
                chunk,
                np.asarray(emb, dtype=np.float32).tobytes(),
                now,
            )
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO documents
                (source_file, chunk_index, content, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_all_embeddings(
        self,
        source_files: list[str] | None = None,
    ) -> list[tuple[int, str, str, int, np.ndarray]]:
        """Return (id, source_file, content, chunk_index, embedding) for every row.

        If *source_files* is given, only rows matching those filenames are returned.
        """
        if source_files:
            placeholders = ",".join("?" for _ in source_files)
            cursor = self.conn.execute(
                f"SELECT id, source_file, content, chunk_index, embedding "
                f"FROM documents WHERE source_file IN ({placeholders})",
                source_files,
            )
        else:
            cursor = self.conn.execute(
                "SELECT id, source_file, content, chunk_index, embedding FROM documents"
            )
        results = []
        for row_id, src, content, cidx, blob in cursor:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            results.append((row_id, src, content, cidx, vec))
        return results

    def get_ingested_files(self) -> list[str]:
        """Return a sorted list of distinct source filenames in the database."""
        cursor = self.conn.execute(
            "SELECT DISTINCT source_file FROM documents ORDER BY source_file"
        )
        return [row[0] for row in cursor]

    def file_already_ingested(self, filename: str) -> bool:
        """Check whether a file has already been ingested."""
        cursor = self.conn.execute(
            "SELECT 1 FROM documents WHERE source_file = ? LIMIT 1",
            (filename,),
        )
        return cursor.fetchone() is not None

    def delete_file(self, filename: str) -> int:
        """Remove all chunks for a given source file. Returns rows deleted."""
        cursor = self.conn.execute(
            "DELETE FROM documents WHERE source_file = ?", (filename,)
        )
        self.conn.commit()
        return cursor.rowcount

    def clear_database(self) -> None:
        """Delete **all** stored chunks (irreversible)."""
        self.conn.execute("DELETE FROM documents")
        self.conn.execute("VACUUM")
        self.conn.commit()

    def get_stats(self) -> dict:
        """Return summary statistics about the vector store."""
        total_chunks = self.conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        total_files = self.conn.execute(
            "SELECT COUNT(DISTINCT source_file) FROM documents"
        ).fetchone()[0]
        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "db_path": str(self.db_path),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
