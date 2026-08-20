"""
retrieval.py — Cosine-similarity search over the SQLite vector store.

Embeds the user's query with Foundry Local, then performs a brute-force
scan of all stored embeddings and returns the top-K most similar chunks.
"""

import numpy as np

from src.config import DB_PATH, TOP_K
from src.database import VectorDatabase
from src.embeddings import generate_embedding


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(
    query: str,
    top_k: int = TOP_K,
    db: VectorDatabase | None = None,
    source_files: list[str] | None = None,
) -> list[dict]:
    """
    Embed *query* and return the *top_k* most similar chunks.

    If *source_files* is provided, only chunks from those files are searched.

    Each result dict contains:
        - content:      the chunk text
        - score:        cosine similarity (0–1)
        - source_file:  originating filename
        - chunk_index:  position within that file
    """
    own_db = db is None
    if own_db:
        db = VectorDatabase(DB_PATH)

    try:
        # 1. Embed the query
        query_vec = np.asarray(generate_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # 2. Fetch stored embeddings (optionally filtered by source file)
        rows = db.get_all_embeddings(source_files=source_files)
        if not rows:
            return []

        # 3. Vectorized cosine similarity
        matrix = np.array([r[4] for r in rows], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix_normed = matrix / norms

        scores = matrix_normed @ query_vec

        # 4. Top-K sorting
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row_id, src, content, cidx, _ = rows[idx]
            results.append({
                "id": row_id,
                "content": content,
                "score": round(float(scores[idx]), 4),
                "source_file": src,
                "chunk_index": cidx,
            })
        return results

    finally:
        if own_db:
            db.close()
