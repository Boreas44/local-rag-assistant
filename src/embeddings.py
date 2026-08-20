"""
embeddings.py — Generate text embeddings via Microsoft Foundry Local SDK.

Uses the Foundry Local SDK (in-process) to download, load, and generate
embeddings directly on-device — no external server, no CLI, no API keys.
"""

import threading
from typing import Optional

# pyrefly: ignore [missing-import]
from foundry_local_sdk import Configuration, FoundryLocalManager
from src.config import APP_NAME, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Singleton manager & synchronization
# ---------------------------------------------------------------------------
_manager: Optional[FoundryLocalManager] = None
_init_lock = threading.Lock()

def _resolve_model(manager: FoundryLocalManager, name: str):
    """Resolve a model by alias or explicit ID."""
    m = manager.catalog.get_model(name)
    if not m:
        m = manager.catalog.get_model_variant(name)
    return m



def get_foundry_manager() -> FoundryLocalManager:
    """Return or initialize the singleton FoundryLocalManager instance."""
    global _manager

    if _manager is not None:
        return _manager

    with _init_lock:
        if _manager is not None:
            return _manager

        if FoundryLocalManager.instance is not None:
            _manager = FoundryLocalManager.instance
            return _manager

        print("[models] Initializing Microsoft Foundry Local SDK…")
        config = Configuration(app_name=APP_NAME)
        FoundryLocalManager.initialize(config)
        _manager = FoundryLocalManager.instance

        if _manager is None:
            raise RuntimeError("Failed to initialize Microsoft Foundry Local SDK.")

        return _manager


def ensure_model_loaded(model_name: str) -> None:
    """Ensure a specific model is downloaded into cache and loaded into memory."""
    manager = get_foundry_manager()
    if not manager or not manager.catalog:
        raise RuntimeError("Foundry Local catalog is not available.")

    m = _resolve_model(manager, model_name)
    if not m:
        raise ValueError(f"Model '{model_name}' not found in Foundry Local catalog.")

    # 1. Download if not yet cached
    if not m.is_cached:
        print(f"[models] Model '{model_name}' not cached locally. Starting download (~first time only)...")
        def _progress(percent: float) -> None:
            print(f"     > Downloading '{model_name}': {percent:.1f}%", end="\r", flush=True)

        m.download(progress_callback=_progress)
        print(f"\n[models] Download completed for '{model_name}'.")

    # 2. Load into memory if not loaded
    if not m.is_loaded:
        print(f"[models] Loading model '{model_name}' into memory…")
        m.load()
        print(f"[models] Model '{model_name}' ready.")


# ---------------------------------------------------------------------------
# Public Embedding API
# ---------------------------------------------------------------------------
def generate_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Generate a single embedding vector for *text*."""
    ensure_model_loaded(model)
    manager = get_foundry_manager()
    m = _resolve_model(manager, model)
    client = m.get_embedding_client()
    response = client.generate_embedding(text)
    return response.data[0].embedding


def generate_embeddings_batch(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 4,
) -> list[list[float]]:
    """Generate embeddings for a list of texts in sub-batches."""
    if not texts:
        return []

    ensure_model_loaded(model)
    manager = get_foundry_manager()
    m = _resolve_model(manager, model)
    client = m.get_embedding_client()

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(texts), batch_size), 1):
        sub_batch = texts[i : i + batch_size]
        if total_batches > 1:
            print(
                f"     * Embedding batch {batch_idx}/{total_batches} "
                f"({min(i + batch_size, len(texts))}/{len(texts)} chunks)..."
            )
        response = client.generate_embeddings(sub_batch)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        all_embeddings.extend([d.embedding for d in sorted_data])

    return all_embeddings


# ---------------------------------------------------------------------------
# Compatibility Helpers
# ---------------------------------------------------------------------------
def get_openai_client():
    """Compatibility stub if needed by legacy callers."""
    return None


def get_endpoint() -> str:
    """Return in-process status."""
    return "in-process"
