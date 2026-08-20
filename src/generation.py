"""
generation.py — LLM answer generation via Microsoft Foundry Local SDK.

Takes the user's question plus retrieved context chunks and produces a
grounded, source-cited answer using the local chat model in-process.
"""

from typing import Generator

from src.config import CHAT_MODEL
from src.embeddings import get_foundry_manager, ensure_model_loaded, _resolve_model


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Sen yalnızca sağlanan dökümanlara dayanarak soruları yanıtlayan bir asistansın.

KURALLAR:
1. Sadece aşağıda verilen bağlam (context) metnini kullanarak cevap ver. Dışarıdan veya kendi bilginden hiçbir şey ekleme.
2. Eğer sorunun cevabı aşağıdaki metinlerde hiç geçmiyorsa, sadece "Dökümanlarda bu konu hakkında yeterli bilgi bulunamadı." de.
3. Mümkün olduğunda cevabı verirken hangi dökümandan (kaynak) aldığını belirt.
4. Cevapların net, anlaşılır ve Türkçe olsun.

──── BAĞLAM (CONTEXT) ────
{context}
──── BAĞLAM SONU ────
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "unknown")
        score = chunk.get("score", 0)
        text = chunk["content"]
        parts.append(
            f"[{i}] (source: {source}, relevance: {score})\n{text}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_answer_stream(
    query: str,
    context_chunks: list[dict],
    model: str = CHAT_MODEL,
) -> Generator[str, None, None]:
    """Yield answer tokens as they arrive (streaming).

    Falls back to non-streaming if the SDK streaming hangs in threaded
    contexts (e.g. Gradio worker threads).
    """
    # Streaming can hang in Gradio worker threads.
    # Fallback to non-streaming.
    yield generate_answer(query, context_chunks, model)


def generate_answer(
    query: str,
    context_chunks: list[dict],
    model: str = CHAT_MODEL,
) -> str:
    """Generate a complete (non-streaming) answer using complete_chat."""
    ensure_model_loaded(model)
    manager = get_foundry_manager()
    m = _resolve_model(manager, model)
    client = m.get_chat_client()
    client.settings.temperature = 0.3
    client.settings.max_tokens = 512

    context = _build_context_block(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]

    response = client.complete_chat(messages=messages)
    if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content or ""
    return ""

