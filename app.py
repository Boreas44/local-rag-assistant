"""
app.py — Gradio web UI for the Local RAG Assistant.

Launch with:
    python app.py

Opens a browser at http://localhost:7860 with a chat interface,
document upload, and database status panel.
"""

import tempfile
import threading
import time
from pathlib import Path

import gradio as gr

from src.config import DB_PATH, DOCUMENTS_DIR
from src.database import VectorDatabase
from src.ingestion import ingest_file
from src.retrieval import search
from src.generation import generate_answer, generate_answer_stream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_status() -> str:
    """Return a formatted status string."""
    db = VectorDatabase(DB_PATH)
    stats = db.get_stats()
    db.close()
    return (
        f"📊 **Database Status**\n\n"
        f"- **Files ingested:** {stats['total_files']}\n"
        f"- **Total chunks:** {stats['total_chunks']}\n"
        f"- **DB path:** `{stats['db_path']}`"
    )


def _get_status_html() -> str:
    """Return a formatted status string as HTML."""
    db = VectorDatabase(DB_PATH)
    stats = db.get_stats()
    db.close()
    return (
        f"<div style='padding: 12px; background-color: rgba(79, 70, 229, 0.05); border: 1px solid rgba(79, 70, 229, 0.15); border-radius: 8px; font-size: 0.9rem; margin-top: 10px;'>"
        f"📊 <b>Veritabanı Durumu</b><br/>"
        f"• <b>İşlenen dosya sayısı:</b> {stats['total_files']}<br/>"
        f"• <b>Toplam vektör parçası:</b> {stats['total_chunks']}<br/>"
        f"• <b>Dosya yolu:</b> <code style='font-size: 0.85rem; color: #4f46e5;'>{stats['db_path']}</code>"
        f"</div>"
    )


def _get_ingested_files() -> list[str]:
    """Return the list of filenames that have been ingested into the DB."""
    db = VectorDatabase(DB_PATH)
    files = db.get_ingested_files()
    db.close()
    return files


def _format_sources(results: list[dict]) -> str:
    """Format retrieved chunks for the sources panel."""
    if not results:
        return "*No results found.*"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"**[{i}]** `{r['source_file']}` · chunk #{r['chunk_index']} · "
            f"score: {r['score']}\n\n"
            f"> {r['content'][:200]}{'…' if len(r['content']) > 200 else ''}\n"
        )
    return "\n---\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def chat_respond(message: str, history: list[dict], selected_docs: list[str]):
    """Streaming chat callback — filters retrieval to selected documents."""
    if not message.strip():
        yield history, ""
        return

    # Guard: no documents selected
    if not selected_docs:
        history = history + [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": (
                    "⚠️ **Lütfen önce sağ panelden bir veya daha fazla döküman seçin.**\n\n"
                    "Soru sorabilmek için aktif döküman seçmeniz gerekiyor."
                ),
            },
        ]
        yield history, ""
        return

    # Show a "thinking" indicator while retrieving context
    history = history + [{"role": "user", "content": message}]
    yield history + [{"role": "assistant", "content": "🔍 Dökümanlar aranıyor…"}], ""

    try:
        # Retrieve context — scoped to selected documents only
        # Reduced top_k to 2 to prevent local LLM timeout on large contexts
        results = search(message, top_k=2, source_files=selected_docs)
    except Exception as e:
        yield history + [
            {
                "role": "assistant",
                "content": (
                    f"⚠️ **Arama sırasında hata oluştu:**\n\n```\n{e}\n```\n\n"
                    "Lütfen Microsoft Foundry Local servisinin çalıştığından emin olun."
                ),
            },
        ], ""
        return

    if not results:
        yield history + [
            {
                "role": "assistant",
                "content": (
                    "Seçili dökümanlar veritabanında henüz mevcut değil.\n\n"
                    "Lütfen sağ panelden dosya yükleyip **Ingest** butonuna basın."
                ),
            },
        ], ""
        return

    # Show generating indicator
    yield history + [{"role": "assistant", "content": "🧠 Yanıt oluşturuluyor…"}], ""

    import queue
    import threading
    from src.generation import generate_answer_stream

    ans_queue = queue.Queue()

    def _run_gen():
        try:
            for token in generate_answer_stream(message, results):
                ans_queue.put({"type": "token", "data": token})
            ans_queue.put({"type": "done"})
        except Exception as e:
            ans_queue.put({"type": "error", "error": e})

    gen_thread = threading.Thread(target=_run_gen, daemon=True)
    gen_thread.start()

    dots = 0
    start_time = time.time()
    timeout_seconds = 600
    partial = ""

    while True:
        try:
            msg = ans_queue.get(timeout=0.2)
            if msg["type"] == "token":
                partial += msg["data"]
                yield history + [{"role": "assistant", "content": partial}], ""
            elif msg["type"] == "done":
                break
            elif msg["type"] == "error":
                partial += f"\n\n⚠️ **Yanıt üretilirken hata oluştu:**\n\n```\n{msg['error']}\n```"
                yield history + [{"role": "assistant", "content": partial}], ""
                break
        except queue.Empty:
            if time.time() - start_time > timeout_seconds:
                partial += (
                    "\n\n⚠️ **Yanıt üretme zaman aşımına uğradı (600s).** "
                    "Lütfen daha kısa bir soru deneyin."
                )
                yield history + [{"role": "assistant", "content": partial}], ""
                break
            
            if not partial:
                dots = (dots + 1) % 4
                thinking_msg = "🧠 Yanıt oluşturuluyor" + "." * dots
                yield history + [{"role": "assistant", "content": thinking_msg}], ""
            else:
                # If we already have partial text, just yield it to keep alive
                yield history + [{"role": "assistant", "content": partial}], ""

    # Append source citations
    sources = set(r["source_file"] for r in results)
    partial += f"\n\n---\n📎 *Kaynaklar: {', '.join(sources)}*"
    history = history + [{"role": "assistant", "content": partial}]
    yield history, ""


def upload_and_ingest(files):
    """Handle file uploads: copy to documents dir and ingest with progress."""
    if not files:
        yield "<div class='result-error'>❌ Lütfen yüklenecek dosyaları seçin.</div>", gr.update()
        return

    results: list[str] = []
    num_files = len(files)

    for idx, file in enumerate(files, 1):
        src = Path(file.name) if hasattr(file, "name") else Path(file)

        # 1. Copying phase
        yield "\n".join(results) + f"""<div class="loader-container">
          <div class="spinner"></div>
          <div>⏳ <b>[{idx}/{num_files}] {src.name}</b> hazırlanıyor...</div>
        </div>""", gr.update()

        # Copy to documents directory
        dest = DOCUMENTS_DIR / src.name
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())

        # Check if already ingested — skip to avoid redundant embedding computation
        _check_db = VectorDatabase(DB_PATH)
        already_ingested = _check_db.file_already_ingested(src.name)
        _check_db.close()

        if already_ingested:
            results.append(f"""<div class="result-success" style="border-left-color: #f59e0b; background-color: rgba(245,158,11,0.08);">
              ⚠️ <b>{src.name}</b> zaten veritabanında mevcut, tekrar işlenmedi.
              Üzerine yazmak için önce veritabanını temizleyin.
            </div>""")
            if idx < num_files:
                next_src = Path(files[idx].name) if hasattr(files[idx], "name") else Path(files[idx])
                yield "\n".join(results) + f"""<div class="loader-container">
                  <div class="spinner"></div>
                  <div>⏳ Sonraki dosya bekleniyor: <b>{next_src.name}</b>...</div>
                </div>""", gr.update()
            else:
                yield "\n".join(results), gr.update()
            continue

        # 2. Chunking phase
        yield "\n".join(results) + f"""<div class="loader-container">
          <div class="spinner"></div>
          <div>✂️ <b>[{idx}/{num_files}] {src.name}</b> parçalanıyor...</div>
        </div>""", gr.update()

        try:
            # 3. Embedding phase (can take a while)
            yield "\n".join(results) + f"""<div class="loader-container">
              <div class="spinner"></div>
              <div>🧠 <b>[{idx}/{num_files}] {src.name}</b> yerel vektörleri hesaplanıyor...<br/>
              <span style="font-size: 0.8rem; color: #6b7280;">(Yerel model çalıştırılıyor, bu işlem dosya boyutuna göre 1-3 dakika sürebilir)</span></div>
            </div>""", gr.update()

            chunks = ingest_file(dest, force=False)
            results.append(f"""<div class="result-success">
              ✅ <b>{src.name}</b> — {chunks} parça başarıyla veritabanına işlendi.
            </div>""")
        except Exception as e:
            results.append(f"""<div class="result-error">
              ❌ <b>{src.name}</b> işlenirken hata: {e}
            </div>""")

        # Yield current cumulative results plus the loader for the next file if there is one
        current_out = "\n".join(results)
        if idx < num_files:
            next_src = Path(files[idx].name) if hasattr(files[idx], "name") else Path(files[idx])
            current_out += f"""<div class="loader-container">
              <div class="spinner"></div>
              <div>⏳ Sonraki dosya bekleniyor: <b>{next_src.name}</b>...</div>
            </div>"""
        yield current_out, gr.update()

    # Finally yield results, database status, and update the doc list
    updated_files = _get_ingested_files()
    yield (
        "\n".join(results) + _get_status_html(),
        gr.update(choices=updated_files, value=updated_files),
    )


def refresh_status() -> str:
    return _get_status()


def clear_db():
    db = VectorDatabase(DB_PATH)
    db.clear_database()
    db.close()
    return "✅ Database cleared.\n\n" + _get_status(), gr.update(choices=[], value=[])


def select_all_docs():
    """Select all ingested documents."""
    files = _get_ingested_files()
    return gr.update(value=files)


def deselect_all_docs():
    """Deselect all documents."""
    return gr.update(value=[])


def refresh_doc_list():
    """Refresh the document list from the database."""
    files = _get_ingested_files()
    return gr.update(choices=files, value=files)


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
_custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
.status-box {
    font-size: 0.9rem;
    padding: 1rem;
}
@keyframes gradient-bg {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.animated-title {
    background: linear-gradient(-45deg, #4f46e5, #06b6d4, #3b82f6, #ec4899);
    background-size: 400% 400%;
    animation: gradient-bg 8s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.loader-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 8px;
    background-color: rgba(79, 70, 229, 0.08);
    border-left: 4px solid #4f46e5;
    margin-bottom: 10px;
    animation: pulse 2s infinite ease-in-out;
}
.spinner {
    border: 3px solid rgba(0, 0, 0, 0.05);
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border-left-color: #4f46e5;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0% { opacity: 0.9; }
    50% { opacity: 0.7; }
    100% { opacity: 0.9; }
}
.result-success {
    padding: 12px;
    background-color: rgba(16, 185, 129, 0.08);
    border-left: 4px solid #10b981;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 0.9rem;
}
.result-error {
    padding: 12px;
    background-color: rgba(239, 68, 68, 0.08);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 0.9rem;
}
.doc-selector-container {
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 8px;
    padding: 8px;
    background-color: rgba(79, 70, 229, 0.03);
}
.doc-selector-container label {
    font-weight: 600 !important;
}
"""


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    # Pre-load ingested file list
    initial_files = _get_ingested_files()

    with gr.Blocks(title="Local RAG Assistant") as app:

        gr.Markdown(
            "# 🧠 <span class='animated-title'>Local RAG Assistant</span>\n"
            "Offline Q&A powered by **Microsoft Foundry Local**. "
            "All inference runs on your machine — no cloud, no API keys.",
        )

        with gr.Row():
            # ── Left: Chat ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Sohbet",
                    height=520,
                    buttons=["copy"],
                    placeholder=(
                        "📄 Sağ panelden döküman seçin ve soru sorun.\n\n"
                        "Sadece yüklenen dökümanlar hakkında cevap verilir."
                    ),
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Dökümanınız hakkında bir soru sorun…",
                        label="Mesajınız",
                        scale=5,
                        show_label=False,
                        container=False,
                    )
                    send_btn = gr.Button("Gönder", variant="primary", scale=1)

                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Sohbeti Temizle", size="sm")

            # ── Right: Controls ──
            with gr.Column(scale=1):
                # ── Active Documents Selector ──
                gr.Markdown("### 📂 Aktif Dökümanlar")
                gr.Markdown(
                    "<small>Sorularınız yalnızca seçili dökümanlardan cevaplanır.</small>"
                )
                doc_selector = gr.CheckboxGroup(
                    choices=initial_files,
                    value=initial_files,
                    label="Döküman seçin",
                    elem_classes="doc-selector-container",
                )
                with gr.Row():
                    select_all_btn = gr.Button("✅ Tümü", size="sm", scale=1)
                    deselect_all_btn = gr.Button("❌ Hiçbiri", size="sm", scale=1)
                    refresh_docs_btn = gr.Button("🔄", size="sm", scale=0)

                gr.Markdown("---")

                # ── File Upload ──
                gr.Markdown("### 📁 Döküman Yükleme")
                upload = gr.File(
                    label="Dosya seçin",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".pdf"],
                    type="filepath",
                )
                ingest_btn = gr.Button("📥 Yükle ve İşle", variant="primary")
                ingest_output = gr.HTML(value="")

                gr.Markdown("---")
                gr.Markdown("### ℹ️ Durum")
                status_box = gr.Markdown(value=_get_status, elem_classes="status-box")
                refresh_btn = gr.Button("🔄 Yenile")

                gr.Markdown("---")
                clear_btn = gr.Button("🗑️ Veritabanını Temizle", variant="stop")
                clear_output = gr.Markdown()

        # ── Event bindings ──

        # Chat: send on Enter or button click
        msg_input.submit(
            fn=chat_respond,
            inputs=[msg_input, chatbot, doc_selector],
            outputs=[chatbot, msg_input],
        )
        send_btn.click(
            fn=chat_respond,
            inputs=[msg_input, chatbot, doc_selector],
            outputs=[chatbot, msg_input],
        )

        # Clear chat
        clear_chat_btn.click(fn=lambda: [], outputs=[chatbot])

        # Document selector controls
        select_all_btn.click(fn=select_all_docs, outputs=[doc_selector])
        deselect_all_btn.click(fn=deselect_all_docs, outputs=[doc_selector])
        refresh_docs_btn.click(fn=refresh_doc_list, outputs=[doc_selector])

        # File upload & ingest
        ingest_btn.click(
            fn=upload_and_ingest,
            inputs=[upload],
            outputs=[ingest_output, doc_selector],
        )

        # Status refresh
        refresh_btn.click(fn=refresh_status, outputs=[status_box])

        # Clear DB
        clear_btn.click(fn=clear_db, outputs=[clear_output, doc_selector])

    app.queue()
    return app


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_ui()
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            css=_custom_css,
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="slate",
            ),
            share=False,
            ssr_mode=False,
        )
    except OSError:
        app.launch(
            server_name="127.0.0.1",
            css=_custom_css,
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="slate",
            ),
            share=False,
            ssr_mode=False,
        )
