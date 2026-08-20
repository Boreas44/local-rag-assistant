"""
main.py — CLI entry point for the Local RAG Assistant.

Usage:
    python main.py ingest           Ingest documents from ./documents/
    python main.py ask "question"   Ask a single question
    python main.py chat             Interactive chat loop
    python main.py status           Show database statistics
    python main.py clear            Clear all ingested data
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows consoles (avoids UnicodeEncodeError for emoji)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback (shouldn't happen)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents from the configured directory."""
    from src.ingestion import ingest_directory

    dir_path = Path(args.directory) if args.directory else None
    ingest_directory(dir_path, force=args.force)


def cmd_ask(args: argparse.Namespace) -> None:
    """Answer a single question."""
    from src.retrieval import search
    from src.generation import generate_answer

    query = " ".join(args.question)
    if not query.strip():
        print("Please provide a question.")
        sys.exit(1)

    print(f"\n🔍 Searching for relevant context…\n")
    results = search(query, top_k=args.top_k)

    if not results:
        print("No documents found in the database. Run 'python main.py ingest' first.")
        sys.exit(1)

    # Show retrieved context
    print("─" * 60)
    print(f"📚 Retrieved {len(results)} chunk(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['source_file']} (chunk #{r['chunk_index']}, "
              f"score: {r['score']})")
        preview = r["content"][:120].replace("\n", " ")
        print(f"      {preview}…\n")

    # Generate answer
    print("─" * 60)
    print("💬 Generating answer…\n")
    answer = generate_answer(query, results)
    print(answer)
    print()


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat REPL."""
    from src.retrieval import search
    from src.generation import generate_answer_stream

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Local RAG Assistant — Interactive Chat          ║")
    print("║  Type your question and press Enter.                    ║")
    print("║  Type 'quit' or 'exit' to leave.                       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        # Retrieve
        results = search(query, top_k=args.top_k)
        if not results:
            print("\nAssistant: No documents in the database yet. "
                  "Run 'python main.py ingest' first.\n")
            continue

        # Stream answer
        print("\nAssistant: ", end="", flush=True)
        for token in generate_answer_stream(query, results):
            print(token, end="", flush=True)
        print("\n")

        # Show sources
        sources = set(r["source_file"] for r in results)
        print(f"  📎 Sources: {', '.join(sources)}\n")


def cmd_status(args: argparse.Namespace) -> None:
    """Show database statistics."""
    from src.config import DB_PATH
    from src.database import VectorDatabase

    db = VectorDatabase(DB_PATH)
    stats = db.get_stats()
    db.close()

    print("\n📊 Database Status")
    print("─" * 40)
    print(f"  Database path : {stats['db_path']}")
    print(f"  Total files   : {stats['total_files']}")
    print(f"  Total chunks  : {stats['total_chunks']}")
    print()


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear all ingested data."""
    from src.config import DB_PATH
    from src.database import VectorDatabase

    if not args.yes:
        confirm = input("⚠  This will delete ALL ingested data. Continue? [y/N] ")
        if confirm.lower() not in {"y", "yes"}:
            print("Cancelled.")
            return

    db = VectorDatabase(DB_PATH)
    db.clear_database()
    db.close()
    print("✅ Database cleared.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="local-rag-assistant",
        description="Local RAG Assistant — Offline Q&A powered by Foundry Local",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument(
        "-d", "--directory",
        help="Path to documents directory (default: ./documents/)",
    )
    p_ingest.add_argument(
        "-f", "--force",
        action="store_true",
        help="Re-ingest files even if already in the database",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # --- ask ---
    p_ask = subparsers.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", nargs="+", help="The question to ask")
    p_ask.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (default: 5)",
    )
    p_ask.set_defaults(func=cmd_ask)

    # --- chat ---
    p_chat = subparsers.add_parser("chat", help="Interactive chat loop")
    p_chat.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (default: 5)",
    )
    p_chat.set_defaults(func=cmd_chat)

    # --- status ---
    p_status = subparsers.add_parser("status", help="Show database stats")
    p_status.set_defaults(func=cmd_status)

    # --- clear ---
    p_clear = subparsers.add_parser("clear", help="Clear all ingested data")
    p_clear.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    p_clear.set_defaults(func=cmd_clear)

    # --- parse & dispatch ---
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
