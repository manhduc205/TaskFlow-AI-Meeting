"""
main_poc.py — CLI Test Tool (Phase 5)
Dùng để test toàn bộ pipeline từ command line trước khi dùng FastAPI.

Usage:
  # Chạy phân tích toàn bộ (analyze mode)
  python main_poc.py --mode analyze --meeting_id hop001 --transcript transcript/hopAI.txt

  # Chạy chat mode
  python main_poc.py --mode chat --meeting_id hop001 --query "Những task nào liên quan đến AI?"

  # Chạy demo nhanh (dùng file hopAI.txt mặc định)
  python main_poc.py
"""

import os
import sys
import json
import argparse
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_RESULTS_DIR = os.path.join(BASE_DIR, "summary_results")


def run_analyze(meeting_id: str, transcript_path: str, force: bool = False):
    """Pipeline phân tích đầy đủ."""
    from rag_engine import RAGEngine
    from llm_service import LLMService

    print("=" * 65)
    print("🚀  TASKFLOW AI — MEETING ANALYSIS PIPELINE v2.0")
    print("=" * 65)

    rag = RAGEngine(persist_directory=os.path.join(BASE_DIR, "chroma_db"))
    llm = LLMService()

    # Step 1: Ingest
    print(f"\n📥  [1/3] Ingesting transcript: {transcript_path}")
    chunks = rag.ingest(
        meeting_id=meeting_id,
        transcript_path=transcript_path,
        force_reingest=force,
    )
    if chunks:
        print(f"✅  Đã tạo {len(chunks)} chunks.")
    else:
        print("⚡  Meeting đã có trong DB, dùng dữ liệu cũ.")

    # Step 2: Lấy tất cả chunks để phân tích
    print(f"\n🔍  [2/3] Đang lấy toàn bộ chunks...")
    all_chunks = rag.get_all_chunks(meeting_id)
    print(f"✅  {len(all_chunks)} chunks sẵn sàng.")

    # Step 3: LLM phân tích
    print(f"\n🧠  [3/3] Gemini đang phân tích...")
    result = llm.analyze_meeting(chunks=all_chunks, meeting_id=meeting_id)

    # In báo cáo
    print("\n" + "─" * 65)
    print("📝  BÁO CÁO PHÂN TÍCH CUỘC HỌP")
    print("─" * 65)

    print("\n[TÓM TẮT]:")
    print(result.get("summary", "N/A"))

    print("\n[ĐIỂM KỸ THUẬT CHÍNH]:")
    for pt in result.get("key_technical_points", []):
        print(f"  • {pt}")

    print("\n[QUYẾT ĐỊNH]:")
    for dec in result.get("decisions", []):
        print(f"  ✓ {dec}")

    print(f"\n[DANH SÁCH CÔNG VIỆC] ({len(result.get('tasks', []))} tasks):")
    for i, task in enumerate(result.get("tasks", []), 1):
        print(f"  {i}. {task.get('task_name', 'N/A')}")
        print(f"     👤 {task.get('assignee', 'Chưa rõ')}  |  ⏰ {task.get('deadline', 'Không có')}  |  🔥 {task.get('priority', 'N/A')}")

    print("\n" + "─" * 65)

    # Lưu JSON
    os.makedirs(SUMMARY_RESULTS_DIR, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", meeting_id).strip("_") or "meeting"
    out_path = os.path.join(SUMMARY_RESULTS_DIR, f"{safe_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"💾  Đã xuất kết quả: {out_path}")

    return result


def run_chat(meeting_id: str, query: str):
    """Pipeline chat với meeting đã ingest."""
    from rag_engine import RAGEngine
    from llm_service import LLMService

    print("=" * 65)
    print("💬  TASKFLOW AI — CHAT MODE")
    print("=" * 65)

    rag = RAGEngine(persist_directory=os.path.join(BASE_DIR, "chroma_db"))
    llm = LLMService()

    if not rag.meeting_exists(meeting_id):
        print(f"[!] Meeting '{meeting_id}' chưa được ingest. Hãy chạy --mode analyze trước.")
        return

    print(f"\n❓  Query: {query}")
    print("🔍  Đang tìm context liên quan...")

    context_chunks = rag.hybrid_retrieve(query=query, meeting_id=meeting_id)
    print(f"✅  Tìm thấy {len(context_chunks)} chunks liên quan.")

    print("\n🧠  AI đang trả lời...")
    answer = llm.chat(query=query, context_chunks=context_chunks)

    print("\n" + "─" * 65)
    print("🤖  TRẢ LỜI:")
    print(answer)
    print("─" * 65)

    return answer


def run_interactive_chat(meeting_id: str):
    """Chat mode tương tác (multi-turn)."""
    from rag_engine import RAGEngine
    from llm_service import LLMService

    rag = RAGEngine(persist_directory=os.path.join(BASE_DIR, "chroma_db"))
    llm = LLMService()

    if not rag.meeting_exists(meeting_id):
        print(f"[!] Meeting '{meeting_id}' chưa được ingest.")
        return

    print("=" * 65)
    print(f"💬  CHAT VỚI MEETING '{meeting_id}' (gõ 'quit' để thoát)")
    print("=" * 65)

    history = []
    while True:
        try:
            query = input("\n❓  Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Thoát chat mode.")
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        context_chunks = rag.hybrid_retrieve(query=query, meeting_id=meeting_id)
        answer = llm.chat(
            query=query,
            context_chunks=context_chunks,
            conversation_history=history,
        )

        print(f"\n🤖  AI: {answer}")

        history.append({"role": "user", "text": query})
        history.append({"role": "model", "text": answer})


def main():
    parser = argparse.ArgumentParser(description="TaskFlow AI — Meeting Analysis CLI")
    parser.add_argument(
        "--mode",
        choices=["analyze", "chat", "interactive"],
        default="analyze",
        help="Mode chạy: analyze (phân tích) | chat (hỏi đáp 1 lần) | interactive (chat nhiều lượt)",
    )
    parser.add_argument("--meeting_id", default="demo_meeting", help="ID cuộc họp")
    parser.add_argument(
        "--transcript",
        default=os.path.join(BASE_DIR, "transcript", "rag.txt"),
        help="Đường dẫn file transcript (.txt hoặc .json Whisper segments)",
    )
    parser.add_argument("--query", default="", help="Câu hỏi (dùng cho --mode chat)")
    parser.add_argument("--force", action="store_true", help="Force re-ingest dù đã có dữ liệu")

    args = parser.parse_args()

    if args.mode == "analyze":
        run_analyze(
            meeting_id=args.meeting_id,
            transcript_path=args.transcript,
            force=args.force,
        )
    elif args.mode == "chat":
        if not args.query:
            print("[!] --mode chat cần truyền --query 'câu hỏi của bạn'")
            sys.exit(1)
        run_chat(meeting_id=args.meeting_id, query=args.query)
    elif args.mode == "interactive":
        run_interactive_chat(meeting_id=args.meeting_id)


if __name__ == "__main__":
    main()