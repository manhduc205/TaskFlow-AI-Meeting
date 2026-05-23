"""
test_youtube.py — Công cụ CLI test độc lập module YouTube và RAG
Chạy trực tiếp trên Terminal để kiểm tra bóc tách và ingestion vào ChromaDB.

Cách chạy:
  python test_youtube.py --url "https://www.youtube.com/watch?v=xxx" --meeting_id "yt_test_01"
"""

import os
import argparse
import sys

# Khởi tạo đường dẫn cơ sở
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đảm bảo repo root nằm trong sys.path để import đúng thư mục utils/ và core/
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser(description="TaskFlow AI — YouTube Local Test Tool")
    parser.add_argument(
        "--url",
        required=True,
        help="Đường dẫn URL của video YouTube cần test tóm tắt"
    )
    parser.add_argument(
        "--meeting_id",
        default="youtube_local_test",
        help="ID định danh cuộc họp/video để lưu vào Database"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ép buộc xóa dữ liệu cũ và nạp lại từ đầu (force_reingest)"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("📺  TASKFLOW AI — YOUTUBE LOCAL INGESTION TEST")
    print("=" * 65)

    # 1. Kiểm tra môi trường và import các module nội bộ
    try:
        from utils.ytb_text_utils import get_youtube_transcript_as_text
        from core.rag_engine import RAGEngine
    except ImportError as e:
        print(f"[!] Lỗi Import: {e}")
        print("[!] Đảm bảo các file ytb_text_utils.py và rag_engine.py nằm đúng vị trí.")
        sys.exit(1)

    # 2. Thực hiện cào phụ đề từ YouTube
    print(f"\n📥  [Bước 1/3] Đang cào transcript từ YouTube URL:")
    print(f"    🔗 {args.url}")

    try:
        youtube_text_content = get_youtube_transcript_as_text(args.url)
        word_count = len(youtube_text_content.split())
        print(f"✅  Cào thành công! Trích xuất được sơ bộ ~{word_count} từ văn bản thô.")
    except Exception as e:
        print(f"[!] Thất bại ở bước cào phụ đề YouTube: {e}")
        print("[!] Gợi ý: Kiểm tra lại mạng hoặc video có thực sự hỗ trợ phụ đề (Vi/En) không.")
        sys.exit(0)

    # 3. Khởi tạo kho lưu trữ dữ liệu RAG Engine tại chỗ
    print(f"\n💾  [Bước 2/3] Khởi tạo RAGEngine local...")
    chroma_path = os.path.join(BASE_DIR, "chroma_db")
    rag = RAGEngine(persist_directory=chroma_path)

    # 4. Thực hiện chia nhỏ ngữ nghĩa và nạp dữ liệu (Ingestion)
    print(f"\n🧠  [Bước 3/3] Tiến hành Semantic Chunking và Vectorizing dữ liệu...")
    try:
        chunks = rag.ingest(
            meeting_id=args.meeting_id,
            transcript_path=None,
            segments_json_path=None,
            raw_text=youtube_text_content,
            force_reingest=args.force,
        )

        print("-" * 65)
        if chunks:
            print(f"🎉  THÀNH CÔNG RỰC RỠ!")
            print(f"✅  Hệ thống đã tự động bóc tách thành: {len(chunks)} chunks.")
            print(f"✅  Dữ liệu đã được lưu trữ song song vào:")
            print(f"    1. Kho Vector Similarity (ChromaDB) -> {chroma_path}")
            print(f"    2. Kho Từ khóa Tần suất (BM25 Okapi)")
            print(f"    3. Master Metadata Cache")

            # Preview nhanh 2 chunk đầu tiên ra màn hình
            print(f"\n📝  [Preview] Nội dung 2 Chunks đầu tiên:")
            for mc in chunks[:2]:
                print(f"  • [{mc.chunk_index}] ({mc.start_time:.1f}s → {mc.end_time:.1f}s): {mc.raw_text[:120]}...")
        else:
            print("⚡  Thông báo: Video này đã tồn tại trong DB, hệ thống sử dụng dữ liệu cũ.")
            print("💡  Gợi ý: Thêm flag `--force` nếu bạn muốn xóa dữ liệu cũ và ép nạp lại.")

    except Exception as e:
        print(f"[!] Thất bại khi nạp dữ liệu vào RAG Engine: {e}")

    print("=" * 65)


if __name__ == "__main__":
    main()