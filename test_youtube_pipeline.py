"""
test_youtube_pipeline.py — End-to-End CLI Tool cho YouTube Video
Tính năng: Cào dữ liệu, Lưu Transcript, Ingest ChromaDB, Gọi Gemini Tóm tắt.
Có tích hợp Multi-threading Progress Bar để đo lường hiệu năng.
"""

import os
import sys
import time
import threading
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ProgressTracker:
    """
    Vẽ thanh tiến độ (Progress Bar), hiển thị ETA và thời gian đã chạy.
    Chạy trên luồng (thread) riêng để không block API chính.
    """

    def __init__(self, description: str, estimated_seconds: float):
        self.description = description
        self.estimated_seconds = max(estimated_seconds, 1.0)
        self.start_time = 0
        self.done = False
        self.thread = None

    def _animate(self):
        bar_length = 40
        while not self.done:
            elapsed = time.time() - self.start_time
            # Tính % tiến độ (Dừng ở 95% nếu task chưa thực sự xong để chờ)
            pct = min((elapsed / self.estimated_seconds) * 100, 95.0)
            filled = int(bar_length * pct // 100)
            bar = "█" * filled + "-" * (bar_length - filled)

            # Dùng \r để ghi đè dòng hiện tại liên tục
            sys.stdout.write(
                f"\r[*] {self.description:<30}: |{bar}| {pct:5.1f}% "
                f"(Chạy: {elapsed:.1f}s / ETA: {self.estimated_seconds:.1f}s)"
            )
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        self.start_time = time.time()
        self.done = False
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()

    def stop(self):
        self.done = True
        if self.thread:
            self.thread.join()
        elapsed = time.time() - self.start_time
        # Đẩy thanh tiến độ lên 100% khi hoàn thành
        bar = "█" * 40
        sys.stdout.write(
            f"\r[+] {self.description:<30}: |{bar}| 100.0% "
            f"(Hoàn thành trong: {elapsed:.2f}s)                              \n"
        )
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Link YouTube")
    parser.add_argument("--meeting_id", default="yt_meeting_01", help="ID định danh")
    parser.add_argument("--force", action="store_true", help="Xóa dữ liệu cũ, nạp lại từ đầu")
    args = parser.parse_args()

    print("=" * 85)
    print("🚀 TASKFLOW AI — YOUTUBE ANALYSIS PIPELINE (WITH METRICS)")
    print("=" * 85)

    # Import modules cục bộ
    try:
        from utils.ytb_text_utils import get_youtube_transcript
        from core.rag_engine import RAGEngine
        from core.llm_service import LLMService
    except ImportError as e:
        print(f"[!] Lỗi Import: {e}. Vui lòng kiểm tra lại cấu trúc file.")
        sys.exit(1)

    # ---------------------------------------------------------
    # BƯỚC 1: CÀO VÀ LƯU TRANSCRIPT (Tốc độ mạng: ~2.0 giây)
    # ---------------------------------------------------------
    print(f"\n🔗 Phân tích URL: {args.url}")
    tracker_ytb = ProgressTracker("1. Cào dữ liệu & Lưu File", estimated_seconds=2.0)
    tracker_ytb.start()

    try:
        text_content = get_youtube_transcript(args.url, args.meeting_id, BASE_DIR)
        word_count = len(text_content.split())
    except Exception as e:
        tracker_ytb.stop()
        print(f"\n[!] Lỗi: {e}")
        sys.exit(1)
    finally:
        tracker_ytb.stop()

    transcript_path = os.path.join(BASE_DIR, "transcript", f"{args.meeting_id}_youtube.txt")
    print(f"    ↳ Đã trích xuất {word_count} từ. Đã lưu bản thô tại: {transcript_path}")

    # ---------------------------------------------------------
    # BƯỚC 2: INGEST VÀO VECTOR DB (Tốc độ: ~400 từ/giây)
    # ---------------------------------------------------------
    est_ingest_time = max(word_count / 400.0, 2.0)
    tracker_ingest = ProgressTracker("2. Semantic Chunking & Vectorize", estimated_seconds=est_ingest_time)
    tracker_ingest.start()

    rag = RAGEngine(persist_directory=os.path.join(BASE_DIR, "chroma_db"))
    try:
        chunks = rag.ingest(
            meeting_id=args.meeting_id,
            raw_text=text_content,
            force_reingest=args.force
        )
    except Exception as e:
        tracker_ingest.stop()
        print(f"\n[!] Lỗi Ingest: {e}")
        sys.exit(1)
    finally:
        tracker_ingest.stop()

    if not chunks:
        chunks = rag.get_all_chunks(args.meeting_id)
        print(f"    ↳ Tái sử dụng {len(chunks)} chunks cũ đã lưu trong ChromaDB.")
    else:
        print(f"    ↳ Đã chia cắt và nạp {len(chunks)} chunks mới vào DB.")

    # ---------------------------------------------------------
    # BƯỚC 3: LLM TÓM TẮT (Tốc độ Gemini: ~150 từ/giây)
    # ---------------------------------------------------------
    est_llm_time = max(word_count / 150.0, 10.0)
    tracker_llm = ProgressTracker("3. AI Reasoning & Summarizing", estimated_seconds=est_llm_time)
    tracker_llm.start()

    llm = LLMService()
    try:
        # Gọi Gemini phân tích
        result = llm.analyze_meeting(chunks=chunks, meeting_id=args.meeting_id)
    except Exception as e:
        tracker_llm.stop()
        print(f"\n[!] Lỗi AI: {e}")
        sys.exit(1)
    finally:
        tracker_llm.stop()

    # ---------------------------------------------------------
    # BƯỚC 4: IN KẾT QUẢ VÀ LƯU ARTIFACT
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"📝 KẾT QUẢ TÓM TẮT - BỞI {llm.model_name.upper()}")
    print("=" * 85)

    analysis_text = result.get("result", "N/A")
    print(analysis_text)

    summary_dir = os.path.join(BASE_DIR, "summary_results")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{args.meeting_id}_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(analysis_text)

    print("-" * 85)
    print(f"💾 Đã lưu báo cáo AI Markdown tại: {summary_path}")


if __name__ == "__main__":
    main()