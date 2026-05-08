"""
models.py — Phase 1: Unified Document Model
Định nghĩa Data Class chuẩn cho mỗi chunk hội thoại.
chunk_id là chìa khóa để Vector DB và BM25 "nói chuyện" với nhau.
"""

from dataclasses import dataclass, field
import uuid


@dataclass
class MeetingChunk:
    """
    Đơn vị dữ liệu cơ bản chạy xuyên suốt toàn bộ pipeline RAG.

    Attributes:
        chunk_id       : UUID duy nhất, khóa liên kết ChromaDB ↔ BM25 ↔ Cache
        meeting_id     : Định danh cuộc họp
        chunk_index    : Số thứ tự thời gian gốc (dùng để sort timeline)
        start_time     : Timestamp bắt đầu (giây)
        end_time       : Timestamp kết thúc (giây)
        raw_text       : Văn bản gốc (dành cho LLM đọc và hiển thị)
        normalized_text: Văn bản đã chuẩn hóa pyvi (dành cho RAG tìm kiếm)
    """
    meeting_id: str
    chunk_index: int
    raw_text: str
    normalized_text: str
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: str = field(
        default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}"
    )

    def to_chroma_metadata(self) -> dict:
        """Metadata lưu vào ChromaDB (chỉ scalar types)."""
        return {
            "chunk_id": self.chunk_id,
            "meeting_id": self.meeting_id,
            "chunk_index": self.chunk_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def to_cache_entry(self) -> dict:
        """Entry lưu vào Master Metadata Cache (O(1) lookup)."""
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "chunk_index": self.chunk_index,
            "meeting_id": self.meeting_id,
        }

    def to_context_string(self) -> str:
        """Chuỗi context gửi cho LLM, kèm timestamp nếu có."""
        if self.start_time > 0 or self.end_time > 0:
            return f"[{self.start_time:.1f}s → {self.end_time:.1f}s]: {self.raw_text}"
        return self.raw_text


@dataclass
class MeetingAnalysisResult:
    """Kết quả phân tích cuộc họp từ LLM."""
    meeting_id: str
    summary: str
    key_technical_points: list
    decisions: list
    tasks: list

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "summary": self.summary,
            "key_technical_points": self.key_technical_points,
            "decisions": self.decisions,
            "tasks": self.tasks,
        }
