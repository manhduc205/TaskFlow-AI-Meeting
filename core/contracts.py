"""Typed data contracts shared by all local pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def format_timestamp(milliseconds: int) -> str:
    total_seconds = max(0, int(milliseconds)) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    text: str
    start_ms: int
    end_ms: int
    source: str
    language: str = "unknown"
    confidence: float | None = None
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("TranscriptSegment.text cannot be empty")
        if self.start_ms < 0:
            raise ValueError("TranscriptSegment.start_ms cannot be negative")
        if self.end_ms < self.start_ms:
            raise ValueError("TranscriptSegment.end_ms cannot be before start_ms")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptDocument:
    document_id: str
    source: str
    segments: list[TranscriptSegment]
    language: str = "unknown"
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("TranscriptDocument.document_id cannot be empty")
        if not self.segments:
            raise ValueError("TranscriptDocument.segments cannot be empty")
        self.segments.sort(key=lambda item: (item.start_ms, item.end_ms))

    @property
    def duration_ms(self) -> int:
        return max(segment.end_ms for segment in self.segments)

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments)


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    raw_text: str
    normalized_text: str
    start_ms: int
    end_ms: int
    segment_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def start_time(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_time(self) -> float:
        return self.end_ms / 1000.0

    def to_context_string(self) -> str:
        if self.start_ms == 0 and self.end_ms == 0:
            return self.raw_text
        start = format_timestamp(self.start_ms)
        return f"[{start}] {self.raw_text}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["segment_ids"] = list(self.segment_ids)
        return result


@dataclass(frozen=True)
class SummaryResult:
    document_id: str
    markdown: str
    strategy: str
    input_chunks: int
    output_path: str | None = None
