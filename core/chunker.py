"""Timestamp-aware chunking that preserves source segment boundaries."""

from __future__ import annotations

import hashlib

from core.config import Settings
from core.contracts import TranscriptChunk, TranscriptDocument, TranscriptSegment
from utils.text_utils import normalize_text


class _TimestampChunkerBase:
    def __init__(
        self,
        target_chars: int,
        min_duration_seconds: float,
        max_duration_seconds: float,
        silence_gap_seconds: float,
        chunk_prefix: str,
    ):
        self.target_chars = target_chars
        self.min_duration_ms = int(min_duration_seconds * 1000)
        self.max_duration_ms = int(max_duration_seconds * 1000)
        self.silence_gap_ms = int(silence_gap_seconds * 1000)
        self.chunk_prefix = chunk_prefix

    def chunk(self, document: TranscriptDocument) -> list[TranscriptChunk]:
        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        current_chars = 0

        for segment in document.segments:
            if current:
                current_duration = current[-1].end_ms - current[0].start_ms
                projected_duration = segment.end_ms - current[0].start_ms
                silence_gap = max(0, segment.start_ms - current[-1].end_ms)
                projected_chars = current_chars + 1 + len(segment.text)

                break_for_silence = (
                    current_duration >= self.min_duration_ms
                    and silence_gap >= self.silence_gap_ms
                )
                has_timing = projected_duration > 0
                break_for_size = (
                    projected_chars > self.target_chars
                    and (not has_timing or current_duration >= self.min_duration_ms)
                )
                break_for_duration = projected_duration > self.max_duration_ms

                if break_for_silence or break_for_size or break_for_duration:
                    groups.append(current)
                    current = []
                    current_chars = 0

            current.append(segment)
            current_chars += len(segment.text) + (1 if current_chars else 0)

        if current:
            groups.append(current)

        chunks: list[TranscriptChunk] = []
        for index, group in enumerate(groups):
            raw_text = " ".join(item.text.strip() for item in group).strip()
            digest_input = (
                f"{document.document_id}:{index}:{group[0].start_ms}:"
                f"{group[-1].end_ms}:{raw_text}"
            )
            digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
            chunks.append(
                TranscriptChunk(
                    chunk_id=f"{self.chunk_prefix}_{index:04d}_{digest}",
                    document_id=document.document_id,
                    chunk_index=index,
                    raw_text=raw_text,
                    normalized_text=normalize_text(raw_text),
                    start_ms=group[0].start_ms,
                    end_ms=group[-1].end_ms,
                    segment_ids=tuple(item.segment_id for item in group),
                )
            )
        source_ids = [segment.segment_id for segment in document.segments]
        chunk_ids = [segment_id for chunk in chunks for segment_id in chunk.segment_ids]
        reconstructed_text = " ".join(chunk.raw_text for chunk in chunks)
        if chunk_ids != source_ids or reconstructed_text != document.text:
            raise RuntimeError(
                "Chunk coverage validation failed: transcript content was lost, "
                "duplicated, or reordered"
            )
        return chunks


class TimestampChunker(_TimestampChunkerBase):
    """Small chunks optimized for accurate RAG retrieval and neighboring context."""

    def __init__(self, settings: Settings):
        super().__init__(
            target_chars=settings.rag_chunk_target_chars,
            min_duration_seconds=settings.rag_min_chunk_duration_seconds,
            max_duration_seconds=settings.rag_max_chunk_duration_seconds,
            silence_gap_seconds=settings.rag_silence_gap_seconds,
            chunk_prefix="rag",
        )

