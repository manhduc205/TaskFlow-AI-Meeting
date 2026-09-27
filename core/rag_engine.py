"""Compatibility facade over the new in-memory, timestamp-aware RAG session."""

from __future__ import annotations

import json
from pathlib import Path

from core.chunker import TimestampChunker
from core.config import Settings
from core.contracts import TranscriptDocument, TranscriptSegment
from core.rag_session import RAGSession
from models import MeetingChunk
from utils.text_utils import split_into_sentences


class _CollectionStats:
    def __init__(self, owner: "RAGEngine"):
        self.owner = owner

    def count(self) -> int:
        return sum(len(chunks) for chunks in self.owner._chunks.values())


class RAGEngine:
    """Manages in-memory RAG sessions while preserving the previous public API."""

    def __init__(self, persist_directory: str | None = None, settings: Settings | None = None):
        del persist_directory  # Local pipeline intentionally does not persist transcripts.
        self.settings = settings or Settings.from_env()
        self._sessions: dict[str, RAGSession] = {}
        self._chunks: dict[str, list[MeetingChunk]] = {}
        self.collection = _CollectionStats(self)

    @staticmethod
    def _to_meeting_chunks(chunks) -> list[MeetingChunk]:
        return [
            MeetingChunk(
                chunk_id=chunk.chunk_id,
                meeting_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                raw_text=chunk.raw_text,
                normalized_text=chunk.normalized_text,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
            for chunk in chunks
        ]

    @staticmethod
    def _segments_from_json(path: Path, meeting_id: str) -> list[TranscriptSegment]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("segments", []) if isinstance(payload, dict) else payload
        segments: list[TranscriptSegment] = []
        for index, row in enumerate(rows):
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            if "start_ms" in row:
                start_ms = int(row["start_ms"])
                end_ms = int(row.get("end_ms", start_ms))
            else:
                start = float(row.get("start", 0.0))
                end = float(row.get("end", start + float(row.get("duration", 0.0))))
                start_ms = round(start * 1000)
                end_ms = round(end * 1000)
            segments.append(
                TranscriptSegment(
                    segment_id=str(row.get("segment_id", f"{meeting_id}_{index:06d}")),
                    text=text,
                    start_ms=start_ms,
                    end_ms=max(start_ms, end_ms),
                    source=str(row.get("source", "file")),
                    language=str(row.get("language", "unknown")),
                )
            )
        return segments

    @staticmethod
    def _segments_from_text(text: str, meeting_id: str) -> list[TranscriptSegment]:
        sentences = split_into_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()]
        return [
            TranscriptSegment(
                segment_id=f"{meeting_id}_{index:06d}",
                text=sentence,
                start_ms=0,
                end_ms=0,
                source="text",
            )
            for index, sentence in enumerate(sentences)
        ]

    def ingest_segments(
        self,
        meeting_id: str,
        segments: list[TranscriptSegment],
        force_reingest: bool = False,
    ) -> list[MeetingChunk]:
        if meeting_id in self._sessions and not force_reingest:
            return []
        document = TranscriptDocument(
            document_id=meeting_id,
            source=segments[0].source if segments else "unknown",
            segments=list(segments),
            language=segments[0].language if segments else "unknown",
        )
        transcript_chunks = TimestampChunker(self.settings).chunk(document)
        session = RAGSession(transcript_chunks, self.settings).build()
        chunks = self._to_meeting_chunks(transcript_chunks)
        self._sessions[meeting_id] = session
        self._chunks[meeting_id] = chunks
        return chunks

    def ingest(
        self,
        meeting_id: str,
        transcript_path: str | None = None,
        segments_json_path: str | None = None,
        raw_text: str | None = None,
        force_reingest: bool = False,
    ) -> list[MeetingChunk]:
        if meeting_id in self._sessions and not force_reingest:
            return []

        segments: list[TranscriptSegment]
        if segments_json_path:
            path = Path(segments_json_path)
            if not path.is_file():
                raise FileNotFoundError(f"Segments JSON not found: {path}")
            segments = self._segments_from_json(path, meeting_id)
        elif transcript_path:
            path = Path(transcript_path)
            if not path.is_file():
                raise FileNotFoundError(f"Transcript file not found: {path}")
            if path.suffix.lower() == ".json":
                segments = self._segments_from_json(path, meeting_id)
            else:
                segments = self._segments_from_text(
                    path.read_text(encoding="utf-8"), meeting_id
                )
        elif raw_text:
            segments = self._segments_from_text(raw_text, meeting_id)
        else:
            raise ValueError("segments_json_path, transcript_path, or raw_text is required")

        if not segments:
            raise ValueError("Transcript contains no usable segments")
        return self.ingest_segments(meeting_id, segments, force_reingest=True)

    def hybrid_retrieve(
        self, query: str, meeting_id: str, top_k: int | None = None
    ) -> list[MeetingChunk]:
        session = self._sessions.get(meeting_id)
        if session is None:
            return []
        return self._to_meeting_chunks(session.retrieve(query, top_k=top_k))

    def get_all_chunks(self, meeting_id: str) -> list[MeetingChunk]:
        return list(self._chunks.get(meeting_id, []))

    def meeting_exists(self, meeting_id: str) -> bool:
        return meeting_id in self._sessions

    def delete_meeting(self, meeting_id: str) -> int:
        count = len(self._chunks.pop(meeting_id, []))
        self._sessions.pop(meeting_id, None)
        return count
