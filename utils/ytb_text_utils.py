"""Compatibility helpers backed by the structured YouTube transcript service."""

from __future__ import annotations

from core.contracts import TranscriptDocument
from core.youtube_service import (
    YouTubeTranscriptService,
    extract_youtube_video_id,
)


def get_youtube_transcript_as_segments(
    video_url: str,
    document_id: str | None = None,
    languages: list[str] | None = None,
) -> TranscriptDocument:
    """Return structured caption segments without writing transcript files."""
    return YouTubeTranscriptService().fetch(
        video_url,
        document_id=document_id,
        languages=languages,
    )


def get_youtube_transcript_as_text(video_url: str) -> str:
    """Legacy text-only helper. Timestamp structure is intentionally omitted."""
    return get_youtube_transcript_as_segments(video_url).text


def get_youtube_transcript(video_url: str, meeting_id: str, base_dir: str = "") -> str:
    """Legacy helper retained for callers; no transcript file is created."""
    del base_dir
    return get_youtube_transcript_as_segments(video_url, document_id=meeting_id).text


__all__ = [
    "extract_youtube_video_id",
    "get_youtube_transcript",
    "get_youtube_transcript_as_segments",
    "get_youtube_transcript_as_text",
]
