"""YouTube caption adapter that returns structured segments without disk writes."""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from core.contracts import TranscriptDocument, TranscriptSegment


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def extract_youtube_video_id(url: str) -> str:
    candidate_url = url.strip()
    if not candidate_url:
        raise ValueError("YouTube URL cannot be empty")
    if "://" not in candidate_url:
        candidate_url = "https://" + candidate_url

    parsed = urlparse(candidate_url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in YOUTUBE_HOSTS:
        raise ValueError(f"Unsupported YouTube host: {hostname or '(missing)'}")

    video_id = ""
    path_parts = [part for part in parsed.path.split("/") if part]
    if hostname.endswith("youtu.be"):
        video_id = path_parts[0] if path_parts else ""
    elif parsed.path.rstrip("/") == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif len(path_parts) == 2 and path_parts[0] in {"shorts", "embed", "live"}:
        video_id = path_parts[1]

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("URL does not contain a valid YouTube video ID")
    return video_id


class YouTubeTranscriptService:
    def __init__(self, api: YouTubeTranscriptApi | None = None):
        self.api = api or YouTubeTranscriptApi()

    def fetch(
        self,
        url: str,
        document_id: str | None = None,
        languages: list[str] | None = None,
    ) -> TranscriptDocument:
        video_id = extract_youtube_video_id(url)
        preferred_languages = languages or ["vi", "en"]
        try:
            transcript = self.api.fetch(video_id, languages=preferred_languages)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot fetch YouTube captions for video {video_id}: {exc}"
            ) from exc

        language = getattr(transcript, "language_code", None) or "unknown"
        segments: list[TranscriptSegment] = []
        for index, item in enumerate(transcript):
            if isinstance(item, dict):
                text = item.get("text", "")
                start = float(item.get("start", 0.0))
                duration = float(item.get("duration", 0.0))
            else:
                text = getattr(item, "text", "")
                start = float(getattr(item, "start", 0.0))
                duration = float(getattr(item, "duration", 0.0))

            cleaned = re.sub(r"\s+", " ", html.unescape(text)).strip()
            if not cleaned:
                continue
            start_ms = max(0, round(start * 1000))
            end_ms = max(start_ms, round((start + duration) * 1000))
            segments.append(
                TranscriptSegment(
                    segment_id=f"yt_{video_id}_{index:06d}",
                    text=cleaned,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    source="youtube",
                    language=language,
                )
            )

        if not segments:
            raise RuntimeError(f"YouTube video {video_id} returned an empty transcript")

        return TranscriptDocument(
            document_id=document_id or video_id,
            source="youtube",
            segments=segments,
            language=language,
            title=f"YouTube {video_id}",
        )
