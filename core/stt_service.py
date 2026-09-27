"""Lazy Faster-Whisper adapter for local audio and video files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.config import Settings
from core.contracts import TranscriptDocument, TranscriptSegment


ProgressCallback = Callable[[float, str], None]
MIN_TRANSCRIPT_SEGMENT_MS = 20_000
MAX_TRANSCRIPT_SEGMENT_MS = 30_000


def _segment_duration(segments: list[TranscriptSegment]) -> int:
    return segments[-1].end_ms - segments[0].start_ms


def _merge_transcript_segments(
    segments: list[TranscriptSegment],
    min_duration_ms: int = MIN_TRANSCRIPT_SEGMENT_MS,
    max_duration_ms: int = MAX_TRANSCRIPT_SEGMENT_MS,
) -> list[TranscriptSegment]:
    """Group Whisper's short phrases into readable 20-30 second passages."""
    if not segments:
        return []

    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        projected = current + [segment]
        if (current
                and _segment_duration(projected) > max_duration_ms
                and _segment_duration(current) >= min_duration_ms):
            groups.append(current)
            current = []

        current.append(segment)
        duration = _segment_duration(current)
        remaining = segments[index + 1:]
        remaining_duration = (
            remaining[-1].end_ms - remaining[0].start_ms if remaining else 0
        )
        ends_sentence = segment.text.rstrip().endswith((".", "?", "!", "…"))
        if (duration >= min_duration_ms
                and (not remaining or remaining_duration >= min_duration_ms)
                and (ends_sentence or duration >= max_duration_ms)):
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    if len(groups) > 1 and _segment_duration(groups[-1]) < min_duration_ms:
        combined = groups[-2] + groups[-1]
        best_split: tuple[list[TranscriptSegment], list[TranscriptSegment]] | None = None
        best_distance = float("inf")
        for split_at in range(1, len(combined)):
            left, right = combined[:split_at], combined[split_at:]
            left_duration = _segment_duration(left)
            right_duration = _segment_duration(right)
            if (min_duration_ms <= left_duration <= max_duration_ms
                    and min_duration_ms <= right_duration <= max_duration_ms):
                distance = abs(left_duration - 25_000) + abs(right_duration - 25_000)
                if distance < best_distance:
                    best_split = (left, right)
                    best_distance = distance
        if best_split:
            groups[-2], groups[-1] = best_split
        elif _segment_duration(combined) <= max_duration_ms:
            groups[-2:] = [combined]

    merged: list[TranscriptSegment] = []
    for index, group in enumerate(groups):
        merged.append(TranscriptSegment(
            segment_id=f"stt_{index:06d}",
            text=" ".join(item.text.strip() for item in group),
            start_ms=group[0].start_ms,
            end_ms=group[-1].end_ms,
            source="whisper",
            language=group[0].language,
        ))
    return merged


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class WhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._device = self._resolve_device(settings.stt_device)
        self._compute_type = self._resolve_compute_type(settings.stt_compute_type)

    @staticmethod
    def _resolve_device(configured: str) -> str:
        if configured == "auto":
            return "cuda" if _cuda_available() else "cpu"
        if configured == "cuda" and not _cuda_available():
            raise RuntimeError("STT_DEVICE=cuda but CUDA is not available")
        return configured

    def _resolve_compute_type(self, configured: str) -> str:
        if configured != "auto":
            return configured
        return "int8_float16" if self._device == "cuda" else "int8"

    @property
    def device(self) -> str:
        return self._device

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self.settings.stt_download_root.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                self.settings.stt_model,
                device=self._device,
                compute_type=self._compute_type,
                download_root=str(self.settings.stt_download_root),
                num_workers=1,
                cpu_threads=4,
            )
        return self._model

    def transcribe(
        self,
        media_path: str | Path,
        document_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> TranscriptDocument:
        path = Path(media_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio/video file not found: {path}")

        model = self._get_model()
        language = None if self.settings.stt_language == "auto" else self.settings.stt_language
        generator, info = model.transcribe(
            str(path),
            beam_size=self.settings.stt_beam_size,
            language=language,
            vad_filter=self.settings.stt_vad_filter,
            condition_on_previous_text=False,
            vad_parameters={
                "min_silence_duration_ms": self.settings.stt_min_silence_ms
            },
        )

        duration = max(float(getattr(info, "duration", 0.0)), 0.001)
        detected_language = getattr(info, "language", None) or language or "unknown"
        language_probability = getattr(info, "language_probability", None)
        segments: list[TranscriptSegment] = []

        for index, item in enumerate(generator):
            text = item.text.strip()
            if not text:
                continue
            start_ms = max(0, round(float(item.start) * 1000))
            end_ms = max(start_ms, round(float(item.end) * 1000))
            segments.append(
                TranscriptSegment(
                    segment_id=f"stt_{index:06d}",
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    source="whisper",
                    language=detected_language,
                    confidence=language_probability,
                )
            )
            if progress:
                progress(min(float(item.end) / duration, 1.0), text)

        if not segments:
            raise RuntimeError(f"Whisper produced no transcript for: {path}")

        segments = _merge_transcript_segments(segments)

        return TranscriptDocument(
            document_id=document_id or path.stem,
            source="whisper",
            segments=segments,
            language=detected_language,
            title=path.name,
        )
