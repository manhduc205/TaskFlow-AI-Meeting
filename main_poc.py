"""Unified local CLI: source -> timestamped transcript -> summary.md -> optional chat."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.chunker import TimestampChunker
from core.config import Settings
from core.contracts import TranscriptDocument, TranscriptSegment, format_timestamp
from core.llm_service import LLMService, estimate_tokens
from core.rag_session import RAGSession
from core.stt_service import WhisperTranscriber
from core.summary_service import SummaryService, safe_artifact_id
from core.youtube_service import YouTubeTranscriptService
from utils.text_utils import split_into_sentences


LEGACY_TIMESTAMP_LINE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)s\]:\s*(?P<text>.+)$"
)


def _stage(message: str) -> None:
    print(f"\n[*] {message}", flush=True)


def _document_from_transcript_file(path: Path, document_id: str) -> TranscriptDocument:
    if not path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    if path.suffix.lower() == ".json":
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
                start_seconds = float(row.get("start", 0.0))
                end_seconds = float(
                    row.get("end", start_seconds + float(row.get("duration", 0.0)))
                )
                start_ms = round(start_seconds * 1000)
                end_ms = round(end_seconds * 1000)
            segments.append(
                TranscriptSegment(
                    segment_id=str(row.get("segment_id", f"file_{index:06d}")),
                    text=text,
                    start_ms=start_ms,
                    end_ms=max(start_ms, end_ms),
                    source="file",
                    language=str(row.get("language", "unknown")),
                )
            )
    else:
        text = path.read_text(encoding="utf-8")
        legacy_rows = []
        for line in text.splitlines():
            match = LEGACY_TIMESTAMP_LINE.match(line.strip())
            if match:
                legacy_rows.append(
                    (round(float(match.group("start")) * 1000), match.group("text"))
                )
        if legacy_rows:
            segments = []
            for index, (start_ms, row_text) in enumerate(legacy_rows):
                next_start_ms = (
                    legacy_rows[index + 1][0]
                    if index + 1 < len(legacy_rows)
                    else start_ms
                )
                segments.append(
                    TranscriptSegment(
                        segment_id=f"legacy_{index:06d}",
                        text=row_text,
                        start_ms=start_ms,
                        end_ms=max(start_ms, next_start_ms),
                        source="legacy_timestamp_text",
                    )
                )
        else:
            sentences = split_into_sentences(text) or (
                [text.strip()] if text.strip() else []
            )
            segments = [
                TranscriptSegment(
                    segment_id=f"file_{index:06d}",
                    text=sentence,
                    start_ms=0,
                    end_ms=0,
                    source="file",
                )
                for index, sentence in enumerate(sentences)
            ]

    if not segments:
        raise ValueError(f"Transcript has no usable text: {path}")
    language = segments[0].language
    return TranscriptDocument(
        document_id=document_id,
        source="file",
        segments=segments,
        language=language,
        title=path.name,
    )


def _save_debug(document: TranscriptDocument, chunks, settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = safe_artifact_id(document.document_id)
    segments_path = settings.output_dir / f"{safe_id}_segments.json"
    chunks_path = settings.output_dir / f"{safe_id}_chunks.json"
    segments_path.write_text(
        json.dumps(
            [segment.to_dict() for segment in document.segments],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(
            [chunk.to_dict() for chunk in chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[debug] Segments: {segments_path}")
    print(f"[debug] Chunks  : {chunks_path}")


def _build_rag(chunks, settings: Settings) -> tuple[RAGSession, float]:
    started = time.perf_counter()
    session = RAGSession(chunks, settings).build()
    return session, time.perf_counter() - started


def _summary_event(event: str) -> None:
    if event == "first_token":
        print("[+] Gemini returned the first summary token.", flush=True)
    elif event.startswith("request_sent:"):
        strategy = event.split(":", 1)[1]
        print(f"[*] Summary request sent ({strategy}).", flush=True)
    elif event.startswith("map:"):
        print(f"[*] Summarizing section {event.split(':', 1)[1]}...", flush=True)
    elif event.startswith("reduce_round:"):
        print(
            f"[*] Condensing long summary batches {event.split(':', 1)[1]}...",
            flush=True,
        )
    elif event.startswith("retry:"):
        print(f"[!] Retrying Gemini request ({event.split(':', 1)[1]})...", flush=True)


def _interactive_chat(rag: RAGSession, llm: LLMService, initial_query: str = "") -> None:
    print("\n=== CHAT (type 'quit' to stop) ===")
    history: list[dict[str, str]] = []
    pending = initial_query.strip()
    while True:
        if pending:
            query = pending
            pending = ""
            print(f"\nYou: {query}")
        else:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        retrieved = rag.retrieve(query)
        started = time.perf_counter()
        answer = llm.chat(query, retrieved, conversation_history=history)
        print(f"\nAI: {answer}")
        if retrieved:
            citations = ", ".join(
                f"[{format_timestamp(chunk.start_ms)}-{format_timestamp(chunk.end_ms)}]"
                for chunk in retrieved
            )
            print(f"\nRetrieved evidence: {citations}")
        print(f"Chat latency: {time.perf_counter() - started:.2f}s")
        history.extend(
            [
                {"role": "user", "text": query},
                {"role": "model", "text": answer},
            ]
        )


def run_pipeline(
    document: TranscriptDocument,
    settings: Settings,
    with_chat: bool = False,
    initial_query: str = "",
    debug: bool = False,
) -> None:
    timings: dict[str, float] = {}
    wants_chat = with_chat or bool(initial_query)
    _stage("[2/4] Creating detailed timestamp-preserving chunks...")
    started = time.perf_counter()
    chunks = TimestampChunker(settings).chunk(document)
    timings["chunking"] = time.perf_counter() - started
    print(
        f"[+] {len(document.segments)} segments -> {len(chunks)} detailed chunks "
        f"in {timings['chunking']:.3f}s"
    )
    print(f"[+] Chunk coverage: {len(document.segments)}/{len(document.segments)} segments")
    summary_input = "\n".join(chunk.to_context_string() for chunk in chunks)
    summary_strategy = (
        "direct"
        if estimate_tokens(summary_input)
        <= settings.summary_direct_max_input_tokens
        else "map_reduce"
    )
    print(
        f"[+] Summary config: model={settings.llm_summary_model}, "
        f"strategy={summary_strategy}, input≈{estimate_tokens(summary_input)} tokens, "
        f"output_max={settings.llm_summary_max_output_tokens or 'SDK default'}, "
        f"thinking={settings.llm_thinking_level}, "
        f"timeout={settings.llm_timeout_seconds}s, "
        f"attempts={settings.llm_max_retries}"
    )

    if debug or settings.save_debug_artifacts:
        _save_debug(document, chunks, settings)

    llm = LLMService(settings)
    rag_future = None
    executor = ThreadPoolExecutor(max_workers=1) if wants_chat else None
    try:
        if executor:
            _stage("[3/4] Generating summary and building the chat index in parallel...")
            rag_future = executor.submit(_build_rag, chunks, settings)
        else:
            _stage("[3/4] Generating summary...")

        started = time.perf_counter()
        first_token_at: float | None = None

        def summary_event(event: str) -> None:
            nonlocal first_token_at
            if event == "first_token" and first_token_at is None:
                first_token_at = time.perf_counter()
            _summary_event(event)

        summary = SummaryService(settings, llm=llm).create(
            document.document_id,
            chunks,
            on_event=summary_event,
        )
        timings["summary"] = time.perf_counter() - started
        if first_token_at is not None:
            timings["summary_first_token"] = first_token_at - started
        _stage("[4/4] Finalizing local artifacts...")
        print(f"[+] Summary saved: {summary.output_path}")
        print(f"[+] Summary strategy: {summary.strategy} ({timings['summary']:.2f}s)")
        print(f"[+] Summary output: {len(summary.markdown.split())} words")

        rag = None
        if rag_future:
            rag, timings["rag_index"] = rag_future.result()
            print(f"[+] Chat index ready in {timings['rag_index']:.2f}s")

        if rag and wants_chat:
            _interactive_chat(rag, llm, initial_query=initial_query)
    finally:
        if executor:
            executor.shutdown(wait=True)

    print("\n=== METRICS ===")
    print(f"Segments       : {len(document.segments)}")
    print(f"Chunks         : {len(chunks)}")
    print(f"Chunking       : {timings['chunking']:.3f}s")
    print(f"Summary        : {timings.get('summary', 0.0):.2f}s")
    if "summary_first_token" in timings:
        print(f"First token    : {timings['summary_first_token']:.2f}s")
    if "rag_index" in timings:
        print(f"RAG index      : {timings['rag_index']:.2f}s")
    print(f"Summary model  : {settings.llm_summary_model}")
    print(f"Chat model     : {settings.llm_chat_model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local transcript, Markdown summary, and RAG chat pipeline"
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--id", dest="document_id", required=True, help="Local document ID")
    common.add_argument("--chat", action="store_true", help="Open interactive chat")
    common.add_argument("--query", default="", help="Ask this question before interactive chat")
    common.add_argument("--debug", action="store_true", help="Save segments/chunks JSON")

    youtube = subparsers.add_parser("youtube", parents=[common], help="Use YouTube captions")
    youtube.add_argument("--url", required=True, help="YouTube video URL")
    youtube.add_argument(
        "--languages",
        nargs="+",
        default=["vi", "en"],
        help="Caption languages in priority order",
    )

    audio = subparsers.add_parser("audio", parents=[common], help="Transcribe local media")
    audio.add_argument("--file", required=True, help="Local audio/video file")

    transcript = subparsers.add_parser(
        "transcript", parents=[common], help="Use an existing .txt/.json transcript"
    )
    transcript.add_argument("--file", required=True, help="Transcript path")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    print("=== TASKFLOW LOCAL VIDEO PIPELINE ===")
    source_started = time.perf_counter()
    _stage("[1/4] Loading transcript source...")
    try:
        if args.source == "youtube":
            document = YouTubeTranscriptService().fetch(
                args.url,
                document_id=args.document_id,
                languages=args.languages,
            )
        elif args.source == "audio":
            transcriber = WhisperTranscriber(settings)
            print(
                f"[*] Whisper model={settings.stt_model}, "
                f"device={transcriber.device}"
            )

            last_percent = -10

            def progress(fraction: float, _: str) -> None:
                nonlocal last_percent
                percent = int(fraction * 100)
                if percent >= last_percent + 10:
                    last_percent = percent
                    print(f"[*] Transcribing: {percent}%", flush=True)

            document = transcriber.transcribe(
                args.file,
                document_id=args.document_id,
                progress=progress,
            )
        else:
            document = _document_from_transcript_file(
                Path(args.file).expanduser().resolve(), args.document_id
            )

        source_elapsed = time.perf_counter() - source_started
        print(
            f"[+] Source ready: {len(document.segments)} segments, "
            f"duration={format_timestamp(document.duration_ms)}, "
            f"language={document.language}, time={source_elapsed:.2f}s"
        )
        run_pipeline(
            document,
            settings,
            with_chat=args.chat,
            initial_query=args.query,
            debug=args.debug,
        )
        return 0
    except Exception as exc:
        print(f"\n[!] Pipeline failed: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
