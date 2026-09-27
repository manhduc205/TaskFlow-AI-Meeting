"""YouTube caption/chunk smoke test that does not call Gemini or write files."""

from __future__ import annotations

import argparse

from core.chunker import TimestampChunker
from core.config import Settings
from core.contracts import format_timestamp
from core.youtube_service import YouTubeTranscriptService


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test YouTube captions and chunking")
    parser.add_argument("--url", required=True)
    parser.add_argument("--meeting_id", default="youtube_local_test")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated: this smoke test never persists an index",
    )
    parser.add_argument("--languages", nargs="+", default=["vi", "en"])
    args = parser.parse_args()

    document = YouTubeTranscriptService().fetch(
        args.url,
        document_id=args.meeting_id,
        languages=args.languages,
    )
    chunks = TimestampChunker(Settings.from_env()).chunk(document)
    print(f"Segments: {len(document.segments)}")
    print(f"Chunks  : {len(chunks)}")
    for chunk in chunks[:3]:
        print(
            f"[{format_timestamp(chunk.start_ms)}-{format_timestamp(chunk.end_ms)}] "
            f"{chunk.raw_text[:140]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
