"""Backward-compatible command entrypoint for the local Whisper pipeline.

Prefer: python main_poc.py audio --file <media> --id <id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from main_poc import main as pipeline_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe and summarize local media")
    parser.add_argument("--file", required=True, help="Audio/video path")
    parser.add_argument("--id", dest="document_id", required=True)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    forwarded = ["audio", "--file", args.file, "--id", args.document_id]
    if args.chat:
        forwarded.append("--chat")
    if args.debug:
        forwarded.append("--debug")
    return pipeline_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
