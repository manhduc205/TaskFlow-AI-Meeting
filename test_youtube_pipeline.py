"""Convenience entrypoint for the complete local YouTube pipeline."""

from __future__ import annotations

import argparse

from main_poc import main as pipeline_main


def main() -> int:
    parser = argparse.ArgumentParser(description="YouTube -> summary.md -> optional chat")
    parser.add_argument("--url", required=True)
    parser.add_argument("--meeting_id", default="yt_meeting_01")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--query", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated: local sessions are rebuilt on every run",
    )
    parser.add_argument("--languages", nargs="+", default=["vi", "en"])
    args = parser.parse_args()

    forwarded = [
        "youtube",
        "--url",
        args.url,
        "--id",
        args.meeting_id,
        "--languages",
        *args.languages,
    ]
    if args.chat:
        forwarded.append("--chat")
    if args.query:
        forwarded.extend(["--query", args.query])
    if args.debug:
        forwarded.append("--debug")
    return pipeline_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
