"""Creates the single default local artifact: a Markdown summary file."""

from __future__ import annotations

import re
from typing import Callable, Sequence

from core.config import Settings
from core.contracts import SummaryResult
from core.llm_service import LLMService


def safe_artifact_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "summary"


class SummaryService:
    def __init__(self, settings: Settings, llm: LLMService | None = None):
        self.settings = settings
        self.llm = llm or LLMService(settings)

    def create(
        self,
        document_id: str,
        chunks: Sequence,
        on_event: Callable[[str], None] | None = None,
    ) -> SummaryResult:
        markdown, strategy = self.llm.summarize(chunks, document_id, on_event=on_event)
        if not markdown.strip():
            raise RuntimeError("Summary Markdown is empty")

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.settings.output_dir / f"{safe_artifact_id(document_id)}_summary.md"
        temporary_path = output_path.with_suffix(".md.tmp")
        temporary_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        temporary_path.replace(output_path)

        return SummaryResult(
            document_id=document_id,
            markdown=markdown,
            strategy=strategy,
            input_chunks=len(chunks),
            output_path=str(output_path),
        )
