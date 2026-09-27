"""Gemini integration for local Markdown summaries and timestamp-grounded chat."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterator, Sequence

from google import genai
from google.genai import types

from core.config import PROJECT_ROOT, Settings


logger = logging.getLogger(__name__)
EventCallback = Callable[[str], None]
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def estimate_tokens(text: str) -> int:
    """Conservative local estimate for mixed Vietnamese/English content."""
    return max(1, len(text) // 3)


def _load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()

    # The existing prompt files are stored like escaped string literals. Decode
    # only the known escapes so Vietnamese characters are preserved verbatim.
    if raw.startswith(('"', "'")):
        quote = raw[0]
        raw = raw[1:]
        if raw.endswith(quote) and not raw.endswith("\\" + quote):
            raw = raw[:-1]
    return (
        raw.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )


def _append_prompt(prompt: str, content: str) -> str:
    """Append source content without adding or rewriting any prompt sentence."""
    separator = "" if prompt.endswith("\n") else "\n"
    return f"{prompt}{separator}{content}"


def _context_line(item) -> str:
    if hasattr(item, "to_context_string"):
        return item.to_context_string()
    return str(item)


class LLMService:
    def __init__(self, settings: Settings | None = None, client=None):
        self.settings = settings or Settings.from_env()
        self.model_name = self.settings.llm_summary_model  # Legacy callers.
        self.summary_model = self.settings.llm_summary_model
        self.chat_model = self.settings.llm_chat_model
        self.summary_prompt = _load_prompt(PROJECT_ROOT / "prompts" / "summary_prompt.txt")
        self.meeting_summary_prompt = _load_prompt(
            PROJECT_ROOT / "prompts" / "meeting_summary_prompt.txt"
        )
        self.chat_prompt = _load_prompt(PROJECT_ROOT / "prompts" / "chat_prompt.txt")
        if client is not None:
            self.client = client
        elif self.settings.llm_timeout_seconds > 0:
            self.client = genai.Client(
                api_key=self.settings.require_gemini_key(),
                http_options=types.HttpOptions(
                    timeout=self.settings.llm_timeout_seconds * 1000
                ),
            )
        else:
            # Match the original project: use the SDK's default transport timeout.
            self.client = genai.Client(api_key=self.settings.require_gemini_key())

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        for name in ("status_code", "code"):
            value = getattr(exc, name, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    value = None
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                pass
        return None

    def _is_retryable(self, exc: Exception) -> bool:
        status = self._status_code(exc)
        if status in RETRYABLE_STATUS_CODES:
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "timeout",
                "timed out",
                "temporarily unavailable",
                "connection reset",
            )
        )

    def _thinking_config(self) -> types.ThinkingConfig | None:
        level = self.settings.llm_thinking_level
        if level == "default":
            return None
        return types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, level.upper()),
            include_thoughts=False,
        )

    def _config(
        self,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
            max_output_tokens=max_output_tokens or None,
            response_mime_type="text/plain",
            thinking_config=self._thinking_config(),
        )

    @staticmethod
    def _finish_reason(response) -> str | None:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        return str(getattr(reason, "name", None) or getattr(reason, "value", None) or reason)

    @classmethod
    def _ensure_not_truncated(cls, response) -> None:
        reason = cls._finish_reason(response)
        if reason and "MAX_TOKENS" in reason.upper():
            raise RuntimeError(
                "Gemini stopped at MAX_TOKENS; no incomplete summary was saved. "
                "Increase LLM_SUMMARY_MAX_OUTPUT_TOKENS in .env."
            )

    def _generate(
        self,
        model: str,
        contents: str,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> str:
        wait = self.settings.llm_retry_base_seconds
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=self._config(
                        system_instruction, temperature, max_output_tokens
                    ),
                )
                self._ensure_not_truncated(response)
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Gemini returned an empty response")
                return text.strip()
            except Exception as exc:
                if not self._is_retryable(exc) or attempt >= self.settings.llm_max_retries:
                    raise
                logger.warning(
                    "Gemini request failed (%s/%s): %s; retrying in %.1fs",
                    attempt,
                    self.settings.llm_max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                wait *= 2
        raise RuntimeError("Gemini request failed")

    def _generate_stream(
        self,
        model: str,
        contents: str,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int | None,
        on_event: EventCallback | None = None,
    ) -> str:
        wait = self.settings.llm_retry_base_seconds
        for attempt in range(1, self.settings.llm_max_retries + 1):
            pieces: list[str] = []
            yielded_any = False
            last_response = None
            try:
                stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=self._config(
                        system_instruction, temperature, max_output_tokens
                    ),
                )
                for chunk in stream:
                    last_response = chunk
                    text = getattr(chunk, "text", None)
                    if text:
                        if not yielded_any and on_event:
                            on_event("first_token")
                        yielded_any = True
                        pieces.append(text)
                self._ensure_not_truncated(last_response)
                result = "".join(pieces).strip()
                if not result:
                    raise RuntimeError("Gemini returned an empty response")
                return result
            except Exception as exc:
                if (
                    yielded_any
                    or not self._is_retryable(exc)
                    or attempt >= self.settings.llm_max_retries
                ):
                    raise
                if on_event:
                    on_event(f"retry:{attempt}")
                logger.warning("Gemini stream failed: %s; retrying in %.1fs", exc, wait)
                time.sleep(wait)
                wait *= 2
        raise RuntimeError("Gemini stream failed")

    def _generate_stream_iter(
        self,
        model: str,
        contents: str,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> Iterator[str]:
        """Yield model text immediately; retry only before the first token."""
        wait = self.settings.llm_retry_base_seconds
        for attempt in range(1, self.settings.llm_max_retries + 1):
            yielded_any = False
            last_response = None
            try:
                stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=self._config(
                        system_instruction, temperature, max_output_tokens
                    ),
                )
                for chunk in stream:
                    last_response = chunk
                    text = getattr(chunk, "text", None)
                    if text:
                        yielded_any = True
                        yield text
                self._ensure_not_truncated(last_response)
                if not yielded_any:
                    raise RuntimeError("Gemini returned an empty response")
                return
            except Exception as exc:
                if (
                    yielded_any
                    or not self._is_retryable(exc)
                    or attempt >= self.settings.llm_max_retries
                ):
                    raise
                logger.warning("Gemini stream failed: %s; retrying in %.1fs", exc, wait)
                time.sleep(wait)
                wait *= 2
        raise RuntimeError("Gemini stream failed")

    @staticmethod
    def _clean_markdown(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```markdown") and stripped.endswith("```"):
            stripped = stripped[len("```markdown") : -3].strip()
        elif stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped[3:-3].strip()
        return stripped

    @staticmethod
    def _batch_context(items: Sequence, max_tokens: int) -> list[list]:
        batches: list[list] = []
        current: list = []
        current_tokens = 0
        for item in items:
            item_tokens = estimate_tokens(_context_line(item))
            if current and current_tokens + item_tokens > max_tokens:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += item_tokens
        if current:
            batches.append(current)
        return batches

    def summarize(
        self,
        chunks: Sequence,
        document_id: str,
        on_event: EventCallback | None = None,
    ) -> tuple[str, str]:
        if not chunks:
            raise ValueError("Summary requires at least one transcript chunk")
        full_context = "\n".join(_context_line(item) for item in chunks)

        if estimate_tokens(full_context) <= self.settings.summary_direct_max_input_tokens:
            if on_event:
                on_event("request_sent:direct")
            result = self._generate_stream(
                model=self.summary_model,
                contents=_append_prompt(self.summary_prompt, full_context),
                system_instruction="",
                temperature=self.settings.llm_temperature_summary,
                max_output_tokens=self.settings.llm_summary_max_output_tokens,
                on_event=on_event,
            )
            return self._clean_markdown(result), "direct"

        batches = self._batch_context(chunks, self.settings.summary_map_batch_tokens)
        partials: list[str] = []
        for index, batch in enumerate(batches, start=1):
            if on_event:
                on_event(f"map:{index}/{len(batches)}")
            partials.append(
                self._generate(
                    model=self.summary_model,
                    contents=_append_prompt(
                        self.summary_prompt,
                        "\n".join(_context_line(item) for item in batch),
                    ),
                    system_instruction="",
                    temperature=self.settings.llm_temperature_summary,
                    max_output_tokens=self.settings.llm_summary_max_output_tokens,
                )
            )

        reduction_round = 0
        while estimate_tokens("\n\n".join(partials)) > self.settings.summary_direct_max_input_tokens:
            reduction_round += 1
            partial_batches = self._batch_context(
                partials, self.settings.summary_map_batch_tokens
            )
            if on_event:
                on_event(f"reduce_round:{reduction_round}/{len(partial_batches)}")
            condensed: list[str] = []
            for partial_batch in partial_batches:
                condensed.append(
                    self._generate(
                        model=self.summary_model,
                        contents=_append_prompt(
                            self.summary_prompt, "\n\n".join(partial_batch)
                        ),
                        system_instruction="",
                        temperature=self.settings.llm_temperature_summary,
                        max_output_tokens=self.settings.llm_summary_max_output_tokens,
                    )
                )
            if len(condensed) >= len(partials):
                # Avoid a non-shrinking loop if one model response itself exceeds the budget.
                partials = condensed
                break
            partials = condensed

        if on_event:
            on_event("request_sent:reduce")
        reduced = self._generate_stream(
            model=self.summary_model,
            contents=_append_prompt(self.summary_prompt, "\n\n".join(partials)),
            system_instruction="",
            temperature=self.settings.llm_temperature_summary,
            max_output_tokens=self.settings.llm_summary_max_output_tokens,
            on_event=on_event,
        )
        return self._clean_markdown(reduced), "map_reduce"

    def summarize_stream(self, chunks: Sequence, document_id: str) -> Iterator[str]:
        """Stream the final Markdown summary while keeping map steps internal."""
        if not chunks:
            raise ValueError("Summary requires at least one transcript chunk")
        full_context = "\n".join(_context_line(item) for item in chunks)
        if estimate_tokens(full_context) <= self.settings.summary_direct_max_input_tokens:
            yield from self._generate_stream_iter(
                model=self.summary_model,
                contents=_append_prompt(self.meeting_summary_prompt, full_context),
                system_instruction="",
                temperature=self.settings.llm_temperature_summary,
                max_output_tokens=self.settings.llm_summary_max_output_tokens,
            )
            return

        batches = self._batch_context(chunks, self.settings.summary_map_batch_tokens)
        partials = [
            self._generate(
                model=self.summary_model,
                contents=_append_prompt(
                    self.meeting_summary_prompt,
                    "\n".join(_context_line(item) for item in batch),
                ),
                system_instruction="",
                temperature=self.settings.llm_temperature_summary,
                max_output_tokens=self.settings.llm_summary_max_output_tokens,
            )
            for batch in batches
        ]
        while estimate_tokens("\n\n".join(partials)) > self.settings.summary_direct_max_input_tokens:
            partial_batches = self._batch_context(
                partials, self.settings.summary_map_batch_tokens
            )
            condensed = [
                self._generate(
                    model=self.summary_model,
                    contents=_append_prompt(self.meeting_summary_prompt, "\n\n".join(batch)),
                    system_instruction="",
                    temperature=self.settings.llm_temperature_summary,
                    max_output_tokens=self.settings.llm_summary_max_output_tokens,
                )
                for batch in partial_batches
            ]
            partials = condensed
            if len(condensed) == 1:
                break

        yield from self._generate_stream_iter(
            model=self.summary_model,
            contents=_append_prompt(self.meeting_summary_prompt, "\n\n".join(partials)),
            system_instruction="",
            temperature=self.settings.llm_temperature_summary,
            max_output_tokens=self.settings.llm_summary_max_output_tokens,
        )

    def analyze_meeting(
        self, chunks=None, full_text=None, meeting_id: str = "unknown"
    ) -> dict:
        if chunks:
            markdown, strategy = self.summarize(chunks, meeting_id)
        elif full_text:
            markdown, strategy = self.summarize([full_text], meeting_id)
        else:
            raise ValueError("chunks or full_text is required")
        return {"meeting_id": meeting_id, "result": markdown, "strategy": strategy}

    def _chat_contents(self, query: str, context_chunks: Sequence, history=None) -> str:
        context = "\n".join(_context_line(item) for item in context_chunks)
        history_lines: list[str] = []
        for message in (history or [])[-6:]:
            role = "User" if message.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {message.get('text', '')}")
        history_block = "\n".join(history_lines) or "(none)"
        return (
            f"Transcript evidence:\n{context}\n\n"
            f"Recent conversation:\n{history_block}\n\n"
            f"Question:\n{query}"
        )

    def chat(self, query: str, context_chunks: Sequence, conversation_history=None) -> str:
        if not context_chunks:
            return "Không tìm thấy đoạn transcript phù hợp để trả lời câu hỏi này."
        return self._generate(
            model=self.chat_model,
            contents=self._chat_contents(query, context_chunks, conversation_history),
            system_instruction=self.chat_prompt,
            temperature=self.settings.llm_temperature_chat,
            max_output_tokens=self.settings.llm_chat_max_output_tokens,
        )

    def chat_stream(
        self, query: str, context_chunks: Sequence, conversation_history=None
    ) -> Iterator[str]:
        if not context_chunks:
            yield "Không tìm thấy đoạn transcript phù hợp để trả lời câu hỏi này."
            return
        stream = self.client.models.generate_content_stream(
            model=self.chat_model,
            contents=self._chat_contents(query, context_chunks, conversation_history),
            config=self._config(
                self.chat_prompt,
                self.settings.llm_temperature_chat,
                self.settings.llm_chat_max_output_tokens,
            ),
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
