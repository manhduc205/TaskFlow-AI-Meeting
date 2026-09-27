"""Central configuration for the local transcript, summary, and chat pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, got: {raw!r}")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {value}")
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {value}")
    return value


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    llm_provider: str
    llm_summary_model: str
    llm_chat_model: str
    llm_temperature_summary: float
    llm_temperature_chat: float
    llm_timeout_seconds: int
    llm_max_retries: int
    llm_retry_base_seconds: float
    llm_summary_max_output_tokens: int
    llm_chat_max_output_tokens: int
    llm_thinking_level: str

    summary_direct_max_input_tokens: int
    summary_map_batch_tokens: int

    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_language: str
    stt_vad_filter: bool
    stt_min_silence_ms: int
    stt_beam_size: int
    stt_download_root: Path

    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    rag_chunk_target_chars: int
    rag_min_chunk_duration_seconds: float
    rag_max_chunk_duration_seconds: float
    rag_silence_gap_seconds: float
    rag_top_k_candidates: int
    rag_top_k_final: int
    rag_neighbor_window: int
    rag_rrf_k: int

    output_dir: Path
    save_debug_artifacts: bool

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = None) -> "Settings":
        dotenv_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=False)

        legacy_model = os.getenv("GEMINI_MODEL", "").strip()
        summary_model = os.getenv(
            "LLM_SUMMARY_MODEL", legacy_model or "gemini-3-flash-preview"
        ).strip()
        chat_model = os.getenv("LLM_CHAT_MODEL", summary_model).strip() or summary_model
        legacy_max_output_tokens = _env_int("LLM_MAX_OUTPUT_TOKENS", 8192, 128)
        output_dir = Path(os.getenv("OUTPUT_DIR", "summary_results"))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        download_root = Path(os.getenv("STT_DOWNLOAD_ROOT", "models"))
        if not download_root.is_absolute():
            download_root = PROJECT_ROOT / download_root

        settings = cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
            llm_summary_model=summary_model,
            llm_chat_model=chat_model,
            llm_temperature_summary=_env_float("LLM_TEMPERATURE_SUMMARY", 0.1),
            llm_temperature_chat=_env_float("LLM_TEMPERATURE_CHAT", 0.3),
            llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 0, 0),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 3, 1),
            llm_retry_base_seconds=_env_float("LLM_RETRY_BASE_SECONDS", 2.0, 0.0),
            llm_summary_max_output_tokens=_env_int(
                "LLM_SUMMARY_MAX_OUTPUT_TOKENS",
                legacy_max_output_tokens if os.getenv("LLM_MAX_OUTPUT_TOKENS") else 0,
                0,
            ),
            llm_chat_max_output_tokens=_env_int(
                "LLM_CHAT_MAX_OUTPUT_TOKENS", 2048, 128
            ),
            llm_thinking_level=os.getenv("LLM_THINKING_LEVEL", "default")
            .strip()
            .lower(),
            summary_direct_max_input_tokens=_env_int(
                "SUMMARY_DIRECT_MAX_INPUT_TOKENS", 100_000, 1_000
            ),
            summary_map_batch_tokens=_env_int("SUMMARY_MAP_BATCH_TOKENS", 50_000, 500),
            stt_model=os.getenv("STT_MODEL", "large-v3-turbo").strip(),
            stt_device=os.getenv("STT_DEVICE", "auto").strip().lower(),
            stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "auto").strip().lower(),
            stt_language=os.getenv("STT_LANGUAGE", "auto").strip().lower(),
            stt_vad_filter=_env_bool("STT_VAD_FILTER", True),
            stt_min_silence_ms=_env_int("STT_MIN_SILENCE_MS", 500, 0),
            stt_beam_size=_env_int("STT_BEAM_SIZE", 2, 1),
            stt_download_root=download_root.resolve(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                os.getenv("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
            ).strip(),
            embedding_device=os.getenv(
                "EMBEDDING_DEVICE", os.getenv("RAG_EMBEDDING_DEVICE", "auto")
            )
            .strip()
            .lower(),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32, 1),
            rag_chunk_target_chars=_env_int("RAG_CHUNK_TARGET_CHARS", 600, 100),
            rag_min_chunk_duration_seconds=_env_float(
                "RAG_MIN_CHUNK_DURATION_SECONDS", 0.0
            ),
            rag_max_chunk_duration_seconds=_env_float(
                "RAG_MAX_CHUNK_DURATION_SECONDS", 86400.0, 1.0
            ),
            rag_silence_gap_seconds=_env_float("RAG_SILENCE_GAP_SECONDS", 2.0),
            rag_top_k_candidates=_env_int("RAG_TOP_K_CANDIDATES", 12, 1),
            rag_top_k_final=_env_int("RAG_TOP_K_FINAL", 6, 1),
            rag_neighbor_window=_env_int("RAG_NEIGHBOR_WINDOW", 1),
            rag_rrf_k=_env_int("RAG_RRF_K", 60, 1),
            output_dir=output_dir.resolve(),
            save_debug_artifacts=_env_bool("SAVE_DEBUG_ARTIFACTS", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.llm_provider != "gemini":
            raise ValueError("Local pipeline currently supports LLM_PROVIDER=gemini only")
        if not self.llm_summary_model or not self.llm_chat_model:
            raise ValueError("LLM_SUMMARY_MODEL and LLM_CHAT_MODEL cannot be empty")
        if self.llm_thinking_level not in {
            "default",
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "LLM_THINKING_LEVEL must be default, minimal, low, medium, or high"
            )
        if self.stt_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("STT_DEVICE must be auto, cpu, or cuda")
        if self.embedding_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("EMBEDDING_DEVICE must be auto, cpu, or cuda")
        if self.rag_min_chunk_duration_seconds > self.rag_max_chunk_duration_seconds:
            raise ValueError(
                "RAG_MIN_CHUNK_DURATION_SECONDS cannot exceed "
                "RAG_MAX_CHUNK_DURATION_SECONDS"
            )
        if self.summary_map_batch_tokens > self.summary_direct_max_input_tokens:
            raise ValueError(
                "SUMMARY_MAP_BATCH_TOKENS cannot exceed "
                "SUMMARY_DIRECT_MAX_INPUT_TOKENS"
            )

    def require_gemini_key(self) -> str:
        if not self.gemini_api_key or self.gemini_api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY is missing or still contains the placeholder")
        return self.gemini_api_key
