"""In-memory hybrid retrieval for one local transcript session."""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from core.config import Settings
from core.contracts import TranscriptChunk
from utils.text_utils import normalize_text


class RAGSession:
    def __init__(
        self,
        chunks: list[TranscriptChunk],
        settings: Settings,
        embedder: Any | None = None,
    ):
        if not chunks:
            raise ValueError("RAGSession requires at least one chunk")
        self.chunks = sorted(chunks, key=lambda item: item.chunk_index)
        self.settings = settings
        self._embedder = embedder
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _embedding_device(self) -> str:
        configured = self.settings.embedding_device
        if configured == "auto":
            return "cuda" if self._cuda_available() else "cpu"
        if configured == "cuda" and not self._cuda_available():
            raise RuntimeError("EMBEDDING_DEVICE=cuda but CUDA is not available")
        return configured

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                self.settings.embedding_model,
                device=self._embedding_device(),
            )
        return self._embedder

    def build(self) -> "RAGSession":
        tokenized = [
            (chunk.normalized_text or normalize_text(chunk.raw_text)).split()
            for chunk in self.chunks
        ]
        self._bm25 = BM25Okapi(tokenized)

        passages = [f"passage: {chunk.raw_text}" for chunk in self.chunks]
        encoded = self._get_embedder().encode(
            passages,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._embeddings = np.asarray(encoded, dtype=np.float32)
        return self

    def _ensure_built(self) -> None:
        if self._embeddings is None or self._bm25 is None:
            self.build()

    def retrieve(self, query: str, top_k: int | None = None) -> list[TranscriptChunk]:
        if not query.strip():
            raise ValueError("Chat query cannot be empty")
        self._ensure_built()
        assert self._embeddings is not None
        assert self._bm25 is not None

        candidate_count = min(self.settings.rag_top_k_candidates, len(self.chunks))
        final_count = min(top_k or self.settings.rag_top_k_final, len(self.chunks))

        query_vector = self._get_embedder().encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector_scores = self._embeddings @ np.asarray(query_vector[0], dtype=np.float32)
        vector_indices = np.argsort(vector_scores)[::-1][:candidate_count]
        vector_rank = [
            int(index) for index in vector_indices if float(vector_scores[index]) > 0.0
        ]

        normalized_query = normalize_text(query)
        bm25_rank: list[int] = []
        if normalized_query:
            bm25_scores = self._bm25.get_scores(normalized_query.split())
            bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_count]
            bm25_rank = [
                int(index) for index in bm25_indices if float(bm25_scores[index]) > 0.0
            ]

        fused_scores: dict[int, float] = {}
        for rank, index in enumerate(vector_rank, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + 1.0 / (
                self.settings.rag_rrf_k + rank
            )
        for rank, index in enumerate(bm25_rank, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + 1.0 / (
                self.settings.rag_rrf_k + rank
            )

        selected = [
            index
            for index, _ in sorted(
                fused_scores.items(), key=lambda pair: pair[1], reverse=True
            )[:final_count]
        ]
        if not selected:
            return []

        expanded: set[int] = set()
        window = self.settings.rag_neighbor_window
        for index in selected:
            for neighbor in range(index - window, index + window + 1):
                if 0 <= neighbor < len(self.chunks):
                    expanded.add(neighbor)

        return [self.chunks[index] for index in sorted(expanded)]
