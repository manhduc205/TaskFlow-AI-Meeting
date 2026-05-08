"""
rag_engine.py — Phase 2 & 3: Dual Ingestion + Hybrid Retrieval Engine
Trái tim của hệ thống RAG:
  - Semantic Chunking (timestamp-aware hoặc plain text fallback)
  - Dual Ingestion: ChromaDB (Vector) + BM25 (Keyword) + Metadata Cache
  - Hybrid Retrieval: Parallel Search → RRF Fusion → Context Windowing → Dedup
"""

import os
import re
import json
import uuid
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from models import MeetingChunk
from text_utils import normalize_text, split_into_sentences

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# CẤU HÌNH
# ============================================================
SILENCE_GAP_THRESHOLD = 2.0    # giây — ngưỡng im lặng để ngắt chunk
MAX_CHUNK_CHARS = 600           # ký tự — giới hạn an toàn mỗi chunk
TOP_K_RETRIEVAL = 20            # số kết quả lấy từ mỗi kho
TOP_K_FINAL = 15                # số chunk cuối sau RRF
RRF_K = 60                      # hằng số RRF (thường dùng 60)

# Model embedding mặc định (nhẹ, nhanh trên CPU)
# Để dùng BAAI/bge-m3 (chất lượng cao hơn, cần ~2GB RAM):
#   đổi EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class RAGEngine:
    """
    Hybrid RAG Engine với 3 kho dữ liệu song song:
      Kho 1: ChromaDB (Vector similarity search)
      Kho 2: BM25 in-memory (Keyword frequency search)
      Kho 3: Metadata Cache dict (O(1) lookup raw_text + timestamp)
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory

        print(f"[*] Khởi tạo RAGEngine (model: {EMBEDDING_MODEL})...")

        # Kho 1: ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            device="cpu",
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="meeting_chunks",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Kho 2: BM25 per meeting {meeting_id: {"bm25": BM25Okapi, "chunk_ids": [...]}}
        self._bm25_indices: Dict[str, Dict] = {}

        # Kho 3: Master Metadata Cache {chunk_id: cache_entry_dict}
        self._metadata_cache: Dict[str, Dict] = {}

        print(f"[+] RAGEngine sẵn sàng. ChromaDB: {self.collection.count()} chunks hiện có.")

    # ============================================================
    # PHASE 2A: SEMANTIC CHUNKING
    # ============================================================

    def _chunk_from_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Semantic chunking từ Whisper JSON segments (có timestamp).
        Ngắt chunk khi:
          - Silence gap giữa segment hiện tại và tiếp theo > SILENCE_GAP_THRESHOLD
          - HOẶC chunk đã vượt MAX_CHUNK_CHARS
        """
        chunks = []
        current_texts: List[str] = []
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        current_chars = 0

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))

            if current_start is None:
                current_start = start

            # Kiểm tra điều kiện ngắt
            should_break = False
            if current_texts:
                # Tính silence gap: start của segment này - end của segment trước
                prev_end = float(segments[i - 1].get("end", start)) if i > 0 else start
                silence = start - prev_end
                if silence > SILENCE_GAP_THRESHOLD:
                    should_break = True
                elif current_chars + len(text) > MAX_CHUNK_CHARS:
                    should_break = True

            if should_break and current_texts:
                chunks.append({
                    "text": " ".join(current_texts),
                    "start": current_start,
                    "end": current_end,
                })
                current_texts = []
                current_start = start
                current_chars = 0

            current_texts.append(text)
            current_end = end
            current_chars += len(text)

        # Chunk cuối cùng
        if current_texts:
            chunks.append({
                "text": " ".join(current_texts),
                "start": current_start or 0.0,
                "end": current_end or 0.0,
            })

        return chunks

    def _chunk_from_plain_text(self, text: str) -> List[Dict]:
        """
        Fallback chunking từ plain text (không có timestamp).
        Tách theo câu, gộp đến MAX_CHUNK_CHARS.
        """
        sentences = split_into_sentences(text)
        chunks = []
        current_texts: List[str] = []
        current_chars = 0

        for sent in sentences:
            if current_chars + len(sent) > MAX_CHUNK_CHARS and current_texts:
                chunks.append({
                    "text": " ".join(current_texts),
                    "start": 0.0,
                    "end": 0.0,
                })
                current_texts = []
                current_chars = 0
            current_texts.append(sent)
            current_chars += len(sent)

        if current_texts:
            chunks.append({
                "text": " ".join(current_texts),
                "start": 0.0,
                "end": 0.0,
            })

        return chunks

    # ============================================================
    # PHASE 2B: DUAL INGESTION
    # ============================================================

    def ingest(
        self,
        meeting_id: str,
        transcript_path: Optional[str] = None,
        segments_json_path: Optional[str] = None,
        raw_text: Optional[str] = None,
        force_reingest: bool = False,
    ) -> List[MeetingChunk]:
        """
        Ghi dữ liệu vào 3 kho: ChromaDB + BM25 + Metadata Cache.
        Ưu tiên đầu vào: segments_json_path > transcript_path > raw_text.
        Idempotent: bỏ qua nếu meeting đã tồn tại (trừ khi force_reingest=True).
        """
        # Idempotency check
        if not force_reingest:
            existing = self.collection.get(
                where={"meeting_id": meeting_id}, limit=1
            )
            if existing and existing["ids"]:
                print(f"[*] Meeting '{meeting_id}' đã tồn tại. Đang rebuild BM25...")
                self._rebuild_bm25_from_chroma(meeting_id)
                return []

        # Parse input → raw chunks
        raw_chunks: List[Dict] = []

        if segments_json_path and os.path.exists(segments_json_path):
            with open(segments_json_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
            raw_chunks = self._chunk_from_segments(segments)
            print(f"[*] Semantic chunking từ JSON segments: {len(raw_chunks)} chunks")

        elif transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read()
            raw_chunks = self._chunk_from_plain_text(text)
            print(f"[*] Fallback chunking từ plain text: {len(raw_chunks)} chunks")

        elif raw_text:
            raw_chunks = self._chunk_from_plain_text(raw_text)
            print(f"[*] Fallback chunking từ raw_text: {len(raw_chunks)} chunks")

        else:
            raise ValueError("Cần ít nhất một: segments_json_path, transcript_path, hoặc raw_text")

        if not raw_chunks:
            raise ValueError("Không tạo được chunks từ dữ liệu đầu vào")

        # Tạo MeetingChunk objects
        meeting_chunks: List[MeetingChunk] = []
        for i, chunk in enumerate(raw_chunks):
            mc = MeetingChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
                meeting_id=meeting_id,
                chunk_index=i,
                raw_text=chunk["text"],
                normalized_text=normalize_text(chunk["text"]),
                start_time=chunk.get("start", 0.0),
                end_time=chunk.get("end", 0.0),
            )
            meeting_chunks.append(mc)

        # Xóa dữ liệu cũ nếu force_reingest
        if force_reingest:
            old = self.collection.get(where={"meeting_id": meeting_id})
            if old and old["ids"]:
                self.collection.delete(ids=old["ids"])
                print(f"[*] Đã xóa {len(old['ids'])} chunks cũ của '{meeting_id}'")

        # === KHO 1: ChromaDB (Vector Store) ===
        print(f"[*] Vectorizing {len(meeting_chunks)} chunks → ChromaDB...")
        BATCH = 50
        for b in range(0, len(meeting_chunks), BATCH):
            batch = meeting_chunks[b:b + BATCH]
            self.collection.add(
                ids=[mc.chunk_id for mc in batch],
                documents=[f"passage: {mc.normalized_text}" for mc in batch],
                metadatas=[mc.to_chroma_metadata() for mc in batch],
            )
        print(f"[+] ChromaDB: {len(meeting_chunks)} chunks ingested.")

        # === KHO 2: BM25 (Keyword Index, in-memory) ===
        tokenized = [mc.normalized_text.split() for mc in meeting_chunks]
        bm25 = BM25Okapi(tokenized)
        self._bm25_indices[meeting_id] = {
            "bm25": bm25,
            "chunk_ids": [mc.chunk_id for mc in meeting_chunks],
        }
        print(f"[+] BM25: Index built cho {len(meeting_chunks)} chunks.")

        # === KHO 3: Metadata Cache (O(1) lookup) ===
        for mc in meeting_chunks:
            self._metadata_cache[mc.chunk_id] = mc.to_cache_entry()
        print(f"[+] Metadata Cache: {len(meeting_chunks)} entries.")

        return meeting_chunks

    def _rebuild_bm25_from_chroma(self, meeting_id: str):
        """Rebuild BM25 index từ ChromaDB khi khởi động lại server."""
        if meeting_id in self._bm25_indices:
            return

        print(f"[*] Rebuilding BM25 + Cache cho '{meeting_id}'...")
        results = self.collection.get(
            where={"meeting_id": meeting_id},
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return

        combined = sorted(
            zip(results["ids"], results["documents"], results["metadatas"]),
            key=lambda x: x[2].get("chunk_index", 0),
        )

        chunk_ids = []
        tokenized = []
        for cid, doc, meta in combined:
            normalized = doc.replace("passage: ", "", 1)
            chunk_ids.append(cid)
            tokenized.append(normalized.split())
            if cid not in self._metadata_cache:
                self._metadata_cache[cid] = {
                    "raw_text": normalized,
                    "normalized_text": normalized,
                    "start_time": meta.get("start_time", 0.0),
                    "end_time": meta.get("end_time", 0.0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "meeting_id": meta.get("meeting_id", meeting_id),
                }

        self._bm25_indices[meeting_id] = {
            "bm25": BM25Okapi(tokenized),
            "chunk_ids": chunk_ids,
        }
        print(f"[+] BM25 rebuilt: {len(chunk_ids)} chunks.")

    # ============================================================
    # PHASE 3: HYBRID RETRIEVAL ENGINE
    # ============================================================

    def _vector_search(
        self, query_norm: str, meeting_id: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """Vector similarity search trong ChromaDB."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[f"query: {query_norm}"],
            n_results=min(top_k, count),
            where={"meeting_id": meeting_id},
            include=["metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        # cosine distance → similarity
        return [
            (cid, 1.0 - dist)
            for cid, dist in zip(results["ids"][0], results["distances"][0])
        ]

    def _bm25_search(
        self, query_norm: str, meeting_id: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """BM25 keyword search trong in-memory index."""
        if meeting_id not in self._bm25_indices:
            self._rebuild_bm25_from_chroma(meeting_id)
        if meeting_id not in self._bm25_indices:
            return []

        idx = self._bm25_indices[meeting_id]
        scores = idx["bm25"].get_scores(query_norm.split())
        chunk_ids = idx["chunk_ids"]

        top_i = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(chunk_ids[i], float(scores[i])) for i in top_i]

    @staticmethod
    def _rrf_fusion(
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = RRF_K,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion:
          Score_RRF(d) = 1/(k + rank_vector) + 1/(k + rank_bm25)
        """
        scores: Dict[str, float] = {}
        for rank, (cid, _) in enumerate(vector_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        for rank, (cid, _) in enumerate(bm25_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def hybrid_retrieve(
        self,
        query: str,
        meeting_id: str,
        top_k: int = TOP_K_FINAL,
    ) -> List[MeetingChunk]:
        """
        Pipeline truy vấn hybrid đầy đủ:
        1. Normalize query
        2. Song song: Vector Top-20 + BM25 Top-20
        3. RRF Fusion → Top-15
        4. Context Windowing (chunk_index ± 1)
        5. Deduplication bằng Set
        6. Sort theo timeline (chunk_index tăng dần)
        """
        # Bước 1: Normalize query
        query_norm = normalize_text(query)
        print(f"\n[*] Hybrid retrieve: '{query[:60]}...' | norm: '{query_norm[:50]}...'")

        # Bước 2: Song song hóa
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exe:
            f_vec = exe.submit(self._vector_search, query_norm, meeting_id, TOP_K_RETRIEVAL)
            f_bm25 = exe.submit(self._bm25_search, query_norm, meeting_id, TOP_K_RETRIEVAL)
            vec_res = f_vec.result()
            bm25_res = f_bm25.result()

        print(f"[*] Vector: {len(vec_res)} | BM25: {len(bm25_res)}")

        # Bước 3: RRF Fusion
        fused = self._rrf_fusion(vec_res, bm25_res)[:top_k]

        # Bước 4: Context Windowing — lấy chunk_index ± 1
        target_indices: set = set()
        for cid, _ in fused:
            if cid in self._metadata_cache:
                ci = self._metadata_cache[cid]["chunk_index"]
                target_indices.update([ci - 1, ci, ci + 1])
        target_indices = {i for i in target_indices if i >= 0}

        # Bước 5: Lọc và dedup từ cache
        seen: set = set()
        candidates = []
        for cid, cache in self._metadata_cache.items():
            if cache.get("meeting_id") != meeting_id:
                continue
            ci = cache.get("chunk_index", -1)
            if ci in target_indices and ci not in seen:
                seen.add(ci)
                candidates.append((cid, cache))

        # Bước 6: Sort theo timeline
        candidates.sort(key=lambda x: x[1]["chunk_index"])

        # Reconstruct MeetingChunk
        result = []
        for cid, cache in candidates:
            result.append(MeetingChunk(
                chunk_id=cid,
                meeting_id=cache["meeting_id"],
                chunk_index=cache["chunk_index"],
                raw_text=cache.get("raw_text", ""),
                normalized_text=cache.get("normalized_text", ""),
                start_time=cache.get("start_time", 0.0),
                end_time=cache.get("end_time", 0.0),
            ))

        print(f"[+] Kết quả cuối: {len(result)} chunks (sau RRF + Windowing + Dedup)")
        return result

    def get_all_chunks(self, meeting_id: str) -> List[MeetingChunk]:
        """Lấy toàn bộ chunks của một meeting (dùng cho phân tích tổng quan)."""
        results = self.collection.get(
            where={"meeting_id": meeting_id},
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return []

        chunks = []
        for cid, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            raw = self._metadata_cache.get(cid, {}).get(
                "raw_text", doc.replace("passage: ", "", 1)
            )
            chunks.append(MeetingChunk(
                chunk_id=cid,
                meeting_id=meta.get("meeting_id", meeting_id),
                chunk_index=meta.get("chunk_index", 0),
                raw_text=raw,
                normalized_text=doc.replace("passage: ", "", 1),
                start_time=meta.get("start_time", 0.0),
                end_time=meta.get("end_time", 0.0),
            ))

        chunks.sort(key=lambda x: x.chunk_index)
        return chunks

    def meeting_exists(self, meeting_id: str) -> bool:
        """Kiểm tra meeting đã được ingest chưa."""
        existing = self.collection.get(
            where={"meeting_id": meeting_id}, limit=1
        )
        return bool(existing and existing["ids"])

    def delete_meeting(self, meeting_id: str) -> int:
        """Xóa toàn bộ dữ liệu của một meeting."""
        existing = self.collection.get(where={"meeting_id": meeting_id})
        if not existing or not existing["ids"]:
            return 0
        self.collection.delete(ids=existing["ids"])
        count = len(existing["ids"])

        # Xóa khỏi BM25 và cache
        self._bm25_indices.pop(meeting_id, None)
        for cid in existing["ids"]:
            self._metadata_cache.pop(cid, None)

        print(f"[+] Đã xóa {count} chunks của meeting '{meeting_id}'")
        return count