"""
main_api.py — Phase 5: FastAPI Controller
Tầng Controller sẵn sàng cho Spring Boot gọi sang qua HTTP.

Endpoints:
  POST /api/v1/meetings/{meeting_id}/ingest   — Upload transcript & ingest
  POST /api/v1/meetings/{meeting_id}/analyze  — Phân tích toàn bộ meeting
  POST /api/v1/meetings/{meeting_id}/chat     — Q&A với AI
  GET  /api/v1/meetings/{meeting_id}/result   — Lấy kết quả đã cache
  GET  /api/v1/health                         — Health check

Chạy: uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import logging
from typing import Optional, List
import re

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_engine import RAGEngine
from llm_service import LLMService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="TaskFlow AI - Meeting Analysis API",
    description="RAG-powered meeting analysis với Hybrid Search (Vector + BM25 + RRF)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Cho phép Spring Boot gọi sang
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton instances (khởi tạo 1 lần khi server start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_RESULTS_DIR = os.path.join(BASE_DIR, "summary_results")

rag_engine = RAGEngine(persist_directory=os.path.join(BASE_DIR, "chroma_db"))
llm_service = LLMService()

# In-memory results cache {meeting_id: analysis_dict}
_results_cache: dict = {}


def _ensure_summary_dir() -> None:
    os.makedirs(SUMMARY_RESULTS_DIR, exist_ok=True)


def _sanitize_meeting_id(meeting_id: str) -> str:
    # Keep filenames stable and filesystem-safe.
    return re.sub(r"[^A-Za-z0-9_-]+", "_", meeting_id).strip("_") or "meeting"


def _result_file_path(meeting_id: str) -> str:
    safe_id = _sanitize_meeting_id(meeting_id)
    return os.path.join(SUMMARY_RESULTS_DIR, f"{safe_id}.json")


def _load_results_cache_from_disk() -> None:
    _ensure_summary_dir()
    for filename in os.listdir(SUMMARY_RESULTS_DIR):
        if not filename.lower().endswith(".json"):
            continue
        file_path = os.path.join(SUMMARY_RESULTS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meeting_id = data.get("meeting_id")
            if meeting_id:
                _results_cache[meeting_id] = data
        except Exception as exc:
            logger.warning(f"[!] Không thể load cache file '{file_path}': {exc}")


# Load cache từ disk nếu có
_load_results_cache_from_disk()
if _results_cache:
    logger.info(f"[*] Loaded {len(_results_cache)} cached results từ disk.")


def _save_result_file(meeting_id: str, result: dict) -> None:
    """Persist kết quả của từng meeting xuống disk."""
    _ensure_summary_dir()
    file_path = _result_file_path(meeting_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class IngestRequest(BaseModel):
    meeting_id: str
    raw_text: Optional[str] = None
    force_reingest: bool = False


class AnalyzeRequest(BaseModel):
    use_all_chunks: bool = True    # True = phân tích toàn bộ, False = dùng RAG


class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    context_chunks_used: int
    meeting_id: str


class IngestResponse(BaseModel):
    meeting_id: str
    chunks_created: int
    status: str
    message: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint cho Spring Boot gateway."""
    return {
        "status": "healthy",
        "service": "TaskFlow AI Meeting Analysis",
        "version": "2.0.0",
        "chroma_chunks": rag_engine.collection.count(),
        "cached_results": len(_results_cache),
    }


@app.post("/api/v1/meetings/{meeting_id}/ingest", response_model=IngestResponse)
async def ingest_meeting(
    meeting_id: str,
    file: Optional[UploadFile] = File(None),
    force_reingest: bool = Form(False),
    raw_text: Optional[str] = Form(None),
):
    """
    Upload và ingest transcript vào hệ thống RAG.

    - Chấp nhận file (.txt hoặc .json Whisper segments)
    - Hoặc raw_text dạng string trong form data
    - force_reingest=true để xử lý lại từ đầu
    """
    try:
        transcript_path = None
        segments_json_path = None
        text_content = None

        if file:
            filename = file.filename or "transcript"
            content = await file.read()
            text_decoded = content.decode("utf-8")

            # Lưu file tạm thời
            tmp_dir = os.path.join(BASE_DIR, "tmp_uploads")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"{meeting_id}_{filename}")

            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text_decoded)

            if filename.endswith(".json"):
                segments_json_path = tmp_path
            else:
                transcript_path = tmp_path

        elif raw_text:
            text_content = raw_text
        else:
            raise HTTPException(
                status_code=400,
                detail="Cần upload file hoặc cung cấp raw_text"
            )

        chunks = rag_engine.ingest(
            meeting_id=meeting_id,
            transcript_path=transcript_path,
            segments_json_path=segments_json_path,
            raw_text=text_content,
            force_reingest=force_reingest,
        )

        if not chunks and not force_reingest:
            return IngestResponse(
                meeting_id=meeting_id,
                chunks_created=0,
                status="skipped",
                message=f"Meeting '{meeting_id}' đã tồn tại. Dùng force_reingest=true để xử lý lại.",
            )

        return IngestResponse(
            meeting_id=meeting_id,
            chunks_created=len(chunks),
            status="success",
            message=f"Đã ingest thành công {len(chunks)} chunks.",
        )

    except Exception as e:
        logger.error(f"Lỗi ingest meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/meetings/{meeting_id}/analyze")
async def analyze_meeting(
    meeting_id: str,
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    Phân tích toàn bộ cuộc họp → JSON 4 trường.
    Kết quả được cache lại để tái sử dụng.
    """
    if not rag_engine.meeting_exists(meeting_id):
        raise HTTPException(
            status_code=404,
            detail=f"Meeting '{meeting_id}' chưa được ingest. Hãy gọi /ingest trước."
        )

    # Trả về cache nếu đã có
    if meeting_id in _results_cache:
        logger.info(f"[*] Cache hit cho meeting '{meeting_id}'")
        return JSONResponse(content={
            **_results_cache[meeting_id],
            "from_cache": True,
        })

    try:
        chunks = rag_engine.get_all_chunks(meeting_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Không tìm thấy chunks cho meeting này")

        result = llm_service.analyze_meeting(chunks=chunks, meeting_id=meeting_id)

        # Lưu cache
        _results_cache[meeting_id] = result
        background_tasks.add_task(_save_result_file, meeting_id, result)

        return JSONResponse(content={**result, "from_cache": False})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi analyze meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/meetings/{meeting_id}/chat", response_model=ChatResponse)
async def chat_with_meeting(
    meeting_id: str,
    request: ChatRequest,
):
    """
    Q&A với AI về nội dung cuộc họp.
    Sử dụng Hybrid RAG để tìm context phù hợp trước khi hỏi LLM.
    """
    if not rag_engine.meeting_exists(meeting_id):
        raise HTTPException(
            status_code=404,
            detail=f"Meeting '{meeting_id}' chưa được ingest."
        )

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query không được để trống")

    try:
        # Hybrid retrieval
        context_chunks = rag_engine.hybrid_retrieve(
            query=request.query,
            meeting_id=meeting_id,
        )

        # LLM Q&A
        answer = llm_service.chat(
            query=request.query,
            context_chunks=context_chunks,
            conversation_history=request.conversation_history,
        )

        return ChatResponse(
            answer=answer,
            context_chunks_used=len(context_chunks),
            meeting_id=meeting_id,
        )

    except Exception as e:
        logger.error(f"Lỗi chat meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/meetings/{meeting_id}/result")
async def get_analysis_result(meeting_id: str):
    """Lấy kết quả phân tích đã cache (không gọi LLM lại)."""
    if meeting_id in _results_cache:
        return JSONResponse(content=_results_cache[meeting_id])

    file_path = _result_file_path(meeting_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _results_cache[meeting_id] = data
            return JSONResponse(content=data)
        except Exception as e:
            logger.error(f"Lỗi đọc file kết quả cho meeting {meeting_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(
        status_code=404,
        detail=f"Chưa có kết quả phân tích cho meeting '{meeting_id}'. Hãy gọi /analyze trước."
    )


@app.delete("/api/v1/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """Xóa toàn bộ dữ liệu của một meeting."""
    deleted = rag_engine.delete_meeting(meeting_id)
    _results_cache.pop(meeting_id, None)

    file_path = _result_file_path(meeting_id)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            logger.warning(f"Không thể xóa file kết quả '{file_path}': {e}")

    return {
        "meeting_id": meeting_id,
        "deleted_chunks": deleted,
        "status": "deleted",
    }


@app.get("/api/v1/meetings/{meeting_id}/chunks")
async def get_meeting_chunks(meeting_id: str):
    """Lấy danh sách tất cả chunks của meeting (debug/preview)."""
    if not rag_engine.meeting_exists(meeting_id):
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' chưa được ingest.")

    chunks = rag_engine.get_all_chunks(meeting_id)
    return {
        "meeting_id": meeting_id,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "preview": c.raw_text[:100] + "..." if len(c.raw_text) > 100 else c.raw_text,
            }
            for c in chunks
        ],
    }


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
