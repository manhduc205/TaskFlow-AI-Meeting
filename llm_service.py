"""
llm_service.py — Phase 4: Token Budget Manager + LLM Integration
- Token Budget: kiểm soát context window, trim chunks nếu vượt giới hạn
- Auto-Retry: Exponential Backoff cho lỗi 429 / 503
- analyze_meeting(): phân tích toàn bộ cuộc họp → RAW output
- chat(): Q&A với AI dựa trên context chunks đã retrieve
"""

import os
import time
import logging
from typing import List, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from models import MeetingChunk
from text_utils import count_words

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_NAME = "gemini-3-flash-preview"          # model production ổn định
TOKEN_BUDGET_WORDS = 15_000              # giới hạn từ gửi cho LLM
MAX_RETRIES = 3                          # số lần retry tối đa
RETRY_BASE_SECONDS = 5                   # thời gian chờ cơ bản (s)
PROMPT_PATH = os.getenv(
    "PROMPT_PATH",
    os.path.join(os.path.dirname(__file__), "summary_prompt.txt"),
)

# Lỗi có thể retry
RETRYABLE_ERRORS = (429, 503, 500)


class TokenBudgetManager:
    """
    Quản lý ngân sách token gửi cho LLM.
    Trim chunks có RRF score thấp nhất nếu context vượt giới hạn.
    """

    def __init__(self, max_words: int = TOKEN_BUDGET_WORDS):
        self.max_words = max_words

    def trim_chunks(self, chunks: List[MeetingChunk]) -> List[MeetingChunk]:
        """
        Kiểm tra tổng số từ. Nếu vượt giới hạn:
        - Cắt bỏ từ cuối danh sách (chunks có score thấp hơn)
        - Giữ nguyên thứ tự timeline
        """
        total_words = sum(count_words(c.raw_text) for c in chunks)

        if total_words <= self.max_words:
            return chunks

        logger.warning(
            f"[!] Context vượt budget: {total_words} từ > {self.max_words} từ. Đang trim..."
        )

        kept = []
        running_total = 0
        for chunk in chunks:
            w = count_words(chunk.raw_text)
            if running_total + w <= self.max_words:
                kept.append(chunk)
                running_total += w
            else:
                break

        logger.info(f"[*] Sau trim: {len(kept)}/{len(chunks)} chunks ({running_total} từ)")
        return kept


class LLMService:
    """
    Tầng tích hợp Gemini API với:
    - Fault-tolerant Auto-Retry (Exponential Backoff)
    - Token Budget Management
    - Prompt file loading
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("[!] Chưa cấu hình GEMINI_API_KEY trong file .env")

        self.client = genai.Client(api_key=api_key)
        self.model_name = MODEL_NAME
        self.budget_manager = TokenBudgetManager()
        self.prompt_text = self._load_prompt_text(PROMPT_PATH)

        print(f"[*] LLMService khởi tạo thành công (model: {self.model_name})")

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _load_prompt_text(self, path: str) -> str:
        """Load prompt template từ file, hỗ trợ nội dung dạng string có escape."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"[!] Không tìm thấy prompt file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        # Strip wrapping quotes if file stores a quoted string.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"\"", "'"}:
            raw = raw[1:-1]

        # Unescape common sequences like \n to real newlines.
        if "\\" in raw:
            raw = raw.encode("utf-8").decode("unicode_escape")

        return raw

    def _build_prompt(self, context: str, query: Optional[str] = None) -> str:
        """Kết hợp prompt template với context và câu hỏi (nếu có)."""
        prompt = self.prompt_text
        if not prompt.endswith("\n"):
            prompt += "\n"
        prompt += context
        if query:
            prompt += f"\n\nUser Query:\n{query}"
        return prompt

    def _call_api_with_retry(
        self,
        contents: str,
        system_instruction: str = "",
        temperature: float = 0.1,
        response_mime_type: str = "text/plain",
    ) -> str:
        """
        Gọi Gemini API với Auto-Retry Exponential Backoff.
        Lỗi 429 (Quota) / 503 (Overload): đợi 5s → 10s → 20s rồi retry.
        """
        wait = RETRY_BASE_SECONDS
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                        response_mime_type=response_mime_type,
                    ),
                )
                if response.text:
                    return response.text
                raise ValueError("Gemini trả về phản hồi rỗng")

            except Exception as e:
                error_str = str(e)
                is_retryable = any(
                    str(code) in error_str for code in RETRYABLE_ERRORS
                )

                if is_retryable and attempt < MAX_RETRIES:
                    logger.warning(
                        f"[!] Lỗi API (attempt {attempt}/{MAX_RETRIES}): {e}. "
                        f"Đợi {wait}s..."
                    )
                    print(
                        f"[!] Lỗi API (attempt {attempt}/{MAX_RETRIES}): {e}. "
                        f"Retry sau {wait}s..."
                    )
                    time.sleep(wait)
                    wait *= 2  # Exponential Backoff: 5 → 10 → 20
                else:
                    raise

        raise RuntimeError(f"API thất bại sau {MAX_RETRIES} lần retry")

    def _build_context_block(self, chunks: List[MeetingChunk]) -> str:
        """Build chuỗi context từ danh sách chunks, kèm timestamp."""
        lines = []
        for c in chunks:
            lines.append(c.to_context_string())
        return "\n".join(lines)

    # ============================================================
    # PUBLIC API
    # ============================================================

    def analyze_meeting(
        self,
        chunks: Optional[List[MeetingChunk]] = None,
        full_text: Optional[str] = None,
        meeting_id: str = "unknown",
    ) -> dict:
        """
        Phân tích toàn bộ cuộc họp → RAW output.

        Args:
            chunks   : Danh sách MeetingChunk (ưu tiên)
            full_text: Văn bản thô (fallback nếu không có chunks)
            meeting_id: ID cuộc họp

        Returns:
            dict với keys: meeting_id, result
        """
        print(f"[*] LLM đang phân tích meeting '{meeting_id}'...")

        # Chuẩn bị context
        if chunks:
            chunks = self.budget_manager.trim_chunks(chunks)
            context = self._build_context_block(chunks)
            print(f"[*] Phân tích từ {len(chunks)} chunks ({count_words(context)} từ)")
        elif full_text:
            context = full_text
            print(f"[*] Phân tích từ full_text ({count_words(context)} từ)")
        else:
            raise ValueError("Cần cung cấp chunks hoặc full_text")

        prompt = self._build_prompt(context)

        try:
            raw = self._call_api_with_retry(
                contents=prompt,
                system_instruction="",
                temperature=0.1,
                response_mime_type="text/plain",
            )
            print("[+] Phân tích xong.")
            return {
                "meeting_id": meeting_id,
                "result": raw,
            }

        except Exception as e:
            logger.error(f"[!] Lỗi analyze_meeting: {e}")
            return {
                "meeting_id": meeting_id,
                "result": f"Lỗi hệ thống: {str(e)}",
            }

    def chat(
        self,
        query: str,
        context_chunks: List[MeetingChunk],
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Q&A với AI dựa trên context chunks đã retrieve.

        Args:
            query             : Câu hỏi của người dùng
            context_chunks    : Chunks liên quan (từ hybrid_retrieve)
            conversation_history: Lịch sử hội thoại [{"role": "user/model", "text": "..."}]

        Returns:
            Câu trả lời dạng text
        """
        if not context_chunks:
            return "Không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cuộc họp này."

        # Trim context nếu cần
        context_chunks = self.budget_manager.trim_chunks(context_chunks)
        context = self._build_context_block(context_chunks)

        # Build history block nếu có
        if conversation_history:
            lines = []
            for msg in conversation_history[-6:]:  # Giữ 6 lượt gần nhất
                role = "Người dùng" if msg.get("role") == "user" else "AI"
                lines.append(f"{role}: {msg.get('text', '')}")
            context += "\n\nConversation History:\n" + "\n".join(lines)

        prompt = self._build_prompt(context, query=query)

        try:
            answer = self._call_api_with_retry(
                contents=prompt,
                system_instruction="",
                temperature=0.3,
                response_mime_type="text/plain",
            )
            return answer.strip()

        except Exception as e:
            logger.error(f"[!] Lỗi chat: {e}")
            return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
