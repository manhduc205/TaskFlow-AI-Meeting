"""
text_utils.py — Phase 1: Vietnamese Text Normalization Pipeline
Chuẩn hóa văn bản tiếng Việt cho RAG: làm sạch → lowercase → pyvi → stopwords.
"""

import re
from typing import List

# ============================================================
# VIETNAMESE STOPWORDS (Từ không mang ý nghĩa cho BM25)
# ============================================================
VIETNAMESE_STOPWORDS = {
    # Thán từ, từ đệm
    "à", "ừ", "ừm", "ờ", "ư", "ơ", "ạ", "ý", "ôi", "ơi", "ối",
    "thôi", "nhé", "nha", "nhỉ", "vậy", "thế", "thật", "quá", "lắm",
    "ấy", "thưa", "dạ", "vâng", "bây", "giờ", "rồi",

    # Hư từ / quan hệ từ
    "thì", "là", "mà", "và", "của", "có", "không", "trong", "với",
    "được", "này", "đó", "các", "những", "hay", "hoặc", "nhưng",
    "vì", "nên", "để", "cho", "từ", "theo", "về", "như", "cũng",
    "đã", "sẽ", "đang", "rất", "cần", "một", "nhiều", "bởi", "tuy",
    "dù", "vẫn", "lại", "đây", "khi", "nếu", "ở", "ra", "vào",
    "lên", "xuống", "qua", "đi", "đến", "tới", "cùng", "nhau",
    "bị", "làm", "phải", "đấy", "kia", "nào", "ai", "gì", "sao",
    "đâu", "bao", "mấy", "hơn", "kém", "bằng", "hết", "chứ",
    "do", "theo", "qua", "tại", "sau", "trước", "giữa", "ngoài",
    "trên", "dưới", "sang", "chuyển", "tất", "cả", "toàn", "bộ",
    "chỉ", "mới", "đều", "luôn", "ngay", "cả",

    # Đại từ nhân xưng
    "tôi", "bạn", "anh", "chị", "em", "họ", "nó", "mình",
    "chúng", "tôi", "chúng_ta", "chúng_em",

    # Từ kết hợp phổ biến không mang nghĩa
    "bởi_vì", "tuy_nhiên", "do_đó", "vì_vậy", "thế_nên",
    "có_thể", "cần_phải", "đã_được",
}


def normalize_text(text: str) -> str:
    """
    Pipeline chuẩn hóa văn bản tiếng Việt cho RAG tìm kiếm:
    1. Xóa ký tự đặc biệt / dấu câu thừa
    2. Lowercase
    3. pyvi Word Segmentation (Công nghệ thông tin → Công_nghệ thông_tin)
    4. Lọc Stopwords tiếng Việt

    Args:
        text: Văn bản tiếng Việt thô

    Returns:
        Văn bản đã chuẩn hóa, sẵn sàng cho embedding và BM25
    """
    if not text or not text.strip():
        return ""

    # Bước 1: Xóa ký tự đặc biệt, giữ chữ cái Unicode (tiếng Việt), số, khoảng trắng
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()

    # Bước 2: Lowercase
    text = text.lower()

    # Bước 3: pyvi Word Segmentation
    try:
        from pyvi import ViTokenizer
        text = ViTokenizer.tokenize(text)
    except ImportError:
        pass  # Nếu chưa cài pyvi, bỏ qua bước này

    # Bước 4: Lọc Stopwords
    tokens = text.split()
    tokens = [
        t for t in tokens
        if t not in VIETNAMESE_STOPWORDS and len(t) > 1
    ]

    return ' '.join(tokens)


def normalize_query(query: str) -> str:
    """
    Chuẩn hóa query của người dùng trước khi tìm kiếm.
    Wrapper của normalize_text với log rõ ràng hơn.
    """
    normalized = normalize_text(query)
    return normalized


def split_into_sentences(text: str) -> List[str]:
    """
    Tách văn bản thành danh sách câu để chunking fallback.
    Hỗ trợ cả dấu câu tiếng Việt và xuống dòng.
    """
    # Tách theo dấu câu kết thúc
    sentences = re.split(r'(?<=[.!?。])\s+|\n+', text)
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) > 5:  # Bỏ qua câu quá ngắn
            result.append(s)
    return result


def count_words(text: str) -> int:
    """Đếm số từ trong văn bản (dùng cho Token Budget Manager)."""
    return len(text.split())
