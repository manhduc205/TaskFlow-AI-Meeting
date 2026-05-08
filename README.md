# 🚀 TaskFlow AI: RAG-Powered Meeting Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-00a393.svg?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)
![Gemini](https://img.shields.io/badge/Gemini-3_Flash_Preview-blueviolet.svg?logo=google)
![Whisper](https://img.shields.io/badge/Faster_Whisper-STT-yellow.svg)

**TaskFlow AI** là hệ thống Trí tuệ Nhân tạo chuyên dụng được thiết kế để tự động hóa quy trình phân tích biên bản cuộc họp. Ứng dụng công nghệ **Hybrid RAG (Retrieval-Augmented Generation)** tiên tiến, hệ thống không chỉ bóc băng (Speech-to-Text) mà còn hiểu sâu sắc ngữ cảnh, trích xuất chính xác các quyết định, thuật ngữ kỹ thuật và danh sách công việc (Tasks) được giao.

Hệ thống được thiết kế theo kiến trúc Micro-architecture, tối ưu hóa cho tiếng Việt và sẵn sàng tích hợp với các hệ thống Backend (như Spring Boot) thông qua RESTful API.

---

## 🌟 Tính Năng Nổi Bật

- 🎙️ **Tự động bóc băng (STT):** Tích hợp `faster-whisper` để chuyển đổi Video/Audio thành văn bản với độ chính xác cao.
- 🧠 **Hybrid Retrieval Engine:** Kết hợp giữa tìm kiếm ngữ nghĩa (ChromaDB Vector) và tìm kiếm từ khóa (BM25) qua thuật toán **RRF (Reciprocal Rank Fusion)** để đảm bảo không bỏ sót context.
- 🇻🇳 **Tối ưu Tiếng Việt:** Pipeline xử lý ngôn ngữ tự nhiên (NLP) với Pyvi word-segmentation và Vietnamese Stopwords filtering.
- ⏱️ **Semantic Chunking:** Cắt đoạn văn bản thông minh dựa trên độ trễ giọng nói (silence gap) và ngữ nghĩa thay vì cắt cứng theo số lượng ký tự.
- 🛡️ **Fault Tolerance & Token Budgeting:** Tích hợp cơ chế Auto-Retry (Exponential Backoff) chống sập API và tự động quản lý độ dài Token gửi cho LLM để tránh ảo giác (Hallucination).
- 🔌 **FastAPI Ready:** Cung cấp sẵn các Endpoint tốc độ cao cho ứng dụng client hoặc Gateway gọi đến.

---

## 🏗️ Kiến Trúc Hệ Thống (The 5-Phase Model)

Hệ thống được vận hành qua 5 giai đoạn cốt lõi:

### Phase 1: Core Data Modeling (`models.py` & `text_utils.py`)
- Định nghĩa cấu trúc `MeetingChunk` chuẩn hóa với `chunk_id` (UUID).
- Dữ liệu thô chạy qua pipeline NLP: Xóa ký tự thừa → Lowercase → **Pyvi Tokenization** → Xóa Stopwords.

### Phase 2: Semantic Chunking & Dual Ingestion (`rag_engine.py`)
- **Semantic Chunking:** Nhận diện các khoảng lặng (`> 2.0s`) từ log của Whisper để cắt khối văn bản sao cho giữ trọn vẹn ý nghĩa.
- **Dual Ingestion:** Dữ liệu được nạp song song vào 3 kho:
  1. `ChromaDB`: Vector Store sử dụng mô hình embedding đa ngôn ngữ.
  2. `BM25`: Keyword Store chạy in-memory để bắt chính xác các từ khóa kỹ thuật.
  3. `Metadata Cache`: Lưu trữ ánh xạ O(1) phục hồi dòng thời gian.

### Phase 3: The Hybrid Retrieval Engine (`rag_engine.py`)
- Khi có câu hỏi (Query), hệ thống truy vấn song song cả ChromaDB và BM25.
- Áp dụng công thức toán học **RRF** để chấm điểm lại danh sách.
- Kích hoạt **Context Windowing** (lấy thêm chunk trước/sau) để khôi phục ngữ cảnh giao tiếp.

### Phase 4: LLM Integration & Token Budgeting (`llm_service.py`)
- Đếm số lượng từ (Token Budget Manager). Nếu context vượt ngưỡng an toàn (ví dụ: 15,000 từ), hệ thống tự động loại bỏ các chunk kém liên quan nhất.
- Truyền Context sạch vào **Gemini 3 Flash Preview** với Prompt Engineering khắt khe, ép trả về định dạng `STRICT JSON`.

### Phase 5: System Integration (`main_api.py` & `main_poc.py`)
- Mở các API Gateway sử dụng FastAPI.
- Hỗ trợ công cụ CLI (`main_poc.py`) để các kỹ sư AI dễ dàng test thuật toán cục bộ.

---

## 📂 Cấu Trúc Thư Mục

```text
TaskFlow-AI-Meeting/
├── app/                  # Chứa logic ứng dụng (nếu mở rộng)
├── chroma_db/            # Thư mục lưu trữ Vector Database (tự sinh)
├── models/               # Thư mục chứa model AI (Whisper, Embedding)
├── test_audio/           # Chứa các file audio/video mẫu để test
├── transcript/           # Chứa file kết quả STT (.txt, .json)
├── tmp_uploads/          # Thư mục tạm lưu file upload qua API
│
├── main_poc.py           # 🚀 CLI Tool để chạy test toàn bộ Pipeline
├── main_api.py           # 🚀 FastAPI Web Server
├── llm_service.py        # Logic giao tiếp với Google Gemini
├── rag_engine.py         # Trái tim Hybrid RAG (Chroma + BM25)
├── stt_engine.py         # Module Speech-to-Text (Whisper)
├── models.py             # Định nghĩa Data Classes
├── text_utils.py         # Pipeline NLP Tiếng Việt
├── requirements.txt      # Danh sách thư viện Python
└── .env                  # Cấu hình biến môi trường (API Keys)
```

---

## ⚙️ Cài Đặt

### 1. Yêu cầu hệ thống
- Python 3.10+
- (Khuyến nghị) GPU NVIDIA có hỗ trợ CUDA để chạy STT Whisper mượt mà.

### 2. Cài đặt thư viện
Tạo môi trường ảo và cài đặt các dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường
Mở file `.env` và điền API Key của bạn:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 🚀 Hướng Dẫn Sử Dụng

### CÁCH 1: Chạy qua Command Line (CLI) dùng `main_poc.py`

Công cụ CLI hỗ trợ bóc băng, phân tích và chat trực tiếp trên terminal.

**1. Chạy phân tích toàn bộ từ file Video/Audio:**
```bash
python main_poc.py --audio test_audio/meeting.mp4 --meeting_id sprint_planning
```

**2. Phân tích lại từ file văn bản có sẵn (Bỏ qua STT):**
```bash
python main_poc.py --transcript transcript/meeting.txt --meeting_id sprint_planning
```

**3. Chat với hệ thống về cuộc họp (Interactive Mode):**
```bash
python main_poc.py --mode interactive --meeting_id sprint_planning
```

---

### CÁCH 2: Khởi chạy FastAPI Server cho Spring Boot gọi đến

Khởi động server trên cổng `8000`:
```bash
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

**Các Endpoints khả dụng:**
- `GET /api/v1/health` : Kiểm tra trạng thái Server.
- `POST /api/v1/meetings/{meeting_id}/ingest` : Upload file `.txt` hoặc text thô để nạp vào DB.
- `POST /api/v1/meetings/{meeting_id}/analyze` : Ra lệnh cho LLM phân tích cuộc họp và trả về JSON chứa (Summary, Tasks, Decisions,...).
- `POST /api/v1/meetings/{meeting_id}/chat` : API Hỏi/Đáp dựa trên Context cuộc họp.
- `GET /api/v1/meetings/{meeting_id}/result` : Lấy lại kết quả phân tích (Cached, không tốn API Key).

Tài liệu API Swagger UI có sẵn tại: `http://localhost:8000/docs`

---

## 📄 License
Tài liệu và mã nguồn được phát triển riêng cho dự án **TaskFlow AI**.
