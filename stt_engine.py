"""
stt_engine.py — Speech-to-Text Engine (Phase 5 Update)
Bóc băng audio/video với Whisper, xuất 2 file:
  1. transcript/{basename}.txt    — Plain text (1 dòng / segment)
  2. transcript/{basename}.json   — Whisper segments với timestamp (dùng cho Semantic Chunking)

Format JSON output:
[
  {"start": 0.0, "end": 3.5, "text": "Xin chào tất cả mọi người..."},
  ...
]
"""

import os
import json
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ============================================================
# CẤU HÌNH
# Chỉnh sửa theo môi trường của bạn:
#   - Trên GPU server: DEVICE = "cuda", COMPUTE_TYPE = "int8_float16"
#   - Trên CPU local:  DEVICE = "cpu",  COMPUTE_TYPE = "int8"
# ============================================================
MODEL_SIZE = "large-v3-turbo"
DEVICE = "cuda"           # Đổi sang "cpu" nếu không có GPU
COMPUTE_TYPE = "int8_float16"  # Đổi sang "int8" nếu dùng CPU
DOWNLOAD_ROOT = "./models"    # Thư mục lưu model (mount ra ngoài Docker)

print(f"[*] Khởi tạo Whisper '{MODEL_SIZE}' trên {DEVICE}...")
t0 = time.time()

from faster_whisper import WhisperModel

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=DOWNLOAD_ROOT,
    num_workers=4,
    cpu_threads=4,
)
print(f"[+] Model tải xong trong {time.time() - t0:.2f}s.\n")

# ============================================================
# THỰC THI BÓC BĂNG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILE = os.path.join(BASE_DIR, "test_audio", "meeting.mp3")

if not os.path.exists(AUDIO_FILE):
    print(f"[!] Không tìm thấy: {AUDIO_FILE}")
    exit(1)

base_name = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcript")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

output_txt = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")
output_json = os.path.join(TRANSCRIPT_DIR, f"{base_name}.json")

print(f"[*] Bắt đầu bóc băng: {AUDIO_FILE}")
t1 = time.time()

segments_generator, info = model.transcribe(
    AUDIO_FILE,
    beam_size=2,
    language="vi",
    vad_filter=True,
    condition_on_previous_text=False,
    vad_parameters=dict(min_silence_duration_ms=500),
)

total_duration = info.duration
print(f"[*] Ngôn ngữ: '{info.language}' (tin cậy: {info.language_probability:.2f})")
print(f"[*] Tổng thời lượng: {total_duration:.2f}s")
print("-" * 55)

BAR_LEN = 50
all_segments = []

with open(output_txt, "w", encoding="utf-8") as f_txt:
    for seg in segments_generator:
        # Progress bar
        pct = min((seg.end / total_duration) * 100, 100.0)
        filled = int(BAR_LEN * pct // 100)
        bar = "█" * filled + "-" * (BAR_LEN - filled)
        print(f"\r[*] Đang bóc băng: |{bar}| {pct:6.2f}%", end="", flush=True)

        # Ghi txt
        f_txt.write(f"{seg.text}\n")
        f_txt.flush()

        # Thu thập segment cho JSON
        all_segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

print()  # Xuống dòng sau progress bar
print("-" * 55)
print(f"[+] Bóc băng xong trong {time.time() - t1:.2f}s")

# Xuất JSON segments (dùng cho Semantic Chunking có timestamp)
with open(output_json, "w", encoding="utf-8") as f_json:
    json.dump(all_segments, f_json, ensure_ascii=False, indent=2)

print(f"[+] Plain text : {output_txt}")
print(f"[+] JSON segments (có timestamp): {output_json}")
print(f"[+] Tổng số segments: {len(all_segments)}")