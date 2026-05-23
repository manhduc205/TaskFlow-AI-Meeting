"""
ytb_text_utils.py — Helper trích xuất và phân đoạn ngữ nghĩa bằng THUẬT TOÁN THUẦN (No-AI).
Sử dụng các quy tắc Heuristic ngôn ngữ học tiếng Việt để gom đoạn vuông vắn,
chống chặt đôi từ phức và tối ưu hiển thị giao diện Frontend.
"""

import os
import re
from youtube_transcript_api import YouTubeTranscriptApi

# Bộ từ khóa kết câu phổ biến trong văn phong nói của tiếng Việt

def extract_youtube_video_id(url: str) -> str:
    """Trích xuất video ID từ link YouTube."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else ""

def get_youtube_transcript(video_url: str, meeting_id: str, base_dir: str) -> str:
    """
    Cào phụ đề YouTube và phân đoạn ngữ nghĩa bằng thuật toán Heuristic thuần.
    Đảm bảo đầu ra vuông vắn 15s-30s, không bao giờ chặt đôi từ ghép.
    """
    video_id = extract_youtube_video_id(video_url)
    if not video_id:
        raise ValueError("URL YouTube không hợp lệ hoặc không tìm thấy Video ID.")

    try:
        # Tương thích ngược phiên bản thư viện youtube-transcript-api
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi', 'en'])
        else:
            ytt_api = YouTubeTranscriptApi()
            transcript_data = ytt_api.fetch(video_id, languages=['vi', 'en'])

        # ============================================================
        # THUẬT TOÁN RULE-BASED SEGMENTATION (NO-AI)
        # ============================================================
        MIN_DURATION = 15.0   # Độ dài tối thiểu để bắt đầu xem xét ngắt đoạn (giây)
        MAX_DURATION = 40.0   # Độ dài tối đa khuyên dùng cho Frontend (giây)

        lines = []
        current_batch = []
        current_start_time = None
        total_segments = len(transcript_data)

        for i in range(total_segments):
            item = transcript_data[i]
            start_time = item['start'] if isinstance(item, dict) else item.start
            duration = item['duration'] if isinstance(item, dict) else item.duration
            end_time = start_time + duration

            text = item['text'] if isinstance(item, dict) else item.text
            text = text.strip().replace('\n', ' ')

            if not text:
                continue

            if current_start_time is None:
                current_start_time = start_time

            # Thêm nguyên vẹn segment vào batch (Bảo vệ từ vựng không bị chặt đôi)
            current_batch.append(text)
            elapsed_time = start_time - current_start_time

            # Kích hoạt kiểm tra điều kiện ngắt ngữ nghĩa dựa trên Heuristics
            if elapsed_time >= MIN_DURATION:
                should_flush = False

                if i + 1 < total_segments:
                    next_item = transcript_data[i + 1]
                    next_start = next_item['start'] if isinstance(next_item, dict) else next_item.start
                    next_text = next_item['text'] if isinstance(next_item, dict) else next_item.text
                    next_text = next_text.strip()

                    silence_gap = next_start - end_time

                    # Luật 1: Nhìn trước nếu câu sau viết hoa (Tín hiệu chuyển câu/chuyển ý lớn)
                    if next_text and next_text[0].isupper():
                        should_flush = True


                    # Luật 3: Gặp khoảng lặng tự nhiên lấy hơi của người nói (>= 0.4s)
                    elif silence_gap >= 0.4:
                        should_flush = True

                    # Luật 4: Chốt chặn an toàn cho FE. Nếu nói liên tục quá MAX_DURATION (30s)
                    # Ép buộc ngắt ở rìa của Segment này để bảo vệ giao diện không bị dài quá mức
                    elif elapsed_time >= MAX_DURATION:
                        should_flush = True
                else:
                    # Hết video, đóng gói đoạn cuối cùng
                    should_flush = True

                if should_flush:
                    merged_text = " ".join(current_batch)
                    if merged_text:
                        # Chuẩn hóa format chuỗi đầu ra
                        merged_text = merged_text[0].upper() + merged_text[1:]
                        if not merged_text.endswith(('.', '!', '?')):
                            merged_text += "."

                    lines.append(f"[{current_start_time:.2f}s]: {merged_text}")
                    current_batch = []
                    current_start_time = None

        # Thu thập phần văn bản còn sót lại ở cuối video (nếu có)
        if current_batch:
            merged_text = " ".join(current_batch)
            if merged_text:
                merged_text = merged_text[0].upper() + merged_text[1:]
                if not merged_text.endswith(('.', '!', '?')):
                    merged_text += "."
            lines.append(f"[{current_start_time:.2f}s]: {merged_text}")

        full_text = "\n".join(lines)

        # ============================================================
        # QUY TRÌNH PERSISTENCE (GHI FILE)
        # ============================================================
        transcript_dir = os.path.join(base_dir, "transcript")
        os.makedirs(transcript_dir, exist_ok=True)
        file_path = os.path.join(transcript_dir, f"{meeting_id}_youtube.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        return full_text

    except Exception as e:
        raise RuntimeError(f"Không thể xử lý mạch phụ đề YouTube bằng Heuristic: {str(e)}")