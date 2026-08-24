# TTS 評測平台 —— 純用戶端映像檔
#
# 所有模型（各家 TTS 合成、Whisper 評分）都跑在別台主機上，這個容器只負責
# 發 HTTP 請求、計時、算分、存結果，所以完全不需要 torch，建置只要一兩分鐘。
#
# 建置：  docker compose build
# 啟動：  docker compose up -d

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) Python 套件（只有平台本身的相依）
COPY requirements-console.txt /workspace/
RUN python -m pip install --no-cache-dir -r /workspace/requirements-console.txt

# 2) 平台程式
COPY app /workspace/app

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    STORAGE_DIR=/workspace/storage \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
