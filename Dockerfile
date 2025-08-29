# Simple production image
FROM python:3.11-slim

# System deps (ffmpeg required by faster-whisper/yt-dlp)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

# Env defaults
ENV MODEL_SIZE=small
ENV COMPUTE_TYPE=int8

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
