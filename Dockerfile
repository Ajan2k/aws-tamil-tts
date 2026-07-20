FROM python:3.10-slim

# System deps for TTS + audio
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    espeak-ng-data \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Tamil model during build to avoid runtime download (faster cold start)
# Comment this if you use custom MODEL_PATH
RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/ta/cv/vits')" || echo "Pre-download failed, will download at runtime"

COPY app.py .

# Create cache dirs
RUN mkdir -p /tmp/tts_cache /tmp/tts_outputs

EXPOSE 8000

# Environment
ENV MODEL_TYPE=coqui_vits
ENV MODEL_NAME=tts_models/ta/cv/vits
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Use 1 worker because model is heavy and GPU memory. For CPU you can use 2.
