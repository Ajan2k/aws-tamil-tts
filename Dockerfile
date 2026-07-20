FROM python:3.10-slim

# System deps for audio processing
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

# HuggingFace token for gated models (pass via --build-arg HF_TOKEN=hf_xxx)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Pre-download AI4Bharat VITS model during build (faster cold start)
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('ai4bharat/vits_rasa_13', trust_remote_code=True); AutoModel.from_pretrained('ai4bharat/vits_rasa_13', trust_remote_code=True)" || echo "Pre-download failed, will download at runtime"

COPY app.py .

# Create cache dirs
RUN mkdir -p /tmp/tts_cache /tmp/tts_outputs

EXPOSE 8000

# Environment
ENV MODEL_ID=ai4bharat/vits_rasa_13
ENV SPEAKER_ID=18
ENV STYLE_ID=4
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Use 1 worker because model is heavy and GPU memory. For CPU you can use 2.
