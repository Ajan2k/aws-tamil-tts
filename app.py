"""
Tamil TTS FastAPI Service - Using AI4Bharat VITS Rasa
Optimized for Chennai slang testing on AWS EC2
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import hashlib
import uuid
import torch
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_ID = os.getenv("MODEL_ID", "ai4bharat/vits_rasa_13")
SPEAKER_ID = int(os.getenv("SPEAKER_ID", "18"))  # 18 = TAM_F (Tamil Female)
STYLE_ID = int(os.getenv("STYLE_ID", "4"))  # 4 = CONV (Conversational)
CACHE_DIR = Path("/tmp/tts_cache")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("/tmp/tts_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

tts_model = None
tts_tokenizer = None
device = None
sampling_rate = None


class TTSRequest(BaseModel):
    text: str
    language: str = "ta"
    speaker_id: int = None  # override default speaker
    style_id: int = None  # override default style
    cache: bool = True


def get_text_hash(text: str):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_model():
    global tts_model, tts_tokenizer, device, sampling_rate
    logger.info(f"Loading AI4Bharat VITS model: {MODEL_ID}")

    try:
        from transformers import AutoModel, AutoTokenizer

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device_name)
        logger.info(f"Using device: {device_name}")

        logger.info("Loading tokenizer...")
        tts_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        logger.info("Loading model...")
        tts_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        tts_model = tts_model.to(device_obj)
        tts_model.eval()

        device = device_obj
        sampling_rate = tts_model.config.sampling_rate

        logger.info(f"Model loaded successfully. Sampling rate: {sampling_rate}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise e


# Cleanup old files periodically
def setup_cleanup():
    import threading, time
    def cleanup():
        while True:
            time.sleep(3600)  # every hour
            try:
                for f in OUTPUT_DIR.glob("*.wav"):
                    if f.stat().st_mtime < (time.time() - 3600 * 2):  # older than 2hr
                        f.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
    threading.Thread(target=cleanup, daemon=True).start()


@asynccontextmanager
async def lifespan(app):
    load_model()
    setup_cleanup()
    yield


app = FastAPI(title="Tamil Chennai Slang TTS", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": device_name,
        "gpu": gpu_name,
        "model_loaded": tts_model is not None,
        "speaker_id": SPEAKER_ID,
        "style_id": STYLE_ID,
    }


@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    if not tts_model or not tts_tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text empty")

    text = req.text.strip()

    # Use request overrides or fall back to defaults
    spk_id = req.speaker_id if req.speaker_id is not None else SPEAKER_ID
    sty_id = req.style_id if req.style_id is not None else STYLE_ID

    text_hash = get_text_hash(text + f"_spk{spk_id}_sty{sty_id}")
    cached_file = CACHE_DIR / f"{text_hash}.wav"

    if req.cache and cached_file.exists():
        logger.info(f"Cache hit for: {text[:30]}")
        return FileResponse(str(cached_file), media_type="audio/wav", filename="chennai_tts.wav")

    output_file = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"

    try:
        logger.info(f"Synthesizing: {text}")

        # Tokenize
        inputs = tts_tokenizer(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        # Inference
        with torch.no_grad():
            outputs = tts_model(input_ids, speaker_id=spk_id, emotion_id=sty_id)

        # Extract waveform and save
        waveform = outputs.waveform.squeeze().cpu().numpy()
        sf.write(str(output_file), waveform, sampling_rate)

        # Save to cache
        if req.cache:
            import shutil
            shutil.copy(str(output_file), str(cached_file))

        return FileResponse(str(output_file), media_type="audio/wav", filename="chennai_tts.wav")

    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "Tamil Chennai Slang TTS API - vanakkam da!",
        "model": MODEL_ID,
        "endpoints": {
            "health": "/health",
            "synthesize": "POST /synthesize with {text: 'வணக்கம் டா'}",
            "docs": "/docs"
        },
        "example_chennai_phrases": [
            "வணக்கம் டா, என்ன பண்ற?",
            "டேய் மச்சான், சேமா சீன் டா!",
            "சரி டா, நான் வரேன்",
            "சாப்ட்டியா டா? இல்ல டா இன்னும் இல்ல"
        ],
        "speaker_ids": {
            "TAM_F (Tamil Female)": 18,
        },
        "style_ids": {
            "ALEXA": 0,
            "BOOK": 3,
            "CONV": 4,
            "NEWS": 10,
        }
    }
