"""
Tamil TTS FastAPI Service - Supports both Coqui VITS and Indic-TTS checkpoints
Optimized for Chennai slang testing on AWS EC2
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import hashlib
import uuid
import torch
import logging
from pathlib import Path
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    load_model()
    setup_cleanup()
    yield

app = FastAPI(title="Tamil Chennai Slang TTS", version="1.0", lifespan=lifespan)

# Config
MODEL_TYPE = os.getenv("MODEL_TYPE", "coqui_vits") # coqui_vits or indic_tts
MODEL_NAME = os.getenv("MODEL_NAME", "tts_models/ta/cv/vits") # for coqui
MODEL_PATH = os.getenv("MODEL_PATH", None) # custom fine-tuned checkpoint for chennai slang
CACHE_DIR = Path("/tmp/tts_cache")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("/tmp/tts_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

tts_model = None

class TTSRequest(BaseModel):
    text: str
    language: str = "ta"
    speaker_id: str = None # for multi-speaker indic-tts
    cache: bool = True

def get_text_hash(text: str):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_model():
    global tts_model
    logger.info(f"Loading TTS model: type={MODEL_TYPE}, name={MODEL_NAME}, custom_path={MODEL_PATH}")
    
    if MODEL_TYPE == "coqui_vits":
        try:
            from TTS.api import TTS
            # Use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            if MODEL_PATH and Path(MODEL_PATH).exists():
                # Custom fine-tuned Chennai model
                logger.info(f"Loading custom Chennai model from {MODEL_PATH}")
                # For VITS, you might need config path too
                config_path = os.getenv("CONFIG_PATH", None)
                if config_path:
                    tts_model = TTS(model_path=MODEL_PATH, config_path=config_path).to(device)
                else:
                    tts_model = TTS(model_path=MODEL_PATH).to(device)
            else:
                # Pretrained Tamil from Coqui
                logger.info(f"Loading pretrained {MODEL_NAME}")
                tts_model = TTS(model_name=MODEL_NAME).to(device)
            
            logger.info("Coqui TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS: {e}")
            raise e

    elif MODEL_TYPE == "indic_tts":
        # Placeholder for Indic-TTS FastPitch + HiFiGAN
        # You need to implement as per https://github.com/AI4Bharat/Indic-TTS
        try:
            # Example pseudo-code:
            # from indic_tts.inference import FastPitchInference, HiFiGANInference
            # acoustic_model = FastPitchInference(checkpoint=... )
            # vocoder = HiFiGANInference(checkpoint=...)
            # tts_model = (acoustic_model, vocoder)
            logger.warning("Indic-TTS loading not fully implemented - please integrate from Indic-TTS repo")
            # For now fallback to coqui
            from TTS.api import TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_model = TTS(model_name="tts_models/ta/cv/vits").to(device)
        except Exception as e:
            logger.error(f"Indic-TTS load failed: {e}")
            raise e

@app.get("/health")
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "device": device,
        "gpu": gpu_name,
        "model_loaded": tts_model is not None
    }

@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    if not tts_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text empty")
    
    # Chennai slang normalization (keep slang as is, don't over-normalize)
    text = req.text.strip()
    # Optional: add custom replacements for better prosody
    # text = text.replace("da", "da,") # add small pause after dei, da
    
    text_hash = get_text_hash(text + (MODEL_PATH or "default"))
    cached_file = CACHE_DIR / f"{text_hash}.wav"
    
    if req.cache and cached_file.exists():
        logger.info(f"Cache hit for: {text[:30]}")
        return FileResponse(str(cached_file), media_type="audio/wav", filename="chennai_tts.wav")
    
    output_file = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
    
    try:
        logger.info(f"Synthesizing: {text}")
        if MODEL_TYPE == "coqui_vits":
            # Coqui VITS inference
            tts_model.tts_to_file(text=text, file_path=str(output_file))
        else:
            # Indic-TTS path - implement
            # mel = acoustic_model(text)
            # wav = vocoder(mel)
            # save wav
            tts_model.tts_to_file(text=text, file_path=str(output_file))
        
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
        "endpoints": {
            "health": "/health",
            "synthesize": "POST /synthesize with {text: 'Vanakkam da'}",
            "docs": "/docs"
        },
        "example_chennai_phrases": [
            "Vanakkam da, enna panra?",
            "Dei machan, sema scene da!",
            "Seri da, naan varen",
            "Saptiya da? Illa da innum illa"
        ]
    }

# Cleanup old files periodically
def setup_cleanup():
    import threading, time
    def cleanup():
        while True:
            time.sleep(3600) # every hour
            try:
                for f in OUTPUT_DIR.glob("*.wav"):
                    if f.stat().st_mtime < (time.time() - 3600*2): # older than 2hr
                        f.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
    threading.Thread(target=cleanup, daemon=True).start()
