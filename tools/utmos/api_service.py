"""UTMOS speech quality evaluation service.

Lightweight FastAPI wrapper around the UTMOS22 model loaded via torch.hub.
Predicts Mean Opinion Score (MOS) for speech audio quality (1-5 scale).
"""

import base64
import fcntl
import io
import os
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path so tools.common is importable when running directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.common import ServiceRunner, setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
device = None


@contextmanager
def torch_hub_load_lock():
    """Serialize torch.hub repo/cache writes across sibling UTMOS processes."""
    hub_dir = Path(torch.hub.get_dir())
    hub_dir.mkdir(parents=True, exist_ok=True)
    lock_path = hub_dir / ".utmos_load.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def load_model(dev: str = "cuda"):
    global model, device
    device = dev
    logger.info("Loading UTMOS model via torch.hub …")
    t0 = time.time()
    with torch_hub_load_lock() as lock_path:
        logger.info("Acquired UTMOS torch.hub load lock: %s", lock_path)
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
            skip_validation=True,
        )
    model = model.to(device).eval()
    logger.info(f"UTMOS model loaded on {device} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    audio_data: str = Field(..., description="Base64-encoded audio file (wav/mp3/flac)")


class PredictResponse(BaseModel):
    score: float = Field(..., description="Predicted MOS score (1-5)")
    sample_rate: int
    duration_seconds: float
    inference_time_seconds: float


class PredictBatchRequest(BaseModel):
    audio_list: list[str] = Field(..., description="List of base64-encoded audio files")


class PredictBatchResponse(BaseModel):
    scores: list[float]
    count: int
    avg_score: float
    inference_time_seconds: float


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_single(audio_bytes: bytes) -> tuple[float, int, float]:
    """Return (score, sample_rate, duration_sec)."""
    wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
    score = model(wav_tensor, sr).item()
    return score, int(sr), float(len(wav) / sr)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="UTMOS Service", lifespan=lifespan)


@app.get("/health")
@app.post("/health")
def health():
    return {
        "status": "healthy" if model is not None else "not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        audio_bytes = base64.b64decode(req.audio_data)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")
    t0 = time.time()
    score, sr, dur = predict_single(audio_bytes)
    return PredictResponse(
        score=score,
        sample_rate=sr,
        duration_seconds=dur,
        inference_time_seconds=time.time() - t0,
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not req.audio_list:
        raise HTTPException(400, "Empty audio_list")
    t0 = time.time()
    scores = []
    for b64 in req.audio_list:
        try:
            audio_bytes = base64.b64decode(b64)
            s, _, _ = predict_single(audio_bytes)
            scores.append(s)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to process audio: {e}")
            raise HTTPException(500, str(e)) from e
    elapsed = time.time() - t0
    avg = sum(scores) / len(scores)
    return PredictBatchResponse(
        scores=scores, count=len(scores), avg_score=avg, inference_time_seconds=elapsed
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
runner = ServiceRunner(
    app=app,
    service_name="UTMOS",
    default_port=8017,
    needs_device_arg=True,
    on_startup=lambda args: load_model(args.device),
)

if __name__ == "__main__":
    runner.run()
