"""Speaker similarity evaluation service.

FastAPI wrapper around WavLM-large fine-tuned ECAPA-TDNN model for speaker
verification. Computes cosine similarity between two audio samples.

Model: ECAPA_TDNN_SMALL (feat_dim=1024, feat_type=wavlm_large)
Checkpoint: wavlm_large_finetune.pth
Default port: 8019
"""

import argparse
import base64
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add the model directory to path so we can import ECAPA_TDNN
_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
device = None

DEFAULT_CHECKPOINT = (
    str(Path(os.environ["CHECKPOINTS_DIR"]) / "wavlm_large_finetune.pth")
    if os.environ.get("CHECKPOINTS_DIR") else None
)
TARGET_SR = 16000
MAX_SPK_SIM_AUDIO_SECONDS = 60.0
TRUNCATED_SPK_SIM_AUDIO_SECONDS = 59.9


def load_model(checkpoint: str = DEFAULT_CHECKPOINT, dev: str = "cuda"):
    global model, device
    if not checkpoint:
        raise ValueError("Set CHECKPOINTS_DIR or supply a speaker checkpoint")
    device = dev
    logger.info(f"Loading WavLM speaker verification model from {checkpoint} …")
    t0 = time.time()

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=False)
    model = model.to(device).eval()

    logger.info(
        f"Speaker verification model loaded on {device} in {time.time() - t0:.1f}s"
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class SimilarityRequest(BaseModel):
    audio_a: str = Field(..., description="Base64-encoded WAV audio (reference/prompt)")
    audio_b: str = Field(..., description="Base64-encoded WAV audio (synthesized)")


class SimilarityResponse(BaseModel):
    similarity: float = Field(..., description="Cosine similarity (-1.0 to 1.0)")
    inference_time_seconds: float


class EmbeddingRequest(BaseModel):
    audio_data: str = Field(..., description="Base64-encoded WAV audio")


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int
    inference_time_seconds: float


class BatchSimilarityRequest(BaseModel):
    audio_ref: str = Field(
        ..., description="Base64-encoded reference audio (shared across batch)"
    )
    audio_list: list[str] = Field(
        ..., description="List of base64-encoded audio files to compare"
    )


class BatchSimilarityResponse(BaseModel):
    similarities: list[float]
    count: int
    avg_similarity: float
    inference_time_seconds: float


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _decode_audio(audio_bytes: bytes) -> torch.Tensor:
    """Decode audio bytes to a 16kHz mono tensor [1, T]."""
    wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    if sr:
        duration_seconds = len(wav) / float(sr)
        max_samples = int(MAX_SPK_SIM_AUDIO_SECONDS * sr)
        if len(wav) > max_samples:
            truncated_samples = int(TRUNCATED_SPK_SIM_AUDIO_SECONDS * sr)
            logger.warning(
                "Speaker similarity input is %.2fs, exceeding the %.2fs limit; "
                "truncating to the first %.2fs.",
                duration_seconds,
                MAX_SPK_SIM_AUDIO_SECONDS,
                TRUNCATED_SPK_SIM_AUDIO_SECONDS,
            )
            wav = wav[:truncated_samples]
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()
    return wav_tensor


@torch.no_grad()
def compute_embedding(audio_bytes: bytes) -> torch.Tensor:
    """Extract speaker embedding from audio bytes."""
    wav = _decode_audio(audio_bytes).to(device)
    emb = model(wav)
    return emb


@torch.no_grad()
def compute_similarity(audio_a_bytes: bytes, audio_b_bytes: bytes) -> float:
    """Compute cosine similarity between two audio samples."""
    wav_a = _decode_audio(audio_a_bytes).to(device)
    wav_b = _decode_audio(audio_b_bytes).to(device)
    emb_a = model(wav_a)
    emb_b = model(wav_b)
    sim = F.cosine_similarity(emb_a, emb_b)
    return sim.item()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Speaker Similarity Service", lifespan=lifespan)


@app.get("/health")
@app.post("/health")
def health():
    return {
        "status": "healthy" if model is not None else "not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        audio_a = base64.b64decode(req.audio_a)
        audio_b = base64.b64decode(req.audio_b)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")
    t0 = time.time()
    try:
        sim = compute_similarity(audio_a, audio_b)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Speaker similarity inference failed")
        raise HTTPException(500, str(e))
    return SimilarityResponse(similarity=sim, inference_time_seconds=time.time() - t0)


@app.post("/similarity_batch", response_model=BatchSimilarityResponse)
def similarity_batch(req: BatchSimilarityRequest):
    """Compare a reference audio against a batch of audios."""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not req.audio_list:
        raise HTTPException(400, "Empty audio_list")
    try:
        ref_bytes = base64.b64decode(req.audio_ref)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 for ref: {e}")

    t0 = time.time()
    try:
        # Compute reference embedding once
        ref_emb = compute_embedding(ref_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Speaker similarity batch reference inference failed")
        raise HTTPException(500, str(e))

    similarities = []
    for b64 in req.audio_list:
        try:
            audio_bytes = base64.b64decode(b64)
            wav = _decode_audio(audio_bytes).to(device)
            emb = model(wav)
            sim = F.cosine_similarity(ref_emb, emb).item()
            similarities.append(sim)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to process audio in speaker similarity batch")
            raise HTTPException(500, str(e)) from e

    elapsed = time.time() - t0
    avg = sum(similarities) / len(similarities)
    return BatchSimilarityResponse(
        similarities=similarities,
        count=len(similarities),
        avg_similarity=avg,
        inference_time_seconds=elapsed,
    )


@app.post("/embedding", response_model=EmbeddingResponse)
def embedding(req: EmbeddingRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        audio_bytes = base64.b64decode(req.audio_data)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")
    t0 = time.time()
    try:
        emb = compute_embedding(audio_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Speaker embedding inference failed")
        raise HTTPException(500, str(e))
    emb_list = emb.squeeze(0).cpu().tolist()
    return EmbeddingResponse(
        embedding=emb_list,
        embedding_dim=len(emb_list),
        inference_time_seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Speaker Similarity Service")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, required=DEFAULT_CHECKPOINT is None)
    args = parser.parse_args()

    load_model(args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
