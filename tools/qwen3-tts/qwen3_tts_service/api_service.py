"""
Qwen3-TTS HTTP Service: FastAPI endpoints for TTS, voice cloning, and token codec.

Usage:
    cd tools/qwen3-tts
    source envs/pipe-gpu/.venv/bin/activate
    python -m qwen3_tts_service.api_service --port 8007

    # API docs: http://localhost:8007/docs
"""

import base64
import io
import logging
import os
import random
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

try:
    from . import core
except ImportError:
    import core

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEMP_DIR: Optional[str] = None


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- Pydantic models -----------------------------


class EncodedAudio(BaseModel):
    """Base64-encoded audio bytes with format metadata."""

    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format: 'wav', 'mp3', 'flac', etc.")


class EncodedTensor(BaseModel):
    """Base64-encoded numpy array with metadata."""

    data: str = Field(..., description="Base64-encoded tensor bytes")
    shape: List[int] = Field(..., description="Tensor shape")
    dtype: str = Field("float32", description="Numpy dtype string")


# ---- Request / Response ----


class CustomVoiceRequest(BaseModel):
    """Request for CustomVoice synthesis (preset speakers)."""

    text: str = Field(..., description="Text to synthesize.")
    speaker: str = Field(..., description="Preset speaker name, e.g. 'Ryan', 'Vivian'.")
    language: str = Field("Auto", description="Language code or 'Auto'.")
    instruct: Optional[str] = Field(
        None, description="Emotion/style instruction, e.g. 'Very happy.'"
    )
    output_format: str = Field("wav", description="'wav' or 'numpy'.")
    # Generation parameters for deterministic inference
    seed: Optional[int] = Field(
        None, description="Random seed for reproducible inference."
    )
    do_sample: Optional[bool] = Field(
        None, description="Whether to use sampling (False=greedy)."
    )
    temperature: Optional[float] = Field(None, description="Sampling temperature.")
    top_k: Optional[int] = Field(None, description="Top-k sampling.")
    top_p: Optional[float] = Field(None, description="Top-p (nucleus) sampling.")
    # Sub-talker (codebook predictor) generation parameters
    subtalker_dosample: Optional[bool] = Field(
        None, description="Sub-talker sampling switch (False=greedy for determinism)."
    )
    subtalker_top_k: Optional[int] = Field(None, description="Sub-talker top-k.")
    subtalker_top_p: Optional[float] = Field(None, description="Sub-talker top-p.")
    subtalker_temperature: Optional[float] = Field(
        None, description="Sub-talker temperature."
    )

    @model_validator(mode="after")
    def _check(self):
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(
                f"output_format must be 'wav' or 'numpy', got '{self.output_format}'"
            )
        return self




class VoiceCloneRequest(BaseModel):
    """Request for voice-clone synthesis (Base model)."""

    text: str = Field(..., description="Text to synthesize.")
    language: str = Field("Auto", description="Language code or 'Auto'.")
    ref_audio: Union[EncodedAudio, str] = Field(
        ..., description="Reference audio: base64-encoded object or file path."
    )
    ref_text: Optional[str] = Field(
        None,
        description="Reference transcript. Required unless x_vector_only_mode=True.",
    )
    x_vector_only_mode: bool = Field(
        False, description="Use only speaker embedding (skip reference text/code)."
    )
    output_format: str = Field("wav", description="'wav' or 'numpy'.")
    # Generation parameters for deterministic inference
    seed: Optional[int] = Field(
        None, description="Random seed for reproducible inference."
    )
    do_sample: Optional[bool] = Field(
        None, description="Whether to use sampling (False=greedy)."
    )
    temperature: Optional[float] = Field(None, description="Sampling temperature.")
    top_k: Optional[int] = Field(None, description="Top-k sampling.")
    top_p: Optional[float] = Field(None, description="Top-p (nucleus) sampling.")
    # Sub-talker (codebook predictor) generation parameters
    subtalker_dosample: Optional[bool] = Field(
        None, description="Sub-talker sampling switch (False=greedy for determinism)."
    )
    subtalker_top_k: Optional[int] = Field(None, description="Sub-talker top-k.")
    subtalker_top_p: Optional[float] = Field(None, description="Sub-talker top-p.")
    subtalker_temperature: Optional[float] = Field(
        None, description="Sub-talker temperature."
    )

    @model_validator(mode="after")
    def _check(self):
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(
                f"output_format must be 'wav' or 'numpy', got '{self.output_format}'"
            )
        if not self.x_vector_only_mode and not self.ref_text:
            raise ValueError("ref_text is required when x_vector_only_mode=False.")
        return self


class TokenEncodeRequest(BaseModel):
    """Request to encode audio into 12Hz discrete tokens."""

    audio: Union[EncodedAudio, str] = Field(
        ..., description="Audio input: base64-encoded object or file path."
    )
    output_format: str = Field("numpy", description="'numpy' (default).")


class TokenDecodeRequest(BaseModel):
    """Request to decode 12Hz tokens back to audio."""

    codes: EncodedTensor = Field(
        ..., description="Token codes as encoded int64 tensor of shape (T, Q)."
    )
    output_format: str = Field("wav", description="'wav' or 'numpy'.")


class SynthesisResponse(BaseModel):
    status: str = "success"
    output_format: str
    tensor: Optional[EncodedTensor] = None
    sample_rate: int
    duration_seconds: float
    message: Optional[str] = None


class TokenEncodeResponse(BaseModel):
    status: str = "success"
    codes: EncodedTensor
    token_rate_hz: float
    duration_seconds: float
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    voice_design_loaded: bool = False
    custom_voice_loaded: bool
    base_loaded: bool
    tokenizer_loaded: bool
    info: Dict[str, Any] = {}


# ----------------------------- Utilities -----------------------------------


def decode_audio_input(
    audio_input: Union[EncodedAudio, str], prefix: str = "audio"
) -> str:
    """Decode audio input to a temporary file path."""
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        return audio_input
    audio_bytes = base64.b64decode(audio_input.data)
    ext = audio_input.format.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    path = os.path.join(TEMP_DIR, f"{prefix}_{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def encode_numpy(arr: np.ndarray) -> EncodedTensor:
    return EncodedTensor(
        data=base64.b64encode(arr.tobytes()).decode(),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def cleanup(paths: List[str]):
    for p in paths:
        try:
            if p and os.path.exists(p) and TEMP_DIR and p.startswith(TEMP_DIR):
                os.remove(p)
        except Exception:
            pass


# ----------------------------- FastAPI app ---------------------------------

app = FastAPI(
    title="Qwen3-TTS Service",
    description="TTS with preset speakers (CustomVoice), and voice "
    "cloning (Base), plus 12Hz token encode/decode.",
    version="1.0.0",
)


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"service": "Qwen3-TTS Service", "status": "running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    info = core.get_model_info()
    cv, base, tok = core.get_models()
    voice_design = None
    return HealthResponse(
        status="healthy" if (voice_design or cv or base or tok) else "not_ready",
        voice_design_loaded=voice_design is not None,
        custom_voice_loaded=cv is not None,
        base_loaded=base is not None,
        tokenizer_loaded=tok is not None,
        info=info,
    )


@app.get("/info")
async def info():
    return core.get_model_info()






# ---- CustomVoice ----


@app.post("/synthesize_custom")
def synthesize_custom(req: CustomVoiceRequest):
    """Synthesize with a preset speaker. Optionally provide `instruct` for emotion/style."""
    try:
        if req.seed is not None:
            _set_seed(req.seed)
        gen_kwargs = {}
        for k in (
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "subtalker_dosample",
            "subtalker_top_k",
            "subtalker_top_p",
            "subtalker_temperature",
        ):
            v = getattr(req, k, None)
            if v is not None:
                gen_kwargs[k] = v
        result = core.synthesize_custom_voice(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct,
            **gen_kwargs,
        )
        if req.output_format == "wav":
            buf = io.BytesIO()
            sf.write(buf, result.audio, result.sample_rate, format="wav")
            buf.seek(0)
            return Response(
                content=buf.read(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=custom_voice.wav",
                    "X-Sample-Rate": str(result.sample_rate),
                    "X-Duration-Seconds": f"{result.duration:.3f}",
                },
            )
        return SynthesisResponse(
            output_format="numpy",
            tensor=encode_numpy(result.audio),
            sample_rate=result.sample_rate,
            duration_seconds=result.duration,
        )
    except Exception as e:
        logger.error(f"CustomVoice synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---- Voice clone ----


@app.post("/synthesize_clone")
def synthesize_clone(req: VoiceCloneRequest):
    """Clone a speaker from reference audio and synthesize new text."""
    tmp = []
    try:
        ref_path = decode_audio_input(req.ref_audio, prefix="ref")
        if isinstance(req.ref_audio, EncodedAudio):
            tmp.append(ref_path)

        if req.seed is not None:
            _set_seed(req.seed)
        gen_kwargs = {}
        for k in (
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "subtalker_dosample",
            "subtalker_top_k",
            "subtalker_top_p",
            "subtalker_temperature",
        ):
            v = getattr(req, k, None)
            if v is not None:
                gen_kwargs[k] = v
        result = core.synthesize_voice_clone(
            text=req.text,
            language=req.language,
            ref_audio=ref_path,
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
            **gen_kwargs,
        )
        if req.output_format == "wav":
            buf = io.BytesIO()
            sf.write(buf, result.audio, result.sample_rate, format="wav")
            buf.seek(0)
            return Response(
                content=buf.read(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=voice_clone.wav",
                    "X-Sample-Rate": str(result.sample_rate),
                    "X-Duration-Seconds": f"{result.duration:.3f}",
                },
            )
        return SynthesisResponse(
            output_format="numpy",
            tensor=encode_numpy(result.audio),
            sample_rate=result.sample_rate,
            duration_seconds=result.duration,
        )
    except Exception as e:
        logger.error(f"Voice clone failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(tmp)


# ---- Token encode ----


@app.post("/encode_tokens")
def encode_tokens_endpoint(req: TokenEncodeRequest):
    """Encode audio into 12Hz discrete tokens."""
    tmp = []
    try:
        audio_path = decode_audio_input(req.audio, prefix="enc")
        if isinstance(req.audio, EncodedAudio):
            tmp.append(audio_path)

        result = core.encode_tokens(audio_path)
        # Return first item (single audio)
        codes = result.codes[0]
        return TokenEncodeResponse(
            codes=encode_numpy(codes.astype(np.int64)),
            token_rate_hz=result.token_rate_hz,
            duration_seconds=result.durations[0],
            message=f"Encoded tokens shape: {list(codes.shape)}",
        )
    except Exception as e:
        logger.error(f"Token encode failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(tmp)


# ---- Token decode ----


@app.post("/decode_tokens")
def decode_tokens_endpoint(req: TokenDecodeRequest):
    """Decode 12Hz tokens back to audio waveform."""
    try:
        codes_bytes = base64.b64decode(req.codes.data)
        codes = np.frombuffer(codes_bytes, dtype=req.codes.dtype).reshape(
            req.codes.shape
        )

        result = core.decode_tokens(codes)
        audio = result.audios[0]

        if req.output_format == "wav":
            buf = io.BytesIO()
            sf.write(buf, audio, result.sample_rate, format="wav")
            buf.seek(0)
            return Response(
                content=buf.read(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=decoded.wav",
                    "X-Sample-Rate": str(result.sample_rate),
                    "X-Duration-Seconds": f"{len(audio) / result.sample_rate:.3f}",
                },
            )
        return SynthesisResponse(
            output_format="numpy",
            tensor=encode_numpy(audio),
            sample_rate=result.sample_rate,
            duration_seconds=len(audio) / result.sample_rate,
        )
    except Exception as e:
        logger.error(f"Token decode failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------- Main entry ----------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS Service")
    # Server
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    # Model paths
    parser.add_argument(
        "--custom-voice-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    )
    parser.add_argument("--base-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--tokenizer-path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    # Flags
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--no-custom-voice", action="store_true", help="Skip loading CustomVoice model"
    )
    parser.add_argument(
        "--no-base", action="store_true", help="Skip loading Base model"
    )
    parser.add_argument(
        "--no-tokenizer", action="store_true", help="Skip loading tokenizer"
    )

    args = parser.parse_args()

    global TEMP_DIR
    TEMP_DIR = tempfile.mkdtemp(prefix="qwen3tts_service_")
    logger.info(f"Temp directory: {TEMP_DIR}")

    core.init_models(
        custom_voice_path=args.custom_voice_path,
        base_path=args.base_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        dtype=args.dtype,
        load_custom_voice=not args.no_custom_voice,
        load_base=not args.no_base,
        load_tokenizer=not args.no_tokenizer,
    )

    logger.info(f"Starting Qwen3-TTS service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
