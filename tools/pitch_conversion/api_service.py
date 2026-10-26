"""
Pitch Conversion Service: Unified HTTP API for pitch shifting.

Supports four methods via a single ``/shift`` endpoint:
  - "psola" (pure):           Praat PSOLA overlap-add resynthesis
  - "psola+qwen3_codec" (default): PSOLA shift → Qwen3-TTS 12Hz encode→decode
  - "psola+indextts2_cco":   PSOLA shift → IndexTTS2 CCO resynthesis
  - "psola+mimo_codec":       PSOLA shift → MiMo-Audio RVQ tokenize→resyn

The PSOLA step runs locally (CPU-only, parselmouth).  Codec refinement
calls external GPU services over HTTP to remove artifacts introduced by
extreme pitch shifts.

Usage::

    source envs/pipe-cpu/.venv/bin/activate
    python -m tools.pitch_conversion.api_service --port 8027

    # Server runs on http://localhost:8027
    # API docs at http://localhost:8027/docs
"""

import base64
import io
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

try:
    from .core import shift_audio, PitchShiftResult, ALL_METHODS
except ImportError:
    from core import shift_audio, PitchShiftResult, ALL_METHODS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEMP_DIR: Optional[str] = None

# External service URLs (set at startup via CLI args)
QWEN3TTS_URL: str = "http://localhost:8007"
INDEXTTS_URL: str = "http://localhost:8002"
MIMO_URL: str = "http://localhost:8024"

DEFAULT_METHOD = "psola+qwen3_codec"


# -----------------------------------------
# Pydantic Models
# -----------------------------------------


class EncodedAudio(BaseModel):
    """Base64-encoded audio bytes with format metadata."""

    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format: 'wav', 'mp3', 'flac', etc.")


class EncodedTensor(BaseModel):
    """Base64-encoded numpy array with metadata."""

    data: str = Field(..., description="Base64-encoded tensor bytes")
    shape: List[int] = Field(..., description="Tensor shape")
    dtype: str = Field("float32", description="Numpy dtype string")


class ShiftRequest(BaseModel):
    """Request for pitch shifting.

    Required:
      - semitones: Pitch shift amount in semitones (-24 to +24).

    Optional:
      - method: "psola", "psola+qwen3_codec" (default),
        "psola+indextts2_cco", or "psola+mimo_codec".
      - audio: base64-encoded audio (required).
      - output_format: "wav" (default) or "numpy".
    """

    semitones: float = Field(
        ...,
        ge=-24.0,
        le=24.0,
        description="Pitch shift in semitones. Positive = higher, negative = lower.",
    )
    method: str = Field(
        DEFAULT_METHOD,
        description=(
            "Pitch shifting method: 'psola', 'psola+qwen3_codec' (default), "
            "'psola+indextts2_cco', or 'psola+mimo_codec'."
        ),
    )

    audio: Union[EncodedAudio, str] = Field(
        ...,
        description="Input audio: base64-encoded or file path.",
    )
    text: Optional[str] = Field(
        None,
        description="Segment transcript. Required for 'psola+indextts2_cco'.",
    )

    output_format: str = Field(
        "wav",
        description="Output format: 'wav' or 'numpy'.",
    )

    @model_validator(mode="after")
    def _validate(self):
        if self.method not in ALL_METHODS:
            raise ValueError(
                f"method must be one of {ALL_METHODS}, got '{self.method}'"
            )
        if self.output_format not in ("wav", "numpy"):
            raise ValueError("output_format must be 'wav' or 'numpy'")
        if self.method == "psola+indextts2_cco" and not (self.text or "").strip():
            raise ValueError("text is required for method='psola+indextts2_cco'")
        return self


class ShiftResponse(BaseModel):
    """JSON response (for output_format='numpy')."""

    status: str = Field("success")
    output_format: str
    tensor: Optional[EncodedTensor] = None
    sample_rate: int
    original_duration: Optional[float] = None
    shifted_duration: Optional[float] = None
    semitones: float
    method: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    default_method: str = DEFAULT_METHOD
    qwen3tts_url: str = ""
    indextts_url: str = ""
    mimo_url: str = ""


# -----------------------------------------
# Helpers
# -----------------------------------------


def decode_audio_input(
    audio_input: Union[EncodedAudio, str],
    temp_dir: str,
    prefix: str = "audio",
) -> str:
    """Decode audio input (base64 or path) to a local file path."""
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        return audio_input
    raw = base64.b64decode(audio_input.data)
    ext = audio_input.format.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    path = os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def encode_numpy(arr: np.ndarray) -> EncodedTensor:
    return EncodedTensor(
        data=base64.b64encode(arr.tobytes()).decode("utf-8"),
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


def _audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="wav")
    buf.seek(0)
    return buf.read()


# -----------------------------------------
# External service callers
# -----------------------------------------


def _qwen3_encode_tokens(audio_bytes: bytes) -> tuple:
    """Call Qwen3-TTS ``/encode_tokens`` -> (codes_np int64, token_rate)."""
    payload = {
        "audio": {
            "data": base64.b64encode(audio_bytes).decode(),
            "format": "wav",
        },
    }
    resp = requests.post(f"{QWEN3TTS_URL}/encode_tokens", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Qwen3-TTS encode_tokens failed [{resp.status_code}]: {resp.text}"
        )
    data = resp.json()
    raw = base64.b64decode(data["codes"]["data"])
    codes = np.frombuffer(raw, dtype=np.dtype(data["codes"]["dtype"])).reshape(
        data["codes"]["shape"]
    )
    return codes.astype(np.int64), float(data["token_rate_hz"])


def _qwen3_decode_tokens(codes: np.ndarray) -> tuple:
    """Call Qwen3-TTS ``/decode_tokens`` -> (sample_rate, wav_bytes)."""
    payload = {
        "codes": {
            "data": base64.b64encode(codes.astype(np.int64).tobytes()).decode(),
            "shape": list(codes.shape),
            "dtype": "int64",
        },
        "output_format": "wav",
    }
    resp = requests.post(f"{QWEN3TTS_URL}/decode_tokens", json=payload, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Qwen3-TTS decode_tokens failed [{resp.status_code}]: {resp.text}"
        )
    sr = int(resp.headers.get("X-Sample-Rate", 22050))
    return sr, resp.content


def _mimo_tokenize(audio_bytes: bytes) -> dict:
    """Call MiMo-Audio ``/tokenize`` -> codes dict (pass directly to _mimo_resyn)."""
    payload = {
        "audio": {
            "data": base64.b64encode(audio_bytes).decode(),
            "format": "wav",
        },
    }
    resp = requests.post(f"{MIMO_URL}/tokenize", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"MiMo-Audio tokenize failed [{resp.status_code}]: {resp.text}"
        )
    data = resp.json()
    return data["codes"]


def _mimo_resyn(codes: dict) -> tuple:
    """Call MiMo-Audio ``/resyn`` -> (sample_rate, wav_bytes)."""
    payload = {
        "codes": codes,
        "output_format": "wav",
    }
    resp = requests.post(f"{MIMO_URL}/resyn", json=payload, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"MiMo-Audio resyn failed [{resp.status_code}]: {resp.text}")
    sr = int(resp.headers.get("X-Sample-Rate", 16000))
    return sr, resp.content


def _indextts_extract_tokens(audio_bytes: bytes) -> tuple:
    """Call IndexTTS /extract_tokens -> (codes_np int64, token_rate)."""
    payload = {
        "audio": {
            "data": base64.b64encode(audio_bytes).decode(),
            "format": "wav",
        },
    }
    resp = requests.post(f"{INDEXTTS_URL}/extract_tokens", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"IndexTTS extract_tokens failed [{resp.status_code}]: {resp.text}"
        )
    data = resp.json()
    raw = base64.b64decode(data["codes"]["data"])
    codes = np.frombuffer(raw, dtype=np.dtype(data["codes"]["dtype"])).reshape(
        data["codes"]["shape"]
    )
    return codes.astype(np.int64), float(data["token_rate"])


def _indextts_synthesize_from_tokens_with_cco(
    codes: np.ndarray,
    text: str,
    original_audio_bytes: bytes,
    converted_audio_bytes: bytes,
) -> bytes:
    """Call IndexTTS /synthesize_from_tokens using the CCO pathway split."""
    payload = {
        "codes": {
            "data": base64.b64encode(codes.astype(np.int64).tobytes()).decode(),
            "shape": list(codes.shape),
            "dtype": "int64",
        },
        "text": text,
        "spk_audio_prompt": {
            "data": base64.b64encode(original_audio_bytes).decode(),
            "format": "wav",
        },
        "spk_cond_audio_source": {
            "data": base64.b64encode(converted_audio_bytes).decode(),
            "format": "wav",
        },
        "style_audio_source": {
            "data": base64.b64encode(converted_audio_bytes).decode(),
            "format": "wav",
        },
        "output_format": "wav",
    }
    resp = requests.post(
        f"{INDEXTTS_URL}/synthesize_from_tokens", json=payload, timeout=300
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"IndexTTS synthesize_from_tokens failed [{resp.status_code}]: {resp.text}"
        )
    return resp.content


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    title="Pitch Conversion Service",
    description=(
        "Pitch shifting via Praat PSOLA with optional codec refinement.  "
        "Methods: psola (pure), psola+qwen3_codec (default), "
        "psola+indextts2_cco, psola+mimo_codec."
    ),
    version="1.0.0",
)


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "Pitch Conversion Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        qwen3tts_url=QWEN3TTS_URL,
        indextts_url=INDEXTTS_URL,
        mimo_url=MIMO_URL,
    )


@app.post("/shift")
def shift(request: ShiftRequest):
    """
    Pitch-shift endpoint.

    - **psola**: Pure Praat PSOLA overlap-add.  CPU-only, no external deps.
    - **psola+qwen3_codec** (default): PSOLA then Qwen3-TTS 12Hz
      encode->decode to clean up artifacts.
    - **psola+indextts2_cco**: PSOLA then IndexTTS2 CCO resynthesis.
    - **psola+mimo_codec**: PSOLA then MiMo-Audio tokenize->resyn.

    Returns WAV bytes (default) or JSON with numpy array.
    """
    tmp: List[str] = []
    try:
        if request.method == "psola":
            return _handle_psola(request, tmp)
        elif request.method == "psola+qwen3_codec":
            return _handle_psola_qwen3(request, tmp)
        elif request.method == "psola+indextts2_cco":
            return _handle_psola_indextts2_cco(request, tmp)
        elif request.method == "psola+mimo_codec":
            return _handle_psola_mimo(request, tmp)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown method: {request.method}"
            )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Shift failed (method={request.method}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(tmp)


# -----------------------------------------
# Method handlers
# -----------------------------------------


def _handle_psola(req: ShiftRequest, tmp: List[str]):
    """Pure PSOLA pitch shift — no refinement."""
    audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="psola_in")
    if isinstance(req.audio, EncodedAudio):
        tmp.append(audio_path)

    result: PitchShiftResult = shift_audio(audio_path, req.semitones, temp_dir=TEMP_DIR)
    logger.info(
        f"PSOLA: {result.original_duration:.2f}s -> {result.shifted_duration:.2f}s "
        f"(semitones={req.semitones:+.1f})"
    )
    return _format_audio_response(
        result.audio,
        result.sample_rate,
        req,
        original_duration=result.original_duration,
        shifted_duration=result.shifted_duration,
    )


def _handle_psola_qwen3(req: ShiftRequest, tmp: List[str]):
    """PSOLA pitch shift followed by Qwen3-TTS 12Hz codec roundtrip."""
    # Step 1: PSOLA pitch shift
    audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="pq_in")
    if isinstance(req.audio, EncodedAudio):
        tmp.append(audio_path)

    result: PitchShiftResult = shift_audio(audio_path, req.semitones, temp_dir=TEMP_DIR)
    logger.info(
        f"PSOLA+Qwen3 step 1 (PSOLA): {result.original_duration:.2f}s -> "
        f"{result.shifted_duration:.2f}s (semitones={req.semitones:+.1f})"
    )

    # Step 2: Qwen3-TTS codec roundtrip on the PSOLA output
    psola_wav_bytes = _audio_to_wav_bytes(result.audio, result.sample_rate)

    codes, token_rate = _qwen3_encode_tokens(psola_wav_bytes)
    logger.info(f"PSOLA+Qwen3 step 2 (encode): {codes.shape} codes at {token_rate}Hz")

    sr, refined_wav_bytes = _qwen3_decode_tokens(codes)
    logger.info(f"PSOLA+Qwen3 step 2 (decode): sr={sr}")

    if req.output_format == "wav":
        return Response(
            content=refined_wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=shifted.wav",
                "X-Sample-Rate": str(sr),
                "X-Semitones": str(req.semitones),
                "X-Method": "psola+qwen3_codec",
                "X-Original-Duration": f"{result.original_duration:.3f}",
                "X-Shifted-Duration": f"{result.shifted_duration:.3f}",
            },
        )

    # numpy output
    audio_arr, sr2 = sf.read(io.BytesIO(refined_wav_bytes))
    return ShiftResponse(
        output_format="numpy",
        tensor=encode_numpy(audio_arr.astype(np.float32)),
        sample_rate=sr2,
        original_duration=result.original_duration,
        shifted_duration=result.shifted_duration,
        semitones=req.semitones,
        method="psola+qwen3_codec",
        message=f"PSOLA -> Qwen3 codec ({codes.shape[0]} tokens @ {token_rate}Hz)",
    )


def _handle_psola_indextts2_cco(req: ShiftRequest, tmp: List[str]):
    """PSOLA pitch shift followed by IndexTTS2 CCO resynthesis."""
    audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="picco_in")
    if isinstance(req.audio, EncodedAudio):
        tmp.append(audio_path)

    result: PitchShiftResult = shift_audio(audio_path, req.semitones, temp_dir=TEMP_DIR)
    logger.info(
        f"PSOLA+IndexTTS2 CCO step 1 (PSOLA): {result.original_duration:.2f}s -> "
        f"{result.shifted_duration:.2f}s (semitones={req.semitones:+.1f})"
    )

    with open(audio_path, "rb") as f:
        original_wav_bytes = f.read()
    psola_wav_bytes = _audio_to_wav_bytes(result.audio, result.sample_rate)

    codes, _ = _indextts_extract_tokens(psola_wav_bytes)
    text = req.text or ""
    refined_wav_bytes = _indextts_synthesize_from_tokens_with_cco(
        codes=codes,
        text=text,
        original_audio_bytes=original_wav_bytes,
        converted_audio_bytes=psola_wav_bytes,
    )

    if req.output_format == "wav":
        return Response(
            content=refined_wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=shifted.wav",
                "X-Method": "psola+indextts2_cco",
                "X-Semitones": str(req.semitones),
                "X-Original-Duration": f"{result.original_duration:.3f}",
                "X-Shifted-Duration": f"{result.shifted_duration:.3f}",
            },
        )

    audio_arr, sr2 = sf.read(io.BytesIO(refined_wav_bytes))
    return ShiftResponse(
        output_format="numpy",
        tensor=encode_numpy(audio_arr.astype(np.float32)),
        sample_rate=sr2,
        original_duration=result.original_duration,
        shifted_duration=result.shifted_duration,
        semitones=req.semitones,
        method="psola+indextts2_cco",
        message="PSOLA -> IndexTTS2 CCO",
    )


def _handle_psola_mimo(req: ShiftRequest, tmp: List[str]):
    """PSOLA pitch shift followed by MiMo-Audio codec roundtrip."""
    # Step 1: PSOLA pitch shift
    audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="pm_in")
    if isinstance(req.audio, EncodedAudio):
        tmp.append(audio_path)

    result: PitchShiftResult = shift_audio(audio_path, req.semitones, temp_dir=TEMP_DIR)
    logger.info(
        f"PSOLA+MiMo step 1 (PSOLA): {result.original_duration:.2f}s -> "
        f"{result.shifted_duration:.2f}s (semitones={req.semitones:+.1f})"
    )

    # Step 2: MiMo-Audio codec roundtrip on the PSOLA output
    psola_wav_bytes = _audio_to_wav_bytes(result.audio, result.sample_rate)

    codes = _mimo_tokenize(psola_wav_bytes)
    logger.info(f"PSOLA+MiMo step 2 (tokenize): codes received")

    sr, refined_wav_bytes = _mimo_resyn(codes)
    logger.info(f"PSOLA+MiMo step 2 (resyn): sr={sr}")

    if req.output_format == "wav":
        return Response(
            content=refined_wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=shifted.wav",
                "X-Sample-Rate": str(sr),
                "X-Semitones": str(req.semitones),
                "X-Method": "psola+mimo_codec",
                "X-Original-Duration": f"{result.original_duration:.3f}",
                "X-Shifted-Duration": f"{result.shifted_duration:.3f}",
            },
        )

    # numpy output
    audio_arr, sr2 = sf.read(io.BytesIO(refined_wav_bytes))
    return ShiftResponse(
        output_format="numpy",
        tensor=encode_numpy(audio_arr.astype(np.float32)),
        sample_rate=sr2,
        original_duration=result.original_duration,
        shifted_duration=result.shifted_duration,
        semitones=req.semitones,
        method="psola+mimo_codec",
        message="PSOLA -> MiMo codec",
    )


# -----------------------------------------
# Shared response helper
# -----------------------------------------


def _format_audio_response(
    audio: np.ndarray,
    sr: int,
    req: ShiftRequest,
    *,
    original_duration: float = None,
    shifted_duration: float = None,
):
    if req.output_format == "wav":
        wav_bytes = _audio_to_wav_bytes(audio, sr)
        headers = {
            "Content-Disposition": "attachment; filename=shifted.wav",
            "X-Sample-Rate": str(sr),
            "X-Semitones": str(req.semitones),
            "X-Method": req.method,
        }
        if original_duration is not None:
            headers["X-Original-Duration"] = f"{original_duration:.3f}"
        if shifted_duration is not None:
            headers["X-Shifted-Duration"] = f"{shifted_duration:.3f}"
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    return ShiftResponse(
        output_format="numpy",
        tensor=encode_numpy(audio.astype(np.float32)),
        sample_rate=sr,
        original_duration=original_duration,
        shifted_duration=shifted_duration,
        semitones=req.semitones,
        method=req.method,
    )


# -----------------------------------------
# Main
# -----------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pitch Conversion Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8027)
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--qwen3tts-url",
        default="http://localhost:8007",
        help="Qwen3-TTS service URL",
    )
    parser.add_argument(
        "--indextts-url",
        default="http://localhost:8002",
        help="IndexTTS service URL",
    )
    parser.add_argument(
        "--mimo-url",
        default="http://localhost:8024",
        help="MiMo-Audio service URL",
    )

    args = parser.parse_args()

    global TEMP_DIR, QWEN3TTS_URL, INDEXTTS_URL, MIMO_URL
    TEMP_DIR = tempfile.mkdtemp(prefix="pitch_service_")
    QWEN3TTS_URL = args.qwen3tts_url.rstrip("/")
    INDEXTTS_URL = args.indextts_url.rstrip("/")
    MIMO_URL = args.mimo_url.rstrip("/")

    logger.info(f"Temp directory: {TEMP_DIR}")
    logger.info(f"Qwen3-TTS URL: {QWEN3TTS_URL}")
    logger.info(f"IndexTTS URL: {INDEXTTS_URL}")
    logger.info(f"MiMo-Audio URL: {MIMO_URL}")
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
