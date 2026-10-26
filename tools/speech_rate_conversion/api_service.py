"""
Speech Rate Conversion Service: Unified HTTP API for time stretching.

Supports five methods via a single ``/stretch`` endpoint:
  - "auto" (default): picks best method based on rate
      * 0.95–1.05x → wsola
      * 0.80–1.25x → indextts2
      * otherwise  → wsola+indextts2
  - "wsola":            SoX WSOLA time stretching (signal-level)
  - "indextts2":        IndexTTS2 50Hz token extract → stretch → re-synthesis
  - "qwen3_tts":        Qwen3-TTS 12Hz token encode → stretch → decode
  - "wsola+indextts2":  WSOLA stretch + IndexTTS2 re-synthesis at rate=1.0

For token-based methods, callers may pass pre-extracted ``codes`` +
``token_rate`` instead of audio to avoid redundant token extraction.

Usage::

    source envs/pipe-cpu/.venv/bin/activate
    python -m tools.speech_rate_conversion.api_service --port 8004

    # Server runs on http://localhost:8004
    # API docs at http://localhost:8004/docs
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
    from .core import stretch_audio, stretch_tokens, StretchResult, TokenStretchResult
except ImportError:
    from core import stretch_audio, stretch_tokens, StretchResult, TokenStretchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEMP_DIR: Optional[str] = None

# External service URLs (set at startup via CLI args)
INDEXTTS_URL: str = "http://localhost:8002"
QWEN3TTS_URL: str = "http://localhost:8007"

METHODS = ("wsola", "indextts2", "qwen3_tts", "wsola+indextts2", "auto")


# -----------------------------------------
# Auto method selection strategy
# -----------------------------------------

# Rate intervals used for automatic method selection.
#   0.95x–1.05x : WSOLA alone (near-identity, no audible artifacts)
#   0.80x–1.25x : IndexTTS2-only (token stretch + re-synthesis, good quality)
#   outside      : WSOLA first, then IndexTTS2 re-synthesis at 1.0x (combined)
_AUTO_WSOLA_LO = 0.95
_AUTO_WSOLA_HI = 1.05
_AUTO_TOKEN_LO = 0.80
_AUTO_TOKEN_HI = 1.25


def resolve_auto_method(rate: float) -> str:
    """Choose the best backend based on the speed factor.

    Selection strategy:
      - Near-identity (0.95–1.05x): WSOLA is sufficient and fastest.
      - Moderate (0.80–1.25x excl. above): IndexTTS2 token stretch gives the
        best naturalness without significant WER degradation.
      - Extreme (<0.80x or >1.25x): WSOLA for the time-stretch (preserves
        content at extreme rates) followed by IndexTTS2 re-synthesis at
        rate=1.0 to recover naturalness.
    """
    if _AUTO_WSOLA_LO <= rate <= _AUTO_WSOLA_HI:
        return "wsola"
    elif _AUTO_TOKEN_LO <= rate <= _AUTO_TOKEN_HI:
        return "indextts2"
    else:
        return "wsola+indextts2"


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


class StretchRequest(BaseModel):
    """Unified request for time stretching (all methods).

    Always required:
      - rate: speed factor (>1 = faster/shorter, <1 = slower/longer)
      - method: stretching method

    Supported methods:
      - ``"wsola"``: SoX WSOLA signal-level stretching
      - ``"indextts2"``: IndexTTS2 50Hz token stretch + re-synthesis
      - ``"qwen3_tts"``: Qwen3-TTS 12Hz token stretch + decode
      - ``"wsola+indextts2"``: WSOLA stretch first, then IndexTTS2
        re-synthesis at rate=1.0 to recover naturalness
      - ``"auto"`` (default): automatically selects the best method based
        on the rate:
          * 0.95–1.05x → wsola
          * 0.80–1.25x → indextts2
          * otherwise  → wsola+indextts2

    Audio input (required for wsola; required for token methods when *codes*
    is not supplied):
      - audio: base64-encoded audio or file path

    Pre-extracted tokens (optional shortcut for indextts2 / qwen3_tts):
      - codes: encoded tensor of discrete speech tokens
      - token_rate: token rate in Hz (50 for indextts2, 12 for qwen3_tts)

    Extra fields needed by **indextts2**, **wsola+indextts2**, and **auto**
    (when auto may resolve to those methods):
      - text: text content (for the GPT forward pass)
      - spk_audio: speaker reference audio
    """

    rate: float = Field(
        ...,
        gt=0.1,
        lt=10.0,
        description="Speed rate factor. >1.0 = faster, <1.0 = slower.",
    )
    method: str = Field(
        "auto",
        description=(
            "Stretching method: 'auto', 'wsola', 'indextts2', "
            "'qwen3_tts', or 'wsola+indextts2'."
        ),
    )

    # -- Audio input --
    audio: Optional[Union[EncodedAudio, str]] = Field(
        None,
        description="Input audio: base64-encoded or file path.",
    )

    # -- Pre-extracted token input --
    codes: Optional[EncodedTensor] = Field(
        None,
        description="Pre-extracted discrete tokens (skip extraction).",
    )
    token_rate: Optional[float] = Field(
        None,
        description="Token rate in Hz.  Required when codes supplied.",
    )

    # -- Extra fields for indextts2 / wsola+indextts2 / auto --
    text: Optional[str] = Field(
        None,
        description="Text content.  Required for indextts2/auto/combined.",
    )
    spk_audio: Optional[Union[EncodedAudio, str]] = Field(
        None,
        description="Speaker reference audio.  Required for indextts2/auto/combined.",
    )

    output_format: str = Field(
        "wav",
        description="Output format: 'wav' or 'numpy'.",
    )

    @model_validator(mode="after")
    def _validate(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got '{self.method}'")
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(f"output_format must be 'wav' or 'numpy'")

        # Resolve 'auto' early so we can validate the required fields.
        effective = self.method
        if effective == "auto":
            effective = resolve_auto_method(self.rate)

        if effective == "wsola":
            if self.audio is None:
                raise ValueError("audio is required for method='wsola'.")

        if effective in ("indextts2", "qwen3_tts"):
            if self.audio is None and self.codes is None:
                raise ValueError(f"audio or codes required for method='{effective}'.")
            if self.codes is not None and self.token_rate is None:
                raise ValueError("token_rate required when codes supplied.")

        if effective in ("indextts2", "wsola+indextts2"):
            if self.text is None:
                raise ValueError(
                    f"text is required for method='{self.method}' "
                    f"(resolved to '{effective}')."
                )
            if self.spk_audio is None:
                raise ValueError(
                    f"spk_audio is required for method='{self.method}' "
                    f"(resolved to '{effective}')."
                )

        if effective == "wsola+indextts2":
            if self.audio is None:
                raise ValueError(
                    f"audio is required for method='{self.method}' "
                    f"(resolved to '{effective}')."
                )

        return self


class StretchResponse(BaseModel):
    """JSON response (for output_format='numpy')."""

    status: str = Field("success")
    output_format: str
    tensor: Optional[EncodedTensor] = None
    sample_rate: int
    original_duration: Optional[float] = None
    stretched_duration: Optional[float] = None
    rate: float
    method: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    indextts_url: str = ""
    qwen3tts_url: str = ""


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


def decode_tensor(t: EncodedTensor) -> np.ndarray:
    raw = base64.b64decode(t.data)
    return np.frombuffer(raw, dtype=np.dtype(t.dtype)).reshape(t.shape)


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


def _indextts_extract_tokens(audio_path: str) -> tuple:
    """Call IndexTTS ``/extract_tokens`` → (codes_np int64, token_rate)."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
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


def _indextts_synthesize_from_tokens(
    codes: np.ndarray,
    text: str,
    spk_audio_path: str,
) -> tuple:
    """Call IndexTTS ``/synthesize_from_tokens`` → (sample_rate, wav_bytes)."""
    with open(spk_audio_path, "rb") as f:
        spk_bytes = f.read()
    payload = {
        "codes": {
            "data": base64.b64encode(codes.astype(np.int64).tobytes()).decode(),
            "shape": list(codes.shape),
            "dtype": "int64",
        },
        "text": text,
        "spk_audio_prompt": {
            "data": base64.b64encode(spk_bytes).decode(),
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
    sr = int(resp.headers.get("X-Sample-Rate", 22050))
    return sr, resp.content


def _qwen3_encode_tokens(audio_path: str) -> tuple:
    """Call Qwen3-TTS ``/encode_tokens`` → (codes_np int64, token_rate)."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
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
    """Call Qwen3-TTS ``/decode_tokens`` → (sample_rate, wav_bytes)."""
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


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    title="Speech Rate Conversion Service",
    description=(
        "Unified speech rate conversion.  "
        "Methods: wsola (signal-level), indextts2 (50Hz tokens), qwen3_tts (12Hz tokens)."
    ),
    version="2.0.0",
)


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "Speech Rate Conversion Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        indextts_url=INDEXTTS_URL,
        qwen3tts_url=QWEN3TTS_URL,
    )


@app.post("/stretch")
def stretch(request: StretchRequest):
    """
    Unified time-stretch endpoint.

    - **auto** (default): picks the best method based on the rate.
    - **wsola**: signal-level WSOLA via SoX.  Requires ``audio``.
    - **indextts2**: 50Hz token extract → stretch → re-synthesis.
      Requires ``text`` and ``spk_audio``.  Optionally pre-supply ``codes``.
    - **qwen3_tts**: 12Hz token encode → stretch → decode.
      Optionally pre-supply ``codes``.
    - **wsola+indextts2**: WSOLA stretch first, then IndexTTS2 re-synthesis
      at rate=1.0 to recover naturalness.

    Returns WAV bytes (default) or JSON with numpy array.
    """
    # Resolve 'auto' to a concrete method — mutate request.method so that
    # downstream handlers emit the resolved method name in response headers.
    if request.method == "auto":
        resolved = resolve_auto_method(request.rate)
        logger.info(f"Auto method resolved: rate={request.rate} → {resolved}")
        request.method = resolved

    tmp: List[str] = []
    try:
        if request.method == "wsola":
            return _handle_wsola(request, tmp)
        elif request.method == "indextts2":
            return _handle_indextts2(request, tmp)
        elif request.method == "qwen3_tts":
            return _handle_qwen3_tts(request, tmp)
        elif request.method == "wsola+indextts2":
            return _handle_wsola_plus_indextts2(request, tmp)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown method: {request.method}"
            )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Stretch failed (method={request.method}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(tmp)


# -----------------------------------------
# Method handlers
# -----------------------------------------


def _handle_wsola(req: StretchRequest, tmp: List[str]):
    audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="wsola_in")
    if isinstance(req.audio, EncodedAudio):
        tmp.append(audio_path)

    result: StretchResult = stretch_audio(audio_path, req.rate, temp_dir=TEMP_DIR)
    logger.info(
        f"WSOLA: {result.original_duration:.2f}s → {result.stretched_duration:.2f}s "
        f"(rate={req.rate})"
    )
    return _format_audio_response(
        result.audio,
        result.sample_rate,
        req,
        original_duration=result.original_duration,
        stretched_duration=result.stretched_duration,
    )


def _handle_indextts2(req: StretchRequest, tmp: List[str]):
    # 1. Get codes
    if req.codes is not None:
        codes = decode_tensor(req.codes).astype(np.int64)
        token_rate = req.token_rate
    else:
        audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="idx_in")
        if isinstance(req.audio, EncodedAudio):
            tmp.append(audio_path)
        codes, token_rate = _indextts_extract_tokens(audio_path)

    # 2. Stretch tokens
    tr: TokenStretchResult = stretch_tokens(codes, req.rate, token_rate)
    logger.info(
        f"IndexTTS2 token stretch: {tr.original_num_tokens} → "
        f"{tr.stretched_num_tokens} (rate={req.rate})"
    )

    # 3. Re-synthesis
    spk_path = decode_audio_input(req.spk_audio, TEMP_DIR, prefix="idx_spk")
    if isinstance(req.spk_audio, EncodedAudio):
        tmp.append(spk_path)

    sr, wav_bytes = _indextts_synthesize_from_tokens(
        tr.stretched_codes,
        req.text,
        spk_path,
    )

    if req.output_format == "wav":
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=stretched.wav",
                "X-Sample-Rate": str(sr),
                "X-Rate": str(req.rate),
                "X-Method": "indextts2",
            },
        )
    audio_arr, sr2 = sf.read(io.BytesIO(wav_bytes))
    return StretchResponse(
        output_format="numpy",
        tensor=encode_numpy(audio_arr.astype(np.float32)),
        sample_rate=sr2,
        rate=req.rate,
        method="indextts2",
        message=f"Tokens {tr.original_num_tokens}→{tr.stretched_num_tokens}",
    )


def _handle_qwen3_tts(req: StretchRequest, tmp: List[str]):
    # 1. Get codes
    if req.codes is not None:
        codes = decode_tensor(req.codes).astype(np.int64)
        token_rate = req.token_rate
    else:
        audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="q3_in")
        if isinstance(req.audio, EncodedAudio):
            tmp.append(audio_path)
        codes, token_rate = _qwen3_encode_tokens(audio_path)

    # 2. Stretch tokens
    tr: TokenStretchResult = stretch_tokens(codes, req.rate, token_rate)
    logger.info(
        f"Qwen3-TTS token stretch: {tr.original_num_tokens} → "
        f"{tr.stretched_num_tokens} (rate={req.rate})"
    )

    # 3. Decode tokens
    sr, wav_bytes = _qwen3_decode_tokens(tr.stretched_codes)

    if req.output_format == "wav":
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=stretched.wav",
                "X-Sample-Rate": str(sr),
                "X-Rate": str(req.rate),
                "X-Method": "qwen3_tts",
            },
        )
    audio_arr, sr2 = sf.read(io.BytesIO(wav_bytes))
    return StretchResponse(
        output_format="numpy",
        tensor=encode_numpy(audio_arr.astype(np.float32)),
        sample_rate=sr2,
        rate=req.rate,
        method="qwen3_tts",
        message=f"Tokens {tr.original_num_tokens}→{tr.stretched_num_tokens}",
    )


def _handle_wsola_plus_indextts2(req: StretchRequest, tmp: List[str]):
    """Combined method: WSOLA stretch → IndexTTS2 re-synthesis at rate=1.0.

    Step 1: Apply WSOLA time stretching at the requested rate.
    Step 2: Extract tokens from the WSOLA output, then re-synthesize via
            IndexTTS2 at rate=1.0 (identity stretch) using the *original*
            audio as speaker reference. This recovers naturalness lost by
            signal-level manipulation.
    """
    # Step 1: WSOLA stretch
    audio_path = decode_audio_input(req.audio, TEMP_DIR, prefix="comb_in")
    if isinstance(req.audio, EncodedAudio):
        tmp.append(audio_path)

    wsola_result: StretchResult = stretch_audio(audio_path, req.rate, temp_dir=TEMP_DIR)
    logger.info(
        f"Combined step 1 (WSOLA): {wsola_result.original_duration:.2f}s → "
        f"{wsola_result.stretched_duration:.2f}s (rate={req.rate})"
    )

    # Write WSOLA output to temp file for IndexTTS2 token extraction
    wsola_wav_bytes = _audio_to_wav_bytes(wsola_result.audio, wsola_result.sample_rate)
    wsola_path = os.path.join(TEMP_DIR, f"comb_wsola_{uuid.uuid4().hex}.wav")
    with open(wsola_path, "wb") as f:
        f.write(wsola_wav_bytes)
    tmp.append(wsola_path)

    # Step 2: IndexTTS2 re-synthesis at rate=1.0
    # Extract tokens from the WSOLA-stretched audio
    codes, token_rate = _indextts_extract_tokens(wsola_path)

    # Stretch tokens at rate=1.0 (identity — keeps same token count)
    tr: TokenStretchResult = stretch_tokens(codes, rate=1.0, token_rate=token_rate)
    logger.info(
        f"Combined step 2 (IndexTTS2 re-synth): {tr.original_num_tokens} tokens, "
        f"rate=1.0 (identity)"
    )

    # Re-synthesize using the *original* audio as speaker reference
    spk_path = decode_audio_input(req.spk_audio, TEMP_DIR, prefix="comb_spk")
    if isinstance(req.spk_audio, EncodedAudio):
        tmp.append(spk_path)

    sr, wav_bytes = _indextts_synthesize_from_tokens(
        tr.stretched_codes,
        req.text,
        spk_path,
    )

    method_label = "wsola+indextts2"
    if req.output_format == "wav":
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=stretched.wav",
                "X-Sample-Rate": str(sr),
                "X-Original-Duration": f"{wsola_result.original_duration:.3f}",
                "X-Stretched-Duration": f"{wsola_result.stretched_duration:.3f}",
                "X-Rate": str(req.rate),
                "X-Method": method_label,
            },
        )
    audio_arr, sr2 = sf.read(io.BytesIO(wav_bytes))
    return StretchResponse(
        output_format="numpy",
        tensor=encode_numpy(audio_arr.astype(np.float32)),
        sample_rate=sr2,
        original_duration=wsola_result.original_duration,
        stretched_duration=wsola_result.stretched_duration,
        rate=req.rate,
        method=method_label,
        message=f"WSOLA→IndexTTS2 ({tr.original_num_tokens} tokens)",
    )


# -----------------------------------------
# Shared response helper
# -----------------------------------------


def _format_audio_response(
    audio: np.ndarray,
    sr: int,
    req: StretchRequest,
    *,
    original_duration: float = None,
    stretched_duration: float = None,
):
    if req.output_format == "wav":
        wav_bytes = _audio_to_wav_bytes(audio, sr)
        headers = {
            "Content-Disposition": "attachment; filename=stretched.wav",
            "X-Sample-Rate": str(sr),
            "X-Rate": str(req.rate),
            "X-Method": req.method,
        }
        if original_duration is not None:
            headers["X-Original-Duration"] = f"{original_duration:.3f}"
        if stretched_duration is not None:
            headers["X-Stretched-Duration"] = f"{stretched_duration:.3f}"
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    return StretchResponse(
        output_format="numpy",
        tensor=encode_numpy(audio.astype(np.float32)),
        sample_rate=sr,
        original_duration=original_duration,
        stretched_duration=stretched_duration,
        rate=req.rate,
        method=req.method,
    )


# -----------------------------------------
# Main
# -----------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Speech Rate Conversion Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--indextts-url", default="http://localhost:8002", help="IndexTTS service URL"
    )
    parser.add_argument(
        "--qwen3tts-url", default="http://localhost:8007", help="Qwen3-TTS service URL"
    )

    args = parser.parse_args()

    global TEMP_DIR, INDEXTTS_URL, QWEN3TTS_URL
    TEMP_DIR = tempfile.mkdtemp(prefix="speechrate_service_")
    INDEXTTS_URL = args.indextts_url.rstrip("/")
    QWEN3TTS_URL = args.qwen3tts_url.rstrip("/")

    logger.info(f"Temp directory: {TEMP_DIR}")
    logger.info(f"IndexTTS URL: {INDEXTTS_URL}")
    logger.info(f"Qwen3-TTS URL: {QWEN3TTS_URL}")
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
