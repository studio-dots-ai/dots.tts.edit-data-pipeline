"""
Volume Normalizer Service: HTTP API for audio loudness normalization.

FastAPI service wrapping volume_normalizer_core for network access.
Same interface as volume_normalizer_func but accessible via HTTP.

Usage:
    source envs/pipe-cpu/.venv/bin/activate
    python -m tools.volume_normalizer.volume_normalizer_service --port 8006

    # Server runs on http://localhost:8006
    # API docs at http://localhost:8006/docs
"""

import io
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

# Add project root to path so tools.common is importable when running directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.common import EncodedAudio, EncodedTensor, ServiceRunner, setup_logging
from tools.common.utils import decode_audio_input, encode_numpy, cleanup_temp_files

try:
    from .core import normalize_audio, normalize_multiple, NormalizeResult
except ImportError:
    from core import normalize_audio, normalize_multiple, NormalizeResult

logger = setup_logging(__name__)

TEMP_DIR: Optional[str] = None


# -----------------------------------------
# Pydantic Models
# -----------------------------------------


class NormalizeRequest(BaseModel):
    """Request for volume normalization."""

    audio: Union[EncodedAudio, str] = Field(
        ...,
        description="Input audio. Either Base64-encoded audio object or file path string.",
    )
    target_db: float = Field(
        -20.0,
        ge=-60.0,
        le=0.0,
        description="Target loudness level in dB (default: -20 dB)",
    )
    method: str = Field(
        "rms",
        description="Normalization method: 'rms' or 'peak'",
    )
    headroom_db: float = Field(
        -1.0,
        ge=-20.0,
        le=0.0,
        description="Maximum peak level to prevent clipping (default: -1 dB)",
    )
    output_format: str = Field(
        "wav",
        description="Output format: 'wav' (audio bytes) or 'numpy' (encoded waveform array)",
    )

    @model_validator(mode="after")
    def check_values(self):
        if self.method not in ("rms", "peak"):
            raise ValueError(f"method must be 'rms' or 'peak', got '{self.method}'")
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(
                f"output_format must be 'wav' or 'numpy', got '{self.output_format}'"
            )
        return self


class NormalizeBatchRequest(BaseModel):
    """Request for batch volume normalization."""

    audios: List[Union[EncodedAudio, str]] = Field(
        ...,
        description="List of input audios",
    )
    target_db: float = Field(-20.0, ge=-60.0, le=0.0)
    method: str = Field("rms")
    headroom_db: float = Field(-1.0, ge=-20.0, le=0.0)
    output_format: str = Field("wav")


class NormalizeResponse(BaseModel):
    """Response for normalize request (for numpy output format)."""

    status: str = Field("success", description="Request status")
    output_format: str = Field(..., description="Output format returned")
    tensor: Optional[EncodedTensor] = Field(None, description="Output waveform tensor")
    sample_rate: int = Field(..., description="Audio sample rate")
    original_rms_db: float = Field(..., description="Original RMS level in dB")
    normalized_rms_db: float = Field(..., description="Normalized RMS level in dB")
    gain_db: float = Field(..., description="Applied gain in dB")
    message: Optional[str] = Field(None, description="Additional message")


class NormalizeBatchResponse(BaseModel):
    """Response for batch normalization."""

    status: str = Field("success")
    count: int = Field(..., description="Number of audios processed")
    results: List[NormalizeResponse] = Field(..., description="Individual results")


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    title="Volume Normalizer Service",
    description="Audio loudness normalization service using RMS or peak normalization.",
    version="1.0.0",
)


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "Volume Normalizer Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/normalize")
def normalize(request: NormalizeRequest):
    """
    Normalize audio loudness to target level.

    Args:
        audio: Input audio
        target_db: Target loudness in dB
        method: 'rms' or 'peak'
        headroom_db: Maximum peak level

    Returns:
        - If output_format='wav': Binary WAV file response
        - If output_format='numpy': JSON with encoded waveform array
    """
    temp_files = []

    try:
        # Decode audio input
        audio_path = decode_audio_input(request.audio, TEMP_DIR, prefix="input")
        if isinstance(request.audio, EncodedAudio):
            temp_files.append(audio_path)

        logger.info(
            f"Normalizing: target={request.target_db}dB, method={request.method}"
        )

        result: NormalizeResult = normalize_audio(
            audio=audio_path,
            target_db=request.target_db,
            method=request.method,
            headroom_db=request.headroom_db,
        )

        logger.info(
            f"Normalized: {result.original_rms:.2f}dB -> {result.normalized_rms:.2f}dB (gain={result.gain_db:.2f}dB)"
        )

        if request.output_format == "wav":
            buffer = io.BytesIO()
            sf.write(buffer, result.audio, result.sample_rate, format="wav")
            buffer.seek(0)
            wav_bytes = buffer.read()

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=normalized.wav",
                    "X-Sample-Rate": str(result.sample_rate),
                    "X-Original-RMS-dB": f"{result.original_rms:.2f}",
                    "X-Normalized-RMS-dB": f"{result.normalized_rms:.2f}",
                    "X-Gain-dB": f"{result.gain_db:.2f}",
                },
            )

        elif request.output_format == "numpy":
            return NormalizeResponse(
                status="success",
                output_format="numpy",
                tensor=encode_numpy(result.audio),
                sample_rate=result.sample_rate,
                original_rms_db=result.original_rms,
                normalized_rms_db=result.normalized_rms,
                gain_db=result.gain_db,
                message=f"Normalized waveform shape: {list(result.audio.shape)}",
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Normalization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Normalization failed: {e}")
    finally:
        cleanup_temp_files(temp_files, temp_dir=TEMP_DIR)


@app.post("/normalize_batch", response_model=NormalizeBatchResponse)
def normalize_batch(request: NormalizeBatchRequest):
    """
    Normalize multiple audio files to the same target level.
    """
    temp_files = []
    results_list = []

    try:
        # Decode all audio inputs
        audio_paths = []
        for i, audio_input in enumerate(request.audios):
            path = decode_audio_input(audio_input, TEMP_DIR, prefix=f"input_{i}")
            if isinstance(audio_input, EncodedAudio):
                temp_files.append(path)
            audio_paths.append(path)

        # Normalize all
        results = normalize_multiple(
            audios=audio_paths,
            target_db=request.target_db,
            method=request.method,
            headroom_db=request.headroom_db,
        )

        for result in results:
            if request.output_format == "numpy":
                results_list.append(
                    NormalizeResponse(
                        status="success",
                        output_format="numpy",
                        tensor=encode_numpy(result.audio),
                        sample_rate=result.sample_rate,
                        original_rms_db=result.original_rms,
                        normalized_rms_db=result.normalized_rms,
                        gain_db=result.gain_db,
                    )
                )
            else:
                # For wav format in batch, return numpy with metadata
                results_list.append(
                    NormalizeResponse(
                        status="success",
                        output_format="numpy",
                        tensor=encode_numpy(result.audio),
                        sample_rate=result.sample_rate,
                        original_rms_db=result.original_rms,
                        normalized_rms_db=result.normalized_rms,
                        gain_db=result.gain_db,
                        message="Batch mode always returns numpy format",
                    )
                )

        return NormalizeBatchResponse(
            status="success",
            count=len(results_list),
            results=results_list,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch normalization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch normalization failed: {e}")
    finally:
        cleanup_temp_files(temp_files, temp_dir=TEMP_DIR)


# -----------------------------------------
# Main Entry Point
# -----------------------------------------


def _on_startup(args):
    global TEMP_DIR
    TEMP_DIR = runner.temp_dir


runner = ServiceRunner(
    app=app,
    service_name="Volume Normalizer",
    default_port=8006,
    temp_dir_prefix="volnorm_service_",
    on_startup=_on_startup,
)

if __name__ == "__main__":
    runner.run()
