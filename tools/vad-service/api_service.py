"""
VAD (Voice Activity Detection) Service: FastAPI service for silence detection.

A lightweight HTTP service for detecting speech/silence boundaries in audio.
Uses pydub's silence detection to find non-silent regions.

Usage:
    source envs/pipe-gpu/.venv/bin/activate  # or: envs/pipe-gpu/.venv/bin/python  # or base
    python api_service.py --port 8005

    # Server runs on http://localhost:8005
    # API docs at http://localhost:8005/docs

Example client usage:
    import requests
    import base64

    def encode_audio_file(path: str) -> dict:
        with open(path, "rb") as f:
            audio_bytes = f.read()
        return {
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": "wav",
        }

    response = requests.post("http://localhost:8005/detect", json={
        "audio": encode_audio_file("input.wav"),
        "min_silence_len": 300,
        "silence_thresh": -40,
    })

    result = response.json()
    # result["segments"]: list of {"start": float, "end": float} in seconds
"""

import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
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
    from .core import (
        detect_speech_segments,
        trim_silence,
        waveform_to_wav_bytes,
    )
except ImportError:
    from core import (
        detect_speech_segments,
        trim_silence,
        waveform_to_wav_bytes,
    )

logger = setup_logging(__name__)

TEMP_DIR: Optional[str] = None


class SpeechSegmentModel(BaseModel):
    """A detected speech segment."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")


class DetectRequest(BaseModel):
    """Request for VAD detection."""

    audio: Union[EncodedAudio, str] = Field(
        ...,
        description="Input audio. Either Base64-encoded audio object or file path string.",
    )

    # Detection parameters
    min_silence_len: int = Field(
        300,
        ge=50,
        description="Minimum length of silence (in ms) to consider as a break between speech segments.",
    )
    silence_thresh: int = Field(
        -40,
        description="Silence threshold in dBFS. Audio below this level is considered silence. "
        "Typical values: -40 to -50 dBFS.",
    )
    seek_step: int = Field(
        10,
        ge=1,
        description="Step size for searching in ms. Smaller = more accurate but slower.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "audio": "/path/to/audio.wav",
                "min_silence_len": 300,
                "silence_thresh": -40,
            }
        }


class DetectResponse(BaseModel):
    """Response for VAD detection."""

    status: str = Field("success", description="Request status")
    segments: List[SpeechSegmentModel] = Field(
        ..., description="Detected speech segments"
    )
    total_duration: float = Field(..., description="Total audio duration in seconds")
    speech_duration: float = Field(..., description="Total speech duration in seconds")
    silence_duration: float = Field(
        ..., description="Total silence duration in seconds"
    )
    num_segments: int = Field(..., description="Number of detected segments")


class TrimRequest(BaseModel):
    """Request for trimming silence from audio boundaries."""

    audio: Union[EncodedAudio, str] = Field(
        ...,
        description="Input audio. Either Base64-encoded audio object or file path string.",
    )

    target_start_silence: float = Field(
        0.05,
        ge=0.0,
        description="Target silence duration at start in seconds. Default 50ms.",
    )
    target_end_silence: float = Field(
        0.05,
        ge=0.0,
        description="Target silence duration at end in seconds. Default 50ms.",
    )

    # Detection parameters
    min_silence_len: int = Field(100, ge=50, description="Minimum silence length in ms")
    silence_thresh: int = Field(-40, description="Silence threshold in dBFS")

    output_format: str = Field("wav", description="Output format: 'wav' or 'numpy'")

    @model_validator(mode="after")
    def check_output_format(self):
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(
                f"output_format must be 'wav' or 'numpy', got '{self.output_format}'"
            )
        return self


class TrimResponse(BaseModel):
    """Response for trim request (for numpy output format)."""

    status: str = Field("success", description="Request status")
    output_format: str = Field(..., description="Output format returned")
    tensor: Optional[EncodedTensor] = Field(None, description="Output waveform tensor")
    sample_rate: int = Field(..., description="Audio sample rate")
    original_duration: float = Field(..., description="Original duration in seconds")
    trimmed_duration: float = Field(..., description="Trimmed duration in seconds")
    speech_start: float = Field(..., description="Speech start time in original audio")
    speech_end: float = Field(..., description="Speech end time in original audio")


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    title="VAD Service",
    description="Voice Activity Detection service using pydub silence detection.",
    version="1.0.0",
)


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "VAD Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    """
    Detect speech segments in audio.

    Returns a list of speech segments with start/end times.
    """
    temp_files = []

    try:
        # Decode audio input
        audio_path = decode_audio_input(request.audio, TEMP_DIR, prefix="input")
        if isinstance(request.audio, EncodedAudio):
            temp_files.append(audio_path)

        logger.info(f"VAD detection: {audio_path}")

        # Use core function
        result = detect_speech_segments(
            audio_path,
            min_silence_len=request.min_silence_len,
            silence_thresh=request.silence_thresh,
            seek_step=request.seek_step,
        )

        # Convert to response format
        segments = [
            SpeechSegmentModel(
                start=seg.start,
                end=seg.end,
                duration=seg.duration,
            )
            for seg in result.segments
        ]

        logger.info(
            f"Detected {result.num_segments} speech segments, "
            f"speech={result.speech_duration:.2f}s, "
            f"silence={result.silence_duration:.2f}s"
        )

        return DetectResponse(
            status="success",
            segments=segments,
            total_duration=result.total_duration,
            speech_duration=result.speech_duration,
            silence_duration=result.silence_duration,
            num_segments=result.num_segments,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"VAD detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"VAD detection failed: {e}")
    finally:
        cleanup_temp_files(temp_files, temp_dir=TEMP_DIR)


@app.post("/trim")
def trim(request: TrimRequest):
    """
    Trim silence from audio boundaries.

    Detects speech region and adjusts silence at start/end to target durations.

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

        logger.info(f"Trimming silence: {audio_path}")

        # Use core function
        result = trim_silence(
            audio_path,
            target_start_silence=request.target_start_silence,
            target_end_silence=request.target_end_silence,
            min_silence_len=request.min_silence_len,
            silence_thresh=request.silence_thresh,
            return_waveform=(request.output_format == "numpy"),
        )

        logger.info(
            f"Trimmed: {result.original_duration:.2f}s -> {result.trimmed_duration:.2f}s, "
            f"speech=[{result.speech_start:.3f}s, {result.speech_end:.3f}s]"
        )

        if request.output_format == "wav":
            wav_bytes = waveform_to_wav_bytes(result.audio, result.sample_rate)

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=trimmed.wav",
                    "X-Sample-Rate": str(result.sample_rate),
                    "X-Original-Duration": f"{result.original_duration:.3f}",
                    "X-Trimmed-Duration": f"{result.trimmed_duration:.3f}",
                    "X-Speech-Start": f"{result.speech_start:.3f}",
                    "X-Speech-End": f"{result.speech_end:.3f}",
                },
            )

        elif request.output_format == "numpy":
            return TrimResponse(
                status="success",
                output_format="numpy",
                tensor=encode_numpy(result.waveform),
                sample_rate=result.sample_rate,
                original_duration=result.original_duration,
                trimmed_duration=result.trimmed_duration,
                speech_start=result.speech_start,
                speech_end=result.speech_end,
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Trim failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Trim failed: {e}")
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
    service_name="VAD",
    default_port=8005,
    temp_dir_prefix="vad_service_",
    on_startup=_on_startup,
)

if __name__ == "__main__":
    runner.run()
