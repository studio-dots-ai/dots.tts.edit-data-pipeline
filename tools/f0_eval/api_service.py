"""F0 evaluation HTTP service.

CPU-only FastAPI wrapper around Praat/Parselmouth pitch extraction.  The
service returns robust per-span voiced F0 summaries used by speech-editing
pipelines.
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
from typing import List, Optional, Union

import numpy as np
import parselmouth
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.common import EncodedAudio, ServiceRunner, setup_logging


logger = setup_logging(__name__)


app = FastAPI(
    title="F0 Evaluation Service",
    description="Compute per-span voiced F0 summaries with Praat/Parselmouth.",
    version="1.1.0",
)


class F0Span(BaseModel):
    start: float = Field(0.0, ge=0.0, description="Span start time in seconds")
    end: Optional[float] = Field(None, ge=0.0, description="Span end time in seconds")
    label: Optional[str] = Field(None, description="Optional span label")


class F0AnalyzeRequest(BaseModel):
    audio: Union[EncodedAudio, str] = Field(
        ..., description="Base64 audio object or local file path"
    )
    spans: Optional[List[F0Span]] = Field(
        None, description="Optional spans. Omitted means analyze the full audio."
    )
    f0_min_hz: float = Field(50.0, gt=0.0, description="Pitch floor")
    f0_max_hz: float = Field(600.0, gt=0.0, description="Pitch ceiling")
    time_step_sec: Optional[float] = Field(
        None, gt=0.0, description="Praat pitch time step"
    )


class F0SpanResult(BaseModel):
    start: float
    end: float
    duration_sec: float
    label: Optional[str] = None
    min_f0_hz: Optional[float]
    max_f0_hz: Optional[float]
    median_f0_hz: Optional[float]
    mean_f0_hz: Optional[float]
    voiced_fraction: float
    num_frames: int
    num_voiced_frames: int


class F0AnalyzeResponse(BaseModel):
    status: str
    sample_rate: int
    duration_sec: float
    spans: List[F0SpanResult]


def _model_to_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return dict(value)


def _decode_audio(audio: Union[EncodedAudio, str]) -> tuple[np.ndarray, int]:
    if isinstance(audio, str):
        wav, sr = sf.read(audio, dtype="float32", always_2d=False)
    else:
        payload = base64.b64decode(audio.data)
        wav, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)

    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def analyze_span(
    audio: np.ndarray,
    sample_rate: int,
    span: F0Span,
    *,
    f0_min_hz: float = 50.0,
    f0_max_hz: float = 600.0,
    time_step_sec: Optional[float] = None,
) -> F0SpanResult:
    """Analyze a single audio span and return voiced F0 statistics."""
    duration_sec = len(audio) / sample_rate if sample_rate > 0 else 0.0
    start = min(max(float(span.start), 0.0), duration_sec)
    end = duration_sec if span.end is None else min(max(float(span.end), start), duration_sec)
    start_sample = int(round(start * sample_rate))
    end_sample = int(round(end * sample_rate))
    segment = audio[start_sample:end_sample]
    segment_duration = max(0.0, end - start)

    # Praat requires enough samples for three periods at the requested pitch
    # floor. A shorter, otherwise valid span has no measurable F0; it is not a
    # service failure.
    minimum_samples = math.ceil(3.0 * sample_rate / f0_min_hz)
    if (
        segment.size == 0
        or segment_duration <= 0.0
        or segment.size < minimum_samples
    ):
        return F0SpanResult(
            start=start,
            end=end,
            duration_sec=segment_duration,
            label=span.label,
            min_f0_hz=None,
            max_f0_hz=None,
            median_f0_hz=None,
            mean_f0_hz=None,
            voiced_fraction=0.0,
            num_frames=0,
            num_voiced_frames=0,
        )

    sound = parselmouth.Sound(segment, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(
        time_step=time_step_sec,
        pitch_floor=f0_min_hz,
        pitch_ceiling=f0_max_hz,
    )
    freqs = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    voiced = freqs[freqs > 0.0]
    num_frames = int(freqs.size)
    num_voiced = int(voiced.size)
    voiced_fraction = float(num_voiced / num_frames) if num_frames else 0.0

    return F0SpanResult(
        start=start,
        end=end,
        duration_sec=segment_duration,
        label=span.label,
        min_f0_hz=float(np.min(voiced)) if num_voiced else None,
        max_f0_hz=float(np.max(voiced)) if num_voiced else None,
        median_f0_hz=float(np.median(voiced)) if num_voiced else None,
        mean_f0_hz=float(np.mean(voiced)) if num_voiced else None,
        voiced_fraction=voiced_fraction,
        num_frames=num_frames,
        num_voiced_frames=num_voiced,
    )


@app.get("/health")
@app.post("/health")
def health():
    return {
        "status": "healthy",
        "service": "f0_eval",
        "version": "1.1.0",
        "backend": "praat-parselmouth",
    }


@app.post("/analyze", response_model=F0AnalyzeResponse)
def analyze(req: F0AnalyzeRequest):
    try:
        audio, sample_rate = _decode_audio(req.audio)
    except Exception as exc:
        raise HTTPException(400, f"Failed to decode audio: {exc}") from exc

    if req.f0_min_hz >= req.f0_max_hz:
        raise HTTPException(400, "f0_min_hz must be smaller than f0_max_hz")

    duration_sec = len(audio) / sample_rate if sample_rate > 0 else 0.0
    spans = req.spans or [F0Span(start=0.0, end=duration_sec, label="full")]

    try:
        results = [
            analyze_span(
                audio,
                sample_rate,
                span,
                f0_min_hz=req.f0_min_hz,
                f0_max_hz=req.f0_max_hz,
                time_step_sec=req.time_step_sec,
            )
            for span in spans
        ]
    except Exception as exc:
        logger.error("F0 analysis failed: %s", exc, exc_info=True)
        raise HTTPException(500, str(exc)) from exc

    return F0AnalyzeResponse(
        status="success",
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        spans=results,
    )


runner = ServiceRunner(
    app=app,
    service_name="F0 Evaluation",
    default_port=8034,
)


if __name__ == "__main__":
    runner.run()
