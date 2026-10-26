"""FastAPI wrapper for audio assembly driven by `edit_tagged_text`."""

from __future__ import annotations

import argparse
import base64
import io
import logging
import time

from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from assembler import TARGET_SR, decode_audio, encode_wav, run_assembly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EncodedAudio(BaseModel):
    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format")


class SmoothingConfig(BaseModel):
    regen_margin_sec: Optional[float] = Field(None)
    f5_steps: int = Field(32, ge=1)
    f5_cfg_strength: float = Field(2.0)
    f5_sway_sampling_coef: float = Field(-1.0)
    f5_seed: Optional[int] = Field(None)
    smooth_continuation_boundaries: bool = Field(True)

    model_config = {
        "json_schema_extra": {
            "description": (
                "Audio Assembly forwards only regen_margin_sec and F5 settings to "
                "Boundary Pause Smoother because splice smoothing uses mode='smooth_only'."
            )
        }
    }


class AssembleRequest(BaseModel):
    source_audio: EncodedAudio
    target_audio: EncodedAudio
    edit_tagged_text: str = Field(
        ...,
        description="Full transcript with <target>...</target> spans marking content to take from target audio.",
    )
    time_spans_from_source: Optional[List[Any]] = Field(
        None,
        description=(
            "Optional manual source-audio spans. Expected JSON shape is [[start_sec, end_sec], ...]. "
            "Malformed or ineligible manual spans are rejected."
        ),
    )
    time_spans_from_target: Optional[List[Any]] = Field(
        None,
        description=(
            "Optional manual target-audio spans. Expected JSON shape is [[start_sec, end_sec], ...]. "
            "Malformed or ineligible manual spans are rejected."
        ),
    )
    preserve_outer_boundaries: bool = Field(
        True,
        description=(
            "When true, preserve the full leading segment and trailing segment of the final assembly, "
            "regardless of whether they come from source or target audio."
        ),
    )
    language: Optional[str] = Field(None)
    aligner_backend: str = Field(
        "auto",
        description="Explicit aligner gateway backend: auto or qwen3.",
    )
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    output_format: str = Field("wav")
    debug: bool = Field(False)
    debug_dir: Optional[str] = Field(None)
    trace_id: Optional[str] = Field(None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_audio": {"data": "<base64>", "format": "wav"},
                "target_audio": {"data": "<base64>", "format": "wav"},
                "edit_tagged_text": "<target>Monday,</target> there's gonna be haze, but Tuesday, look for <target>thunderstorms.</target>",
                "time_spans_from_source": [[0.5, 2.0]],
                "time_spans_from_target": [[0.0, 0.6], [2.3, 3.1]],
                "preserve_outer_boundaries": True,
                "language": "eng",
                "aligner_backend": "qwen3",
                "smoothing": {
                    "f5_seed": 42,
                    "smooth_continuation_boundaries": True,
                },
                "output_format": "wav",
            }
        }
    }


class AssembleResponseJson(BaseModel):
    status: str = "success"
    audio: EncodedAudio
    sample_rate: int
    duration_sec: float
    n_splice_points: int
    n_target_spans: int
    target_spans: List[Dict]
    details: Dict
    response_time: Optional[float] = None
    trace_id: Optional[str] = None


SERVICE_CONFIG = {
    "aligner_url": "http://localhost:8029",
    "smoother_url": "http://localhost:8018",
}


app = FastAPI(
    title="Audio Assembly Service",
    description="Splice source and target audio according to <target>...</target> markup",
    version="2.0.0",
)


@app.get("/")
async def root():
    return {
        "service": "Audio Assembly Service",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/assemble")
async def assemble(request: AssembleRequest):
    try:
        t0 = time.time()
        source_bytes = base64.b64decode(request.source_audio.data)
        target_bytes = base64.b64decode(request.target_audio.data)

        result = run_assembly(
            source_bytes=source_bytes,
            target_bytes=target_bytes,
            edit_tagged_text=request.edit_tagged_text,
            aligner_url=SERVICE_CONFIG["aligner_url"],
            smoother_url=SERVICE_CONFIG["smoother_url"],
            language=request.language,
            aligner_backend=request.aligner_backend,
            regen_margin_sec=request.smoothing.regen_margin_sec,
            smooth_continuation_boundaries=request.smoothing.smooth_continuation_boundaries,
            f5_steps=request.smoothing.f5_steps,
            f5_cfg_strength=request.smoothing.f5_cfg_strength,
            f5_sway_sampling_coef=request.smoothing.f5_sway_sampling_coef,
            f5_seed=request.smoothing.f5_seed,
            debug=request.debug,
            debug_dir=request.debug_dir,
            trace_id=request.trace_id,
            time_spans_from_source=request.time_spans_from_source,
            time_spans_from_target=request.time_spans_from_target,
            preserve_outer_boundaries=request.preserve_outer_boundaries,
        )

        response_time = time.time() - t0
        if request.output_format == "json":
            wav_bytes = encode_wav(result.audio, result.sample_rate)
            return AssembleResponseJson(
                audio={
                    "data": base64.b64encode(wav_bytes).decode("utf-8"),
                    "format": "wav",
                },
                sample_rate=result.sample_rate,
                duration_sec=len(result.audio) / result.sample_rate,
                n_splice_points=len(result.splice_points),
                n_target_spans=len(result.target_spans),
                target_spans=result.target_spans,
                details=result.details,
                response_time=response_time,
                trace_id=request.trace_id,
            )

        wav_bytes = encode_wav(result.audio, result.sample_rate)
        headers = {
            "X-Sample-Rate": str(result.sample_rate),
            "X-Duration-Sec": f"{len(result.audio) / result.sample_rate:.3f}",
            "X-Splice-Points": str(len(result.splice_points)),
            "X-Target-Spans": str(len(result.target_spans)),
        }
        if request.trace_id:
            headers["X-Trace-Id"] = request.trace_id
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Assembly failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    global SERVICE_CONFIG

    parser = argparse.ArgumentParser(description="Audio Assembly Service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--aligner-url", type=str, default="http://localhost:8029")
    parser.add_argument("--smoother-url", type=str, default="http://localhost:8018")
    args = parser.parse_args()

    SERVICE_CONFIG = {
        "aligner_url": args.aligner_url,
        "smoother_url": args.smoother_url,
    }

    logger.info(
        "Audio Assembly service starting on %s:%s (aligner=%s smoother=%s)",
        args.host,
        args.port,
        SERVICE_CONFIG["aligner_url"],
        SERVICE_CONFIG["smoother_url"],
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
