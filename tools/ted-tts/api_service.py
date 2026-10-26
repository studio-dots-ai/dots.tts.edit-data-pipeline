"""TED-TTS FastAPI service.

Wraps the TED-TTS/IndexTTS2 inference entry point with the same runtime
environment used by the IndexTTS2 service.
"""

import base64
import io
import logging
import os
import random
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from tag_adapter import adapt_tagged_text

DEFAULT_MODEL_DIR = (
    Path(os.environ["CHECKPOINTS_DIR"]) / "index-tts"
    if os.environ.get("CHECKPOINTS_DIR") else None
)

if DEFAULT_MODEL_DIR:
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_MODEL_DIR / "hf_cache"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TTS_MODEL: Optional[Any] = None
MODEL_CONFIG: Dict[str, Any] = {}
TEMP_DIR: Optional[str] = None
INFERENCE_LOCK = threading.Lock()
SAMPLING_RATE = 22050
MEL_TOKEN_SECONDS = 0.02
TAGGED_DURATION_MAX_MEL_TOKENS = 850
DEFAULT_EOS_CONTROL_POLICY = "attention_gated_delayed"
DEFAULT_EOS_GRACE_SECONDS = 1.0
DEFAULT_INTRA_ONSET_GRACE_SECONDS = 0.5
DEFAULT_INTRA_NEGATIVE_CORRECTION_DELAY_SECONDS = 1.0
DEFAULT_INTRA_DEADBAND_RATIO = 0.0
DEFAULT_INTRA_DURATION_EMBEDDING_GRACE_SECONDS = 0.0
# Duration-control knobs are intentionally scoped to TED-TTS modes 4/5 and to
# tagged synthesis requests where every <emo> tag carries dur. Plain mode 3
# (including the emotion pipeline's default tagged path) does not pass
# target_duration_tokens, so EOS policy, EOS grace, and intra controller
# options below are inert there.
DURATION_CONTROL_DISABLED_DETAIL = (
    "TED-TTS duration-control generation is temporarily disabled due to a "
    "known conditioning slot-order bug. Expected [duration_ctrl, duration_free] "
    "to match the IndexTTS2 duration-control fix; the old TED path used "
    "[duration_free, duration_ctrl]. Use no-duration mode 0/1/2/3 until the "
    "duration path is revalidated."
)
EMOTION_VECTOR_ORDER = [
    "happy",
    "angry",
    "sad",
    "fear",
    "hate",
    "low",
    "surprise",
    "neutral",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _temp_dir() -> str:
    global TEMP_DIR
    if TEMP_DIR is None:
        TEMP_DIR = tempfile.mkdtemp(prefix="ted_tts_service_")
    return TEMP_DIR




def _duration_limit_seconds(max_mel_tokens: int) -> float:
    return max_mel_tokens * MEL_TOKEN_SECONDS


def _duration_control_disabled() -> HTTPException:
    return HTTPException(status_code=501, detail=DURATION_CONTROL_DISABLED_DETAIL)


class EncodedAudio(BaseModel):
    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format suffix, e.g. wav/flac/mp3")


AudioPayload = Union[EncodedAudio, str]
DurationMode = Literal["none", "global", "intra", "both"]
EOSControlPolicy = Literal[
    "ted_original",
    "delayed",
    "attention_gated",
    "attention_gated_delayed",
]






class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize. Use '|' to split TED-TTS segments.")
    spk_audio_prompt: AudioPayload = Field(
        ..., description="Speaker prompt audio as a file path or base64 EncodedAudio object."
    )
    mode: Literal[0, 1, 2, 3, 4, 5] = Field(
        3,
        description="TED-TTS mode: 0 baseline, 1 emotion audio, 2 vector, 3 segment emotion text, 4 global duration, 5 global+local duration.",
    )
    emo_audio_prompt: Optional[AudioPayload] = Field(
        None, description="Mode 1 emotion reference audio."
    )
    emo_alpha: float = Field(1.0, ge=0.0, le=1.0)
    emo_vector: Optional[List[float]] = Field(
        None,
        description="Mode 2 8-dim vector in order: happy, angry, sad, fear, hate, low, surprise, neutral.",
    )
    emo_text: Optional[str] = Field(
        None,
        description="Modes 3/4/5 per-segment emotion descriptions separated by '|'.",
    )
    target_seconds: Optional[List[float]] = Field(
        None, description="Modes 4/5 target duration seconds per text segment."
    )
    duration_mode: Optional[DurationMode] = Field(
        None,
        description="Override duration mode for modes 4/5. Defaults to global for mode 4 and both for mode 5.",
    )
    eos_control_policy: Optional[EOSControlPolicy] = Field(
        None,
        description=(
            "EOS logits policy. Only affects modes 4/5 when duration_mode is "
            "global or both. ted_original preserves the upstream behavior."
        ),
    )
    eos_grace_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Delay before positive EOS duration bias. Only affects modes 4/5 "
            "when duration_mode is global or both."
        ),
    )
    intra_onset_guard_enabled: bool = Field(
        False,
        description=(
            "When duration_mode is intra or both, suppress negative local "
            "duration corrections at the start of each segment."
        ),
    )
    intra_onset_grace_seconds: float = Field(
        DEFAULT_INTRA_ONSET_GRACE_SECONDS,
        ge=0.0,
        description="Segment-onset grace window for intra duration control.",
    )
    intra_negative_correction_delay_seconds: float = Field(
        DEFAULT_INTRA_NEGATIVE_CORRECTION_DELAY_SECONDS,
        ge=0.0,
        description="Delay before intra duration control may apply negative correction.",
    )
    intra_deadband_ratio: float = Field(
        DEFAULT_INTRA_DEADBAND_RATIO,
        ge=0.0,
        le=1.0,
        description="No-op normalized-error band for intra duration control.",
    )
    intra_delta_ema_alpha: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Optional EMA alpha for smoothing intra duration deltas.",
    )
    intra_duration_embedding_grace_seconds: float = Field(
        DEFAULT_INTRA_DURATION_EMBEDDING_GRACE_SECONDS,
        ge=0.0,
        description=(
            "Segment-onset window that keeps mode-3-style free duration "
            "conditioning before local duration steering begins."
        ),
    )
    method: Literal["hmm", "max_head"] = Field(
        "hmm", description="Segment-switch detection method."
    )
    seed: Optional[int] = Field(None, description="Optional deterministic seed.")
    is_fp16: bool = Field(False, description="Use FP16 inference.")
    use_random: bool = Field(False, description="Enable TED-TTS random emotion matrix sampling.")
    top_p: float = Field(0.8, ge=0.0, le=1.0)
    top_k: int = Field(30, ge=1)
    temperature: float = Field(0.8, gt=0.0)
    repetition_penalty: float = Field(10.0, ge=1.0)
    length_penalty: float = Field(0.0)
    num_beams: int = Field(3, ge=1)
    max_mel_tokens: int = Field(850, ge=100)
    max_text_tokens_per_sentence: int = Field(150, ge=10)
    output_format: Literal["wav", "json"] = Field(
        "wav", description="Return raw WAV bytes or JSON containing base64 WAV."
    )

    @model_validator(mode="after")
    def validate_mode_inputs(self):
        if self.mode == 1 and self.emo_audio_prompt is None:
            raise ValueError("mode 1 requires emo_audio_prompt")
        if self.mode == 2:
            if self.emo_vector is None:
                raise ValueError("mode 2 requires emo_vector")
            if len(self.emo_vector) != 8:
                raise ValueError("emo_vector must contain exactly 8 values")
        if self.mode in (3, 4, 5):
            if not self.emo_text:
                raise ValueError("modes 3/4/5 require emo_text")
            text_segments = self.text.split("|")
            emo_segments = self.emo_text.split("|")
            if len(text_segments) != len(emo_segments):
                raise ValueError(
                    "text and emo_text must have the same number of '|' separated segments"
                )
        if self.mode in (4, 5):
            if not self.target_seconds:
                raise ValueError("modes 4/5 require target_seconds")
            if len(self.target_seconds) != len(self.text.split("|")):
                raise ValueError(
                    "target_seconds must have one value per '|' separated text segment"
                )
        return self


class TaggedAdaptRequest(BaseModel):
    tagged_text: str = Field(..., description="IndexTTS-style XML tagged text.")


class TaggedAdaptResponse(BaseModel):
    text: str
    emo_text: str
    segments: List[Dict[str, Any]]
    target_seconds: Optional[List[float]] = None


class TaggedSynthesisRequest(BaseModel):
    tagged_text: str = Field(..., description="IndexTTS-style XML tagged text.")
    spk_audio_prompt: AudioPayload = Field(
        ..., description="Speaker prompt audio as a file path or base64 EncodedAudio object."
    )
    method: Literal["hmm", "max_head"] = Field(
        "hmm", description="Segment-switch detection method."
    )
    seed: Optional[int] = Field(None, description="Optional deterministic seed.")
    use_random: bool = Field(False, description="Enable TED-TTS random emotion matrix sampling.")
    top_p: float = Field(0.8, ge=0.0, le=1.0)
    top_k: int = Field(30, ge=1)
    temperature: float = Field(0.8, gt=0.0)
    repetition_penalty: float = Field(10.0, ge=1.0)
    length_penalty: float = Field(0.0)
    num_beams: int = Field(3, ge=1)
    max_mel_tokens: int = Field(850, ge=100)
    max_text_tokens_per_sentence: int = Field(150, ge=10)
    duration_mode: Optional[DurationMode] = Field(
        None,
        description="Override tagged duration control mode when all tags include dur. Defaults to both.",
    )
    eos_control_policy: Optional[EOSControlPolicy] = Field(
        None,
        description=(
            "EOS logits policy for tagged duration control. Only affects the "
            "mode 5-equivalent path when duration_mode is global or both."
        ),
    )
    eos_grace_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Delay before positive EOS duration bias. Only affects tagged "
            "duration control when duration_mode is global or both."
        ),
    )
    intra_onset_guard_enabled: bool = Field(
        False,
        description=(
            "When duration_mode is intra or both, suppress negative local "
            "duration corrections at the start of each segment."
        ),
    )
    intra_onset_grace_seconds: float = Field(
        DEFAULT_INTRA_ONSET_GRACE_SECONDS,
        ge=0.0,
        description="Segment-onset grace window for intra duration control.",
    )
    intra_negative_correction_delay_seconds: float = Field(
        DEFAULT_INTRA_NEGATIVE_CORRECTION_DELAY_SECONDS,
        ge=0.0,
        description="Delay before intra duration control may apply negative correction.",
    )
    intra_deadband_ratio: float = Field(
        DEFAULT_INTRA_DEADBAND_RATIO,
        ge=0.0,
        le=1.0,
        description="No-op normalized-error band for intra duration control.",
    )
    intra_delta_ema_alpha: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Optional EMA alpha for smoothing intra duration deltas.",
    )
    intra_duration_embedding_grace_seconds: float = Field(
        DEFAULT_INTRA_DURATION_EMBEDDING_GRACE_SECONDS,
        ge=0.0,
        description=(
            "Segment-onset window that keeps mode-3-style free duration "
            "conditioning before local duration steering begins."
        ),
    )
    output_format: Literal["wav", "json"] = Field(
        "wav", description="Return raw WAV bytes or JSON containing base64 WAV."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: Optional[str]
    config: Dict[str, Any]


class InfoResponse(BaseModel):
    service: str = "TED-TTS"
    sampling_rate: int = SAMPLING_RATE
    supported_modes: List[int] = [0, 1, 2, 3, 4, 5]
    emotion_vector_order: List[str] = EMOTION_VECTOR_ORDER
    duration_token_seconds: float = MEL_TOKEN_SECONDS
    tagged_duration_max_mel_tokens: int = TAGGED_DURATION_MAX_MEL_TOKENS
    tagged_duration_max_seconds: float = _duration_limit_seconds(
        TAGGED_DURATION_MAX_MEL_TOKENS
    )
    default_eos_control_policy: str = DEFAULT_EOS_CONTROL_POLICY
    default_eos_grace_seconds: float = DEFAULT_EOS_GRACE_SECONDS
    eos_control_policies: List[str] = [
        "ted_original",
        "delayed",
        "attention_gated",
        "attention_gated_delayed",
    ]


app = FastAPI(
    title="TED-TTS Service",
    description="Training-free intra-utterance emotion and duration control built on IndexTTS2.",
    version="1.0.0",
)


def decode_audio_input(audio: AudioPayload, prefix: str) -> str:
    if isinstance(audio, str):
        if not Path(audio).exists():
            raise HTTPException(status_code=400, detail=f"Audio path does not exist: {audio}")
        return audio

    try:
        audio_bytes = base64.b64decode(audio.data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc

    suffix = audio.format.lstrip(".") or "wav"
    path = Path(_temp_dir()) / f"{prefix}_{uuid.uuid4().hex}.{suffix}"
    path.write_bytes(audio_bytes)
    return str(path)


def load_model(
    cfg_path: str,
    model_dir: str,
    device: Optional[str],
    is_fp16: bool,
    use_cuda_kernel: Optional[bool],
) -> None:
    global TTS_MODEL, MODEL_CONFIG

    from adapter import IndexTTS2

    logger.info("Loading TED-TTS model from %s", model_dir)
    TTS_MODEL = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        is_fp16=is_fp16,
        device=device,
        use_cuda_kernel=use_cuda_kernel,
    )
    MODEL_CONFIG = {
        "cfg_path": cfg_path,
        "model_dir": model_dir,
        "device": getattr(TTS_MODEL, "device", device or "auto"),
        "is_fp16": getattr(TTS_MODEL, "is_fp16", is_fp16),
        "use_cuda_kernel": getattr(TTS_MODEL, "use_cuda_kernel", use_cuda_kernel),
    }
    logger.info("TED-TTS model loaded")


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"service": "TED-TTS Service", "status": "running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if TTS_MODEL is not None else "not_ready",
        model_loaded=TTS_MODEL is not None,
        device=MODEL_CONFIG.get("device", "unknown"),
        model_version=str(getattr(TTS_MODEL, "model_version", "ted-tts")) if TTS_MODEL else None,
        config=MODEL_CONFIG,
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    return InfoResponse()


@app.post("/adapt_tagged", response_model=TaggedAdaptResponse)
def adapt_tagged(request: TaggedAdaptRequest):
    try:
        adapted = adapt_tagged_text(request.tagged_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TaggedAdaptResponse(**adapted.to_dict())


@app.post("/synthesize")
def synthesize(request: SynthesisRequest):
    if TTS_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.seed is not None:
        _set_seed(request.seed)

    temp_files: List[str] = []
    started = time.time()
    try:
        spk_audio_path = decode_audio_input(request.spk_audio_prompt, "spk")
        if isinstance(request.spk_audio_prompt, EncodedAudio):
            temp_files.append(spk_audio_path)

        emo_audio_path = None
        if request.emo_audio_prompt is not None:
            emo_audio_path = decode_audio_input(request.emo_audio_prompt, "emo")
            if isinstance(request.emo_audio_prompt, EncodedAudio):
                temp_files.append(emo_audio_path)

        text = request.text
        if request.mode in (0, 1, 2) and "|" in text:
            text = text.replace("|", " ")

        output_path = str(Path(_temp_dir()) / f"ted_tts_{uuid.uuid4().hex}.wav")
        temp_files.append(output_path)

        common_kwargs: Dict[str, Any] = {
            "spk_audio_prompt": spk_audio_path,
            "text": text,
            "output_path": output_path,
            "verbose": True,
            "use_random": request.use_random,
            "max_text_tokens_per_sentence": request.max_text_tokens_per_sentence,
            "do_sample": True,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "temperature": request.temperature,
            "length_penalty": request.length_penalty,
            "num_beams": request.num_beams,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
            "method": request.method,
        }

        with INFERENCE_LOCK:
            if request.mode == 0:
                TTS_MODEL.infer(**common_kwargs)
            elif request.mode == 1:
                TTS_MODEL.infer(
                    **common_kwargs,
                    emo_audio_prompt=emo_audio_path,
                    emo_alpha=request.emo_alpha,
                )
            elif request.mode == 2:
                TTS_MODEL.infer(
                    **common_kwargs,
                    emo_audio_prompt=None,
                    emo_alpha=0,
                    emo_vector=[request.emo_vector],
                )
            elif request.mode == 3:
                TTS_MODEL.infer(
                    **common_kwargs,
                    emo_audio_prompt=None,
                    emo_alpha=0,
                    use_emo_text=True,
                    emo_text=request.emo_text,
                )
            else:
                raise _duration_control_disabled()

        wav_bytes = Path(output_path).read_bytes()
        elapsed = time.time() - started
        if request.output_format == "json":
            return JSONResponse(
                {
                    "status": "success",
                    "output_format": "wav_base64",
                    "audio": base64.b64encode(wav_bytes).decode("ascii"),
                    "sample_rate": SAMPLING_RATE,
                    "response_time": elapsed,
                }
            )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(SAMPLING_RATE),
                "X-Response-Time": f"{elapsed:.3f}",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TED-TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for path in temp_files:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temp file %s", path, exc_info=True)


@app.post("/synthesize_tagged")
def synthesize_tagged(request: TaggedSynthesisRequest):
    if TTS_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.seed is not None:
        _set_seed(request.seed)

    try:
        adapted = adapt_tagged_text(request.tagged_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    temp_files: List[str] = []
    started = time.time()
    try:
        spk_audio_path = decode_audio_input(request.spk_audio_prompt, "spk")
        if isinstance(request.spk_audio_prompt, EncodedAudio):
            temp_files.append(spk_audio_path)

        output_path = str(Path(_temp_dir()) / f"ted_tts_{uuid.uuid4().hex}.wav")
        temp_files.append(output_path)

        common_kwargs: Dict[str, Any] = {
            "spk_audio_prompt": spk_audio_path,
            "text": adapted.text,
            "output_path": output_path,
            "verbose": True,
            "use_random": request.use_random,
            "max_text_tokens_per_sentence": request.max_text_tokens_per_sentence,
            "do_sample": True,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "temperature": request.temperature,
            "length_penalty": request.length_penalty,
            "num_beams": request.num_beams,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
            "method": request.method,
        }

        duration_control: Dict[str, Any] = {
            "enabled": adapted.target_seconds is not None,
            "duration_mode": None,
            "eos_control_policy": None,
            "eos_grace_seconds": None,
            "eos_grace_tokens": None,
            "intra_control": {
                "enabled": False,
                "onset_guard_enabled": None,
                "onset_grace_seconds": None,
                "onset_grace_tokens": None,
                "negative_correction_delay_seconds": None,
                "negative_correction_delay_tokens": None,
                "deadband_ratio": None,
                "delta_ema_alpha": None,
                "duration_embedding_grace_seconds": None,
                "duration_embedding_grace_tokens": None,
            },
            "target_seconds": adapted.target_seconds,
            "target_duration_tokens": None,
            "max_mel_tokens": request.max_mel_tokens,
            "effective_max_mel_tokens": request.max_mel_tokens,
            "max_seconds": _duration_limit_seconds(request.max_mel_tokens),
        }
        duration_kwargs: Dict[str, Any] = {}
        if adapted.target_seconds is not None:
            raise _duration_control_disabled()

        with INFERENCE_LOCK:
            TTS_MODEL.infer(
                **common_kwargs,
                emo_audio_prompt=None,
                emo_alpha=0,
                use_emo_text=True,
                emo_text=adapted.emo_text,
                **duration_kwargs,
            )

        wav_bytes = Path(output_path).read_bytes()
        elapsed = time.time() - started
        if request.output_format == "json":
            return JSONResponse(
                {
                    "status": "success",
                    "output_format": "wav_base64",
                    "audio": base64.b64encode(wav_bytes).decode("ascii"),
                    "sample_rate": SAMPLING_RATE,
                    "response_time": elapsed,
                    "adapted": adapted.to_dict(),
                    "duration_control": duration_control,
                }
            )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(SAMPLING_RATE),
                "X-Response-Time": f"{elapsed:.3f}",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TED-TTS tagged synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for path in temp_files:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temp file %s", path, exc_info=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TED-TTS Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8035)
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--cfg-path", default=str(DEFAULT_MODEL_DIR / "config.yaml") if DEFAULT_MODEL_DIR else None, required=DEFAULT_MODEL_DIR is None)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR else None, required=DEFAULT_MODEL_DIR is None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--cuda-kernel", action="store_true", default=None)
    args = parser.parse_args()

    _temp_dir()
    load_model(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        device=args.device,
        is_fp16=args.fp16,
        use_cuda_kernel=args.cuda_kernel,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
