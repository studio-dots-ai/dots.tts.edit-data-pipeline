"""
IndexTTS Synthesis Service: Direct inference endpoint for TTS with emotion control.

A lightweight FastAPI service wrapping IndexTTS2 for voice cloning with emotion control.
Supports three emotion prompt modes: audio, text, and vector.
Supports tensor/audio transmission via Base64-encoded bytes or file paths.

Usage:
    python tools/index-tts/api_service.py --port 8002

    # Server runs on http://localhost:8002
    # API docs at http://localhost:8002/docs

Example client usage:
    import requests
    import numpy as np
    import base64

    # Encode audio file to base64
    def encode_audio_file(path: str) -> dict:
        with open(path, "rb") as f:
            audio_bytes = f.read()
        return {
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": "wav",  # or "mp3", "flac", etc.
        }

    # Make request with emotion vector
    response = requests.post("http://localhost:8002/synthesize", json={
        "text": "你好世界。",
        "spk_audio_prompt": encode_audio_file("speaker.wav"),
        "emo_vector": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],  # happy + calm
        "output_format": "wav",
    })

    # Save response
    with open("output.wav", "wb") as f:
        f.write(response.content)
"""

import base64
import io
import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

_CHECKPOINTS_DIR = (
    os.path.join(os.environ["CHECKPOINTS_DIR"], "index-tts")
    if os.environ.get("CHECKPOINTS_DIR") else None
)

if _CHECKPOINTS_DIR:
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(_CHECKPOINTS_DIR, "hf_cache"))

from adapter import IndexTTS2, UPSTREAM_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("IndexTTS upstream source: %s", UPSTREAM_ROOT)


# -----------------------------------------
# Global State
# -----------------------------------------

TTS_MODEL: Optional[Any] = None
MODEL_CONFIG: Dict[str, Any] = {}
TEMP_DIR: Optional[str] = None
SAMPLING_RATE: int = 22050  # IndexTTS2 output sample rate
# Model requests share one CUDA context, and CUDA graph capture is not safe
# alongside any other model operation. Serialize every model endpoint from
# request preparation through response construction.
INFERENCE_LOCK = threading.RLock()


def serialized_model_request(func):
    """Run a model endpoint under the process-wide inference lock."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        with INFERENCE_LOCK:
            return func(*args, **kwargs)

    wrapped.uses_global_inference_lock = True
    return wrapped

# Emotion presets mapping emotion name -> 8-dim vector
# Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMO_PRESETS = {
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "angry": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "afraid": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "disgusted": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "melancholic": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "calm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}
NEUTRAL_TAG = "none"
EMOTION_VECTOR_ORDER = list(EMO_PRESETS.keys())
EMO_BIAS = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class EmotionTaggedSegment:
    """Parsed inline emotion-tagged segment."""

    emotion: Optional[str]
    text: str
    alpha: Optional[float] = None


def _has_untagged_emotion_text(tagged_text: str) -> bool:
    """Return True if non-whitespace text exists outside inline emotion tags."""

    pattern = re.compile(
        r"<(\w+)"  # tag name
        r"(?:\s*,\s*"
        r"alpha\s*=\s*([\d.]+)"  # alpha=X
        r")?\s*>"
        r"(.*?)"  # text content
        r"</\1>",  # closing tag
        re.DOTALL,
    )
    last_end = 0
    for match in pattern.finditer(tagged_text):
        if match.start() > last_end and tagged_text[last_end : match.start()].strip():
            return True
        last_end = match.end()
    return last_end < len(tagged_text) and bool(tagged_text[last_end:].strip())


def parse_emotion_tags(
    tagged_text: str,
) -> List[EmotionTaggedSegment]:
    """
    Parse emotion-tagged text into structured segments.

    Supported formats::

        <happy>Hello world!</happy>               -> EmotionTaggedSegment("happy", "Hello world!")
        <happy, alpha=0.75>Hello!</happy>         -> EmotionTaggedSegment("happy", "Hello!", alpha=0.75)

    Level-to-alpha conversion should be done by the caller before invoking
    this function (see pipeline ``tag_utils.convert_levels_to_alpha``).
    Explicit `<none>` spans are represented as neutral segments with
    ``emotion=None``.
    """
    pattern = re.compile(
        r"<(\w+)"  # tag name
        r"(?:\s*,\s*"
        r"alpha\s*=\s*([\d.]+)"  # alpha=X
        r")?\s*>"
        r"(.*?)"  # text content
        r"</\1>",  # closing tag
        re.DOTALL,
    )
    segments: List[EmotionTaggedSegment] = []
    for m in pattern.finditer(tagged_text):
        emotion = m.group(1).lower()
        explicit_neutral = emotion == NEUTRAL_TAG
        if explicit_neutral:
            emotion = None
        control_value = m.group(2)
        text = m.group(3).strip()

        alpha: Optional[float] = None
        if control_value is not None:
            alpha = float(control_value)

        if text:
            segments.append(
                EmotionTaggedSegment(
                    emotion=emotion,
                    text=text,
                    alpha=alpha,
                )
            )
    return segments


# -----------------------------------------
# Pydantic Models for Request/Response
# -----------------------------------------


class EncodedAudio(BaseModel):
    """Base64-encoded audio bytes with format metadata."""

    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format: 'wav', 'mp3', 'flac', etc.")


class AudioInput(BaseModel):
    """Audio input that can be either encoded bytes or a file path."""

    encoded: Optional[EncodedAudio] = Field(None, description="Base64-encoded audio")
    path: Optional[str] = Field(None, description="File path to audio")

    @model_validator(mode="after")
    def check_one_provided(self):
        if self.encoded is None and self.path is None:
            raise ValueError("Either 'encoded' or 'path' must be provided")
        if self.encoded is not None and self.path is not None:
            raise ValueError("Only one of 'encoded' or 'path' should be provided")
        return self


class SynthesisRequest(BaseModel):
    """Request for TTS synthesis with emotion control."""

    # Required inputs
    text: str = Field(..., description="Text to synthesize")
    spk_audio_prompt: Union[EncodedAudio, str] = Field(
        ...,
        description="Speaker reference audio. Either Base64-encoded audio object or file path string.",
    )

    # Emotion control (mutually exclusive - at most one should be provided)
    emo_audio_prompt: Optional[Union[EncodedAudio, str]] = Field(
        None,
        description="Emotion reference audio. Either Base64-encoded audio object or file path string.",
    )
    emo_text: Optional[str] = Field(
        None,
        description="Text for emotion detection via QwenEmotion LLM. "
        "The model extracts emotion vectors from this text.",
    )
    emo_vector: Optional[List[float]] = Field(
        None,
        description="8-dimensional emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]. "
        "Values typically range 0.0-1.0, total sum should be <= 0.8 for best results.",
    )

    # Emotion-tagged text mode
    use_emotion_tags: bool = Field(
        False,
        description="If True, parse emotion tags in text like '<happy>Hello!</happy><calm>World.</calm>'. "
        "Each tagged segment is synthesized with its emotion vector. "
        "When enabled, emo_audio_prompt/emo_text/emo_vector are ignored (per-segment tags take over).",
    )

    # Continuation-based emotion-tagged synthesis
    generation_paradigm: Literal[
        "independent", "continuation", "independent_smoothed"
    ] = Field(
        "independent",
        description="Generation paradigm for emotion-tagged text (only used when use_emotion_tags=True):\n"
        "- 'independent': Call infer() separately for each segment, concatenate with silence.\n"
        "- 'continuation': Each step generates the full remaining text, then truncates at the current "
        "segment boundary to build the next continuation prefix. Requires aligner gateway service.\n"
        "- 'independent_smoothed' has been deprecated and is no longer supported by this service; "
        "run boundary smoothing in the pipeline instead.",
        json_schema_extra={"enum": ["independent", "continuation"]},
    )
    aligner_url: str = Field(
        "http://localhost:8029",
        description="URL of the aligner gateway service. "
        "Only used when generation_paradigm='continuation'.",
    )
    aligner_backend: Literal["auto", "qwen3"] = Field(
        "auto",
        description="Explicit aligner gateway backend for continuation boundary queries.",
    )
    aligner_language: Optional[str] = Field(
        None,
        description="Optional aligner gateway language code, e.g. eng or zho.",
    )
    smoother_url: str = Field(
        "http://localhost:8018",
        description="Boundary Pause Smoother service URL. Only used when "
        "use_emotion_tags=True, generation_paradigm='continuation', and "
        "smooth_continuation_boundaries=True.",
    )
    smoother_debug: bool = Field(
        False,
        description="If True, save boundary smoothing debug artifacts. Only used "
        "for optional continuation boundary smoothing.",
    )
    smoother_debug_dir: Optional[str] = Field(
        None,
        description="Directory for boundary smoothing debug artifacts. Only used "
        "for optional continuation boundary smoothing.",
    )
    continuation_debug: bool = Field(
        False,
        description="If True, save per-step continuation debug artifacts including "
        "each untruncated full-step waveform and the exact texts/params used.",
    )
    continuation_debug_dir: Optional[str] = Field(
        None,
        description="Directory for continuation debug artifacts. Only used when "
        "continuation_debug=True.",
    )
    smooth_continuation_boundaries: bool = Field(
        True,
        description="If True, smooth only the internal segment splice points for "
        "emotion-tagged continuation synthesis using the smoother regen-only fast path.",
    )
    smooth_radius_sec: Optional[float] = Field(
        None,
        description="Half-width of each continuation splice regeneration region in seconds. "
        "When None, the service omits regen_margin_sec and Boundary Pause Smoother uses its own default.",
    )

    # Common parameters
    emo_alpha: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Emotion intensity control (0.0 = ignore emotion, 1.0 = full emotion). Default: 0.8",
    )
    interval_silence: int = Field(
        200,
        ge=0,
        description="Silence duration between segments in milliseconds",
    )

    # Generation parameters (commonly tuned)
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible inference. Sets torch, numpy, and python random seeds.",
    )
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(30, ge=1, description="Top-k sampling parameter")
    temperature: float = Field(0.8, gt=0.0, description="Sampling temperature")
    repetition_penalty: float = Field(10.0, ge=1.0, description="Repetition penalty")
    max_mel_tokens: int = Field(
        1500, ge=100, description="Maximum mel tokens per segment"
    )
    max_text_tokens_per_segment: int = Field(
        120, ge=10, description="Maximum text tokens per segment"
    )

    # Output format
    output_format: str = Field(
        "wav",
        description="Output format: 'wav' (audio bytes) or 'numpy' (encoded waveform array)",
    )

    # Cache control
    return_cache: bool = Field(
        False,
        description="If True, response includes precomputed conditioning tensors "
        "that can be reused in subsequent requests to skip recomputation. "
        "For WAV output, the response becomes multipart/mixed (part 1 = WAV, "
        "part 2 = cache JSON). For numpy output, cache is included in the "
        "JSON response. Not supported for emotion-tagged multi-segment synthesis.",
    )
    cache: Optional["InferenceCache"] = Field(
        None,
        description="Precomputed conditioning cache from a previous response. "
        "When provided, speaker/emotion embeddings are loaded from cache "
        "instead of being recomputed from audio. The spk_audio_prompt field "
        "is still required but will be ignored if cache contains speaker data. "
        "If the cache contains a 'continuation' field, prefix_tokens and "
        "prefix_text are automatically extracted for continuation mode.",
    )

    # Token I/O
    return_tokens: bool = Field(
        False,
        description="If True, return generated semantic tokens alongside audio. "
        "When return_cache=True, the tokens are also stored in the cache's "
        "'continuation' field for efficient chaining without re-extracting "
        "tokens from audio. For WAV output with return_cache, tokens are "
        "embedded in the cache JSON. For numpy output, a 'tokens' field is "
        "added to the JSON response.",
    )
    skip_vocode: bool = Field(
        False,
        description="If True, skip S2Mel + BigVGAN synthesis and return only "
        "semantic tokens. Implies return_tokens=True. Useful for getting tokens "
        "without the cost of audio synthesis.",
    )

    # Tracing
    trace_id: Optional[str] = Field(
        None,
        description="Caller-supplied trace ID for request tracing and profiling.",
    )

    @model_validator(mode="after")
    def check_emotion_mutual_exclusivity(self):
        """Ensure at most one emotion control method is provided."""
        emotion_methods = [
            self.emo_audio_prompt is not None,
            self.emo_text is not None,
            self.emo_vector is not None,
        ]
        if sum(emotion_methods) > 1:
            raise ValueError(
                "Emotion control methods are mutually exclusive. "
                "Provide at most one of: emo_audio_prompt, emo_text, emo_vector"
            )
        return self

    @model_validator(mode="after")
    def check_emo_vector_length(self):
        """Ensure emotion vector has exactly 8 dimensions."""
        if self.emo_vector is not None and len(self.emo_vector) != 8:
            raise ValueError(
                f"emo_vector must have exactly 8 dimensions, got {len(self.emo_vector)}. "
                "Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]"
            )
        return self

    @model_validator(mode="after")
    def check_output_format(self):
        """Ensure output format is valid."""
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(
                f"output_format must be 'wav' or 'numpy', got '{self.output_format}'"
            )
        return self

    @model_validator(mode="after")
    def check_generation_paradigm(self):
        """Reject removed paradigms with a clear migration message."""
        if self.generation_paradigm == "independent_smoothed":
            raise ValueError(
                "generation_paradigm='independent_smoothed' has been deprecated and "
                "is no longer supported by the IndexTTS service. Run boundary pause "
                "smoothing in the pipeline instead."
            )
        if self.generation_paradigm not in ("independent", "continuation"):
            raise ValueError(
                "generation_paradigm must be 'independent' or 'continuation', got "
                f"'{self.generation_paradigm}'"
            )
        return self

    @model_validator(mode="after")
    def check_continuation_requirements(self):
        """Validate continuation paradigm fields (prompt_text no longer required)."""
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "text": "欢迎使用IndexTTS语音合成服务。",
                "spk_audio_prompt": "/path/to/speaker.wav",
                "emo_vector": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                "emo_alpha": 0.8,
                "output_format": "wav",
            }
        }


class EncodedTensor(BaseModel):
    """Base64-encoded numpy array with metadata."""

    data: str = Field(..., description="Base64-encoded tensor bytes")
    shape: List[int] = Field(..., description="Tensor shape")
    dtype: str = Field("float32", description="Numpy dtype string")


# ------------------------------------------------------------------
# Externalized inference cache models
# ------------------------------------------------------------------


class SpeakerCache(BaseModel):
    """Pre-computed speaker conditioning tensors.

    Returned when ``return_cache=True``.  Pass back in a subsequent request
    (via ``cache``) to skip speaker conditioning recomputation.
    """

    spk_cond: EncodedTensor = Field(
        ..., description="w2v-bert speaker conditioning embedding"
    )
    s2mel_style: EncodedTensor = Field(..., description="CAM++ global style vector")
    s2mel_prompt: EncodedTensor = Field(
        ..., description="Length-regulated prompt condition"
    )
    mel: EncodedTensor = Field(..., description="Reference mel spectrogram")


class EmotionCache(BaseModel):
    """Pre-computed emotion conditioning tensor."""

    emo_cond: EncodedTensor = Field(
        ..., description="w2v-bert emotion conditioning embedding"
    )


class InferenceCache(BaseModel):
    """Externalized conditioning cache.

    Returned in the response when ``return_cache=True``.  Pass back in a
    subsequent request via the ``cache`` field to reuse precomputed speaker
    and/or emotion conditioning tensors, avoiding recomputation from audio.

    When ``continuation`` is present, the ``prefix_tokens`` and ``prefix_text``
    from the previous generation are carried forward, enabling efficient
    continuation chaining without re-extracting tokens from audio.
    """

    speaker: SpeakerCache = Field(..., description="Speaker conditioning cache")
    emotion: Optional[EmotionCache] = Field(
        None,
        description="Emotion conditioning cache (None if no separate emotion audio was used)",
    )
    continuation: Optional["ContinuationCache"] = Field(
        None,
        description="Continuation tokens/text from a previous generation (for token-level continuation chaining)",
    )


class ContinuationCache(BaseModel):
    """Continuation prefix from a previous generation.

    Carries the semantic tokens and corresponding text from the previous
    generation so that the next call can continue without re-extracting
    tokens from audio (which introduces lossy roundtrip errors).
    """

    tokens: EncodedTensor = Field(
        ..., description="1-D int64 numpy array of semantic codes (0-8191)"
    )
    text: str = Field(
        ...,
        description="Transcript corresponding to the continuation tokens",
    )


class SynthesisResponse(BaseModel):
    """Response for synthesis request (for numpy output format)."""

    status: str = Field("success", description="Request status")
    output_format: str = Field(..., description="Output format returned")
    tensor: Optional[EncodedTensor] = Field(
        None, description="Output waveform tensor (for numpy format)"
    )
    sample_rate: int = Field(SAMPLING_RATE, description="Audio sample rate")
    duration_seconds: Optional[float] = Field(
        None, description="Audio duration in seconds"
    )
    response_time: Optional[float] = Field(
        None, description="Server-side processing time in seconds"
    )
    trace_id: Optional[str] = Field(
        None, description="Echo of caller-supplied trace ID"
    )
    message: Optional[str] = Field(None, description="Additional message")
    cache: Optional[InferenceCache] = Field(
        None,
        description="Precomputed conditioning cache (only present when return_cache=True)",
    )
    tokens: Optional[EncodedTensor] = Field(
        None,
        description="Generated semantic tokens as 1-D int64 array (only present when return_tokens=True)",
    )
    token_rate: Optional[float] = Field(
        None,
        description="Token rate in Hz (only present when return_tokens=True)",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    model_version: Optional[str]
    config: Dict[str, Any]


class TokenExtractRequest(BaseModel):
    """Request for extracting 50Hz semantic tokens from audio."""

    audio: Union[EncodedAudio, str] = Field(
        ...,
        description="Audio input. Either Base64-encoded audio object or file path string.",
    )


class TokenExtractResponse(BaseModel):
    """Response for token extraction."""

    status: str = "success"
    codes: EncodedTensor = Field(
        ..., description="Extracted semantic codes (1-D int64)"
    )
    token_rate: float = Field(50.0, description="Token rate in Hz")
    num_tokens: int = Field(..., description="Number of tokens extracted")
    message: Optional[str] = None


class SynthesizeFromTokensRequest(BaseModel):
    """Request for synthesizing audio from pre-computed semantic tokens."""

    codes: EncodedTensor = Field(
        ..., description="Pre-computed semantic codes (1-D int64)"
    )
    text: str = Field(..., description="Text corresponding to the codes")
    spk_audio_prompt: Union[EncodedAudio, str] = Field(
        ..., description="Speaker reference audio"
    )

    # Emotion control (mutually exclusive)
    emo_audio_prompt: Optional[Union[EncodedAudio, str]] = Field(None)
    emo_text: Optional[str] = Field(None)
    emo_vector: Optional[List[float]] = Field(None)
    emo_alpha: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Emotion blending strength.",
    )

    # Generation parameters
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible inference.",
    )
    interval_silence: int = Field(200, ge=0)
    max_text_tokens_per_segment: int = Field(120, ge=10)

    # Output format
    output_format: str = Field("wav", description="'wav' or 'numpy'")

    # Per-pathway audio source overrides.
    # When set, the specified pathway tensor(s) are extracted from this audio
    # instead of spk_audio_prompt.  Default (None) means use spk_audio_prompt
    # for all pathways.
    mel_audio_source: Optional[Union[EncodedAudio, str]] = Field(
        None,
        description=(
            "Override audio for mel spectrogram extraction (S2Mel CFM reference). "
            "When None, spk_audio_prompt is used."
        ),
    )
    style_audio_source: Optional[Union[EncodedAudio, str]] = Field(
        None,
        description=(
            "Override audio for CAM++ style vector (s2mel_style) extraction. "
            "When None, spk_audio_prompt is used."
        ),
    )
    spk_cond_audio_source: Optional[Union[EncodedAudio, str]] = Field(
        None,
        description=(
            "Override audio for w2v-bert speaker conditioning (spk_cond) and "
            "length-regulated prompt (s2mel_prompt) extraction. "
            "spk_cond and s2mel_prompt are always kept together. "
            "When None, spk_audio_prompt is used."
        ),
    )

    # Cache control
    return_cache: bool = Field(
        False,
        description="If True, response includes precomputed conditioning cache.",
    )
    cache: Optional["InferenceCache"] = Field(
        None,
        description="Precomputed conditioning cache from a previous response.",
    )

    # Tracing
    trace_id: Optional[str] = Field(
        None,
        description="Caller-supplied trace ID for request tracing and profiling.",
    )

    @model_validator(mode="after")
    def check_emotion_mutual_exclusivity(self):
        emotion_methods = [
            self.emo_audio_prompt is not None,
            self.emo_text is not None,
            self.emo_vector is not None,
        ]
        if sum(emotion_methods) > 1:
            raise ValueError(
                "Provide at most one of: emo_audio_prompt, emo_text, emo_vector"
            )
        return self

    @model_validator(mode="after")
    def check_emo_vector_length(self):
        if self.emo_vector is not None and len(self.emo_vector) != 8:
            raise ValueError(
                f"emo_vector must have exactly 8 dimensions, got {len(self.emo_vector)}."
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "spk_audio_prompt": "/path/to/prompt.wav",
                "codes": {"data": "<base64-int64-tokens>", "shape": [100], "dtype": "int64"},
                "text": "Hello, this is a test.",
                "output_format": "wav",
            }
        }


class InfoResponse(BaseModel):
    """Model and service information."""

    service: str = "IndexTTS Synthesis Service"
    version: str = "1.0.0"
    model_version: Optional[str]
    sampling_rate: int
    emotion_vector_order: List[str] = [
        "happy",
        "angry",
        "sad",
        "afraid",
        "disgusted",
        "melancholic",
        "surprised",
        "calm",
    ]
    supported_output_formats: List[str] = ["wav", "numpy"]


# -----------------------------------------
# Audio Encoding/Decoding Utilities
# -----------------------------------------


def decode_audio_input(
    audio_input: Union[EncodedAudio, str],
    temp_dir: str,
    prefix: str = "audio",
) -> str:
    """
    Decode audio input to a file path.

    If input is a file path string, validate and return it.
    If input is EncodedAudio, decode to temp file and return path.
    """
    if isinstance(audio_input, str):
        # It's a file path
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        return audio_input

    # It's EncodedAudio - decode to temp file
    try:
        audio_bytes = base64.b64decode(audio_input.data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 audio: {e}")

    # Generate temp file path
    file_ext = audio_input.format.lower()
    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"
    temp_path = os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{file_ext}")

    # Write to temp file
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    return temp_path


def encode_numpy(arr: np.ndarray) -> EncodedTensor:
    """Encode numpy array to Base64-encoded format."""
    tensor_bytes = arr.tobytes()
    data_b64 = base64.b64encode(tensor_bytes).decode("utf-8")

    return EncodedTensor(
        data=data_b64,
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def decode_numpy(tensor: EncodedTensor) -> np.ndarray:
    """Decode Base64-encoded tensor to numpy array."""
    raw = base64.b64decode(tensor.data)
    arr = np.frombuffer(raw, dtype=np.dtype(tensor.dtype))
    return arr.reshape(tensor.shape)


def _serialize_cache(cache_dict: dict) -> InferenceCache:
    """Convert model-layer cache dict (torch tensors) to Pydantic model (base64).

    ``cache_dict`` has the shape returned by the model's ``infer()`` when
    ``return_cache=True``::

        {
            "speaker": {"spk_cond": Tensor, "s2mel_style": Tensor,
                        "s2mel_prompt": Tensor, "mel": Tensor},
            "emotion": {"emo_cond": Tensor} | None,
            "continuation": {"tokens": np.ndarray(int64), "text": str} | None,
        }
    """
    spk = cache_dict["speaker"]
    speaker = SpeakerCache(
        spk_cond=encode_numpy(spk["spk_cond"].cpu().float().numpy()),
        s2mel_style=encode_numpy(spk["s2mel_style"].cpu().float().numpy()),
        s2mel_prompt=encode_numpy(spk["s2mel_prompt"].cpu().float().numpy()),
        mel=encode_numpy(spk["mel"].cpu().float().numpy()),
    )
    emotion = None
    emo = cache_dict.get("emotion")
    if emo is not None and emo.get("emo_cond") is not None:
        emotion = EmotionCache(
            emo_cond=encode_numpy(emo["emo_cond"].cpu().float().numpy()),
        )
    continuation = None
    cont = cache_dict.get("continuation")
    if cont is not None and cont.get("tokens") is not None:
        tokens_np = cont["tokens"]
        if not isinstance(tokens_np, np.ndarray):
            tokens_np = np.asarray(tokens_np, dtype=np.int64)
        continuation = ContinuationCache(
            tokens=encode_numpy(tokens_np.astype(np.int64)),
            text=cont["text"],
        )
    return InferenceCache(speaker=speaker, emotion=emotion, continuation=continuation)


def _deserialize_cache(cache: InferenceCache, device) -> dict:
    """Convert Pydantic cache model back to model-layer dict (torch tensors on device).

    Returns dict in the same format expected by the model methods::

        {
            "speaker": {"spk_cond": Tensor, ...} | None,
            "emotion": {"emo_cond": Tensor} | None,
            "continuation": {"tokens": np.ndarray(int64), "text": str} | None,
        }
    """
    speaker_cache = {
        "spk_cond": torch.from_numpy(decode_numpy(cache.speaker.spk_cond)).to(device),
        "s2mel_style": torch.from_numpy(decode_numpy(cache.speaker.s2mel_style)).to(
            device
        ),
        "s2mel_prompt": torch.from_numpy(decode_numpy(cache.speaker.s2mel_prompt)).to(
            device
        ),
        "mel": torch.from_numpy(decode_numpy(cache.speaker.mel)).to(device),
    }
    emotion_cache = None
    if cache.emotion is not None:
        emotion_cache = {
            "emo_cond": torch.from_numpy(decode_numpy(cache.emotion.emo_cond)).to(
                device
            ),
        }
    continuation_cache = None
    if cache.continuation is not None:
        continuation_cache = {
            "tokens": decode_numpy(cache.continuation.tokens),  # stays as numpy int64
            "text": cache.continuation.text,
        }
    return {
        "speaker": speaker_cache,
        "emotion": emotion_cache,
        "continuation": continuation_cache,
    }


def cleanup_temp_files(paths: List[str]):
    """Clean up temporary files."""
    for path in paths:
        try:
            if path and os.path.exists(path) and TEMP_DIR and path.startswith(TEMP_DIR):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {path}: {e}")


def _normalize_emo_vector(
    emo_vector: List[float], apply_bias: bool = True
) -> List[float]:
    """Apply the upstream IndexTTS2 emotion normalization policy."""
    vec = [float(v) for v in emo_vector]
    if apply_bias:
        vec = [value * bias for value, bias in zip(vec, EMO_BIAS)]
    vec_sum = sum(vec)
    if vec_sum > 0.8:
        scale = 0.8 / vec_sum
        vec = [value * scale for value in vec]
    return vec


def _emotion_dict_to_vector(emo_dict: Dict[str, float]) -> List[float]:
    """Convert QwenEmotion output to the canonical 8-dim emotion vector order."""
    return [float(emo_dict.get(name, 0.0)) for name in EMOTION_VECTOR_ORDER]


def _resolve_effective_segment_alpha(
    request_alpha: float,
    segment: Optional[EmotionTaggedSegment] = None,
) -> float:
    """Resolve the effective segment-level public alpha value."""
    if segment is not None:
        if segment.alpha is not None:
            return segment.alpha
    return request_alpha


def _build_emotion_infer_kwargs(
    *,
    emo_audio_path: Optional[str],
    emo_text: Optional[str],
    emo_vector: Optional[List[float]],
    emo_alpha: float,
    text_for_emo: Optional[str],
) -> Dict[str, Any]:
    """Build model-layer kwargs using public alpha semantics."""
    infer_kwargs: Dict[str, Any] = {}
    if emo_audio_path is not None:
        infer_kwargs["emo_audio_prompt"] = emo_audio_path
        infer_kwargs["emo_alpha"] = emo_alpha
        return infer_kwargs

    if emo_text is not None:
        prompt_text = emo_text if emo_text is not None else text_for_emo
        emo_dict = TTS_MODEL.qwen_emo.inference(prompt_text or text_for_emo or "")
        emo_vector = _emotion_dict_to_vector(emo_dict)

    if emo_vector is not None:
        infer_kwargs["emo_vector"] = _normalize_emo_vector(emo_vector, apply_bias=True)
        infer_kwargs["emo_alpha"] = emo_alpha
        infer_kwargs["use_random"] = False

    return infer_kwargs


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    title="IndexTTS Synthesis Service",
    description="Voice cloning TTS with emotion control. Supports audio, text, and vector emotion prompts.",
    version="1.0.0",
)


def load_model(
    cfg_path: str,
    model_dir: str,
    device: Optional[str],
    use_fp16: bool,
    use_deepspeed: bool,
    use_accel: bool,
    use_cuda_kernel: bool,
    use_torch_compile: bool,
):
    """Load IndexTTS2 model."""
    global TTS_MODEL, MODEL_CONFIG

    if use_deepspeed:
        import deepspeed  # Fail before upstream can suppress an import failure.

    logger.info(f"Loading IndexTTS2 model from {model_dir}...")
    logger.info(f"  cfg_path: {cfg_path}")
    logger.info(f"  device: {device or 'auto'}")
    logger.info(f"  use_fp16: {use_fp16}")
    logger.info(f"  use_deepspeed: {use_deepspeed}")
    logger.info(f"  use_accel: {use_accel}")
    logger.info(f"  use_cuda_kernel: {use_cuda_kernel}")
    logger.info(f"  use_torch_compile: {use_torch_compile}")

    TTS_MODEL = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device=device,
        use_fp16=use_fp16,
        use_deepspeed=use_deepspeed,
        use_accel=use_accel,
        use_cuda_kernel=use_cuda_kernel,
        use_torch_compile=use_torch_compile,
    )

    if use_cuda_kernel and not TTS_MODEL.use_cuda_kernel:
        raise RuntimeError("CUDA kernel was requested but the model disabled it")
    if use_deepspeed and getattr(TTS_MODEL.gpt, "ds_engine", None) is None:
        raise RuntimeError("DeepSpeed was requested but its inference engine is unavailable")
    if use_accel and TTS_MODEL.gpt.accel_engine is None:
        raise RuntimeError("Acceleration was requested but its inference engine is unavailable")

    MODEL_CONFIG = {
        "cfg_path": cfg_path,
        "model_dir": model_dir,
        "device": TTS_MODEL.device,
        "use_fp16": TTS_MODEL.use_fp16,
        "use_deepspeed": use_deepspeed,
        "use_accel": use_accel,
        "use_cuda_kernel": TTS_MODEL.use_cuda_kernel,
        "use_torch_compile": use_torch_compile,
    }

    logger.info("✓ IndexTTS2 model loaded successfully")


# -----------------------------------------
# Continuation-Based Emotion-Tagged Synthesis
# -----------------------------------------


def _check_aligner_health(aligner_url: str):
    """Check if the aligner gateway service is reachable. Raises HTTPException if not."""
    import requests as req

    try:
        resp = req.get(f"{aligner_url}/health", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Aligner service at {aligner_url} is not available "
            f"(required for generation_paradigm='continuation'): {e}",
        )


def _build_query(text: str, target_text: str) -> str:
    idx = text.find(target_text)
    if idx < 0:
        raise ValueError(
            f"Cannot build query because target text was not found: {target_text!r}"
        )
    return (
        text[:idx]
        + "<q>"
        + text[idx : idx + len(target_text)]
        + "</q>"
        + text[idx + len(target_text) :]
    )


def _align_audio_text(
    audio_path: str,
    text: str,
    aligner_url: str,
    *,
    queries: Optional[List[str]] = None,
    aligner_backend: str = "auto",
    aligner_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the aligner gateway with an explicit backend and optional queries."""
    import requests as req

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    payload = {
        "backend": aligner_backend,
        "audio": {
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": "wav",
        },
        "text": text,
    }
    if aligner_language is not None:
        payload["language"] = aligner_language
    if queries is not None:
        payload["queries"] = queries
    resp = req.post(f"{aligner_url}/align", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Alignment failed: {resp.status_code} - {resp.text}")
    return resp.json()


def _query_boundary_time(
    response: Dict[str, Any], *, query_count: int, query_index: int = 0
) -> float:
    answers = response.get("answers")
    if not isinstance(answers, list) or len(answers) != query_count:
        raise ValueError(
            f"Aligner query response missing answers: expected {query_count}, got {answers!r}"
        )
    answer = answers[query_index]
    end_sec = float(answer["end"])
    start_sec = float(answer["start"])
    if end_sec <= start_sec:
        raise ValueError(f"Invalid aligner answer span: {answer!r}")
    return end_sec + 0.02


def _trim_audio_at_time(src_path: str, end_secs: float, dst_path: str) -> float:
    """Trim audio to first end_secs seconds. Returns actual duration kept."""
    waveform, sr = torchaudio.load(src_path)
    keep_samples = min(int(end_secs * sr), waveform.shape[-1])
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    torchaudio.save(dst_path, waveform[:, :keep_samples], sr)
    return keep_samples / sr


def _resolve_continuation_debug_dir(
    request: "SynthesisRequest", work_dir: str
) -> Tuple[Optional[Path], bool]:
    if not request.continuation_debug:
        return None, False
    if request.continuation_debug_dir is not None:
        debug_dir = Path(request.continuation_debug_dir).expanduser().resolve()
    else:
        debug_dir = Path(work_dir) / "continuation_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    work_dir_path = Path(work_dir).resolve()
    preserve_work_dir = debug_dir.resolve().is_relative_to(work_dir_path)
    return debug_dir, preserve_work_dir


def _save_continuation_debug_artifacts(
    debug_dir: Optional[Path],
    step_metadatas: List[Dict[str, Any]],
    sample_rate: int,
    final_wav_path: str,
    request_summary: Dict[str, Any],
    final_output_path: Optional[Path] = None,
) -> None:
    if debug_dir is None:
        return

    saved_steps = []
    for step in step_metadatas:
        step_copy = dict(step)
        source_step_wav = step_copy.get("step_wav")
        if source_step_wav:
            source_path = Path(source_step_wav)
            copied_path = debug_dir / source_path.name
            shutil.copy2(source_path, copied_path)
            step_copy["debug_step_wav"] = str(copied_path)
        saved_steps.append(step_copy)

    final_waveform, final_sr = torchaudio.load(final_wav_path)
    torchaudio.save(str(debug_dir / "final_unsmoothed.wav"), final_waveform, final_sr)
    summary = {
        "request": request_summary,
        "sample_rate": sample_rate,
        "final_unsmoothed_wav": str(debug_dir / "final_unsmoothed.wav"),
        "steps": saved_steps,
    }
    if final_output_path is not None:
        summary["final_output"] = str(final_output_path)
    (debug_dir / "continuation_debug_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _join_segment_texts(texts: List[str]) -> str:
    """Join tagged text segments using the same continuation spacing rule."""
    joined = ""
    for text in texts:
        joined = TTS_MODEL._join_continuation_text(joined, text)
    return joined


def _synthesize_emotion_tagged_continuation(
    request: "SynthesisRequest",
    spk_audio_path: str,
    segments: List[EmotionTaggedSegment],
) -> Tuple[int, np.ndarray, Optional[List[Dict[str, Any]]]]:
    """
    Continuation-based emotion-tagged synthesis with direct token passing.

    Each step synthesizes the full remaining text. Step 0 uses standard infer()
    with no continuation prefix; later steps use infer() with prefix_tokens /
    prefix_text. After alignment finds the current segment boundary, tokens are
    truncated there and reused as the next continuation prefix.

    The aligner gateway with backend auto is used to resolve continuation
    boundary queries in the generated audio, and tokens are proportionally
    truncated at those boundaries.

    The final output contains segments 1..N with no extraneous prompt prefix.

    Returns (sample_rate, wav_data_int16, None).
    """
    aligner_url = request.aligner_url

    # Create a temp working directory for intermediate files
    work_dir = tempfile.mkdtemp(prefix="emo_chain_", dir=TEMP_DIR)
    continuation_debug_dir, preserve_work_dir = _resolve_continuation_debug_dir(
        request, work_dir
    )

    try:
        all_segment_texts = [segment.text for segment in segments]
        should_smooth_continuation_boundaries = (
            request.smooth_continuation_boundaries and len(segments) > 1
        )

        # Validate segments
        for idx, segment in enumerate(segments):
            seg_text = segment.text
            if not seg_text or not seg_text.strip():
                raise ValueError(f"Segment {idx} text cannot be empty")

        step_metadatas = []
        smoothing_boundary_positions = []
        request_summary = {
            "text": request.text,
            "generation_paradigm": request.generation_paradigm,
            "aligner_url": request.aligner_url,
            "aligner_backend": request.aligner_backend,
            "aligner_language": request.aligner_language,
            "smooth_continuation_boundaries": request.smooth_continuation_boundaries,
            "smooth_radius_sec": request.smooth_radius_sec,
            "seed": request.seed,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
            "max_text_tokens_per_segment": request.max_text_tokens_per_segment,
            "segments": [
                {
                    "emotion": segment.emotion,
                    "alpha": segment.alpha,
                    "text": segment.text,
                }
                for segment in segments
            ],
        }

        # Track continuation state: tokens and text accumulated so far
        current_prefix_tokens = None  # np.ndarray of int64 codes
        current_prefix_text = ""  # accumulated transcript

        for i, segment in enumerate(segments):
            emo_name = segment.emotion
            seg_text = segment.text
            is_last = i == len(segments) - 1
            seg_alpha = None

            logger.info(
                f"Continuation step {i + 1}/{len(segments)}: "
                f"<{emo_name or 'neutral'}> '{seg_text[:40]}...'"
            )

            # Build emotion kwargs
            emo_kwargs = {}
            if emo_name and emo_name in EMO_PRESETS:
                seg_alpha = _resolve_effective_segment_alpha(request.emo_alpha, segment)
                emo_kwargs = _build_emotion_infer_kwargs(
                    emo_audio_path=None,
                    emo_text=None,
                    emo_vector=EMO_PRESETS[emo_name],
                    emo_alpha=seg_alpha,
                    text_for_emo=seg_text,
                )

            step_wav = os.path.join(
                work_dir, f"step{i:02d}_{emo_name or 'neutral'}_full.wav"
            )

            remaining_text = _join_segment_texts(all_segment_texts[i:])
            has_prefix = current_prefix_tokens is not None
            prefix_text_used = current_prefix_text if has_prefix else None
            infer_full_text = TTS_MODEL._join_continuation_text(
                current_prefix_text, remaining_text
            )
            aligner_full_text = infer_full_text
            boundary_text = TTS_MODEL._join_continuation_text(
                current_prefix_text, seg_text
            )

            infer_kwargs = {
                "spk_audio_prompt": spk_audio_path,
                "text": remaining_text,
                "output_path": step_wav,
                "verbose": False,
                "return_tokens": True,
                "interval_silence": 0,
                "max_mel_tokens": request.max_mel_tokens,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "temperature": request.temperature,
                "repetition_penalty": request.repetition_penalty,
                **emo_kwargs,
            }
            if has_prefix:
                infer_kwargs["prefix_tokens"] = current_prefix_tokens
                infer_kwargs["prefix_text"] = current_prefix_text

            with INFERENCE_LOCK:
                if request.seed is not None:
                    _set_seed(request.seed + i)
                infer_result = TTS_MODEL.infer(**infer_kwargs)

            if infer_result is None:
                raise RuntimeError(
                    f"Failed to synthesize segment {i}: '{seg_text[:50]}...'"
                )

            # Unpack: infer() with return_tokens returns (output_path, tokens_out)
            _, tokens_out = infer_result
            full_tokens = None
            if tokens_out is not None and tokens_out.get("tokens") is not None:
                full_tokens = tokens_out["tokens"]

            metadata = {
                "mode": "token_continuation" if has_prefix else "full_text_seed",
                "num_prefix_tokens": len(current_prefix_tokens) if has_prefix else 0,
                "num_full_tokens": len(full_tokens) if full_tokens is not None else 0,
            }

            step_info = {
                "step": i,
                "emotion": emo_name or "neutral",
                "segment_alpha": seg_alpha,
                "segment_text": seg_text,
                "remaining_text": remaining_text,
                "infer_input_text": remaining_text,
                "infer_full_text": infer_full_text,
                "prefix_text_used": prefix_text_used,
                "aligner_full_text": aligner_full_text,
                "boundary_text_for_alignment": boundary_text,
                "metadata": metadata,
                "seed_used": request.seed + i if request.seed is not None else None,
                "step_wav": step_wav,
            }

            if is_last:
                if should_smooth_continuation_boundaries and has_prefix:
                    logger.info(
                        f"  Aligning final continuation step {i + 1} to locate splice point"
                    )
                    final_alignment = _align_audio_text(
                        step_wav,
                        aligner_full_text,
                        aligner_url,
                        queries=[_build_query(aligner_full_text, current_prefix_text)],
                        aligner_backend=request.aligner_backend,
                        aligner_language=request.aligner_language,
                    )
                    splice_time = _query_boundary_time(
                        final_alignment,
                        query_count=1,
                    )
                    smoothing_boundary_positions.append(splice_time)
                    step_info["splice_time"] = splice_time
                    logger.info(f"  Splice point at {splice_time:.3f}s")

                step_info["action"] = "final"
                step_metadatas.append(step_info)
                logger.info(f"  Final segment → {step_wav}")
                break

            logger.info(f"  Aligning to find boundary at '{boundary_text[-30:]}'...")
            queries = [_build_query(aligner_full_text, boundary_text)]
            if should_smooth_continuation_boundaries and has_prefix:
                queries.insert(0, _build_query(aligner_full_text, current_prefix_text))
            alignment_response = _align_audio_text(
                step_wav,
                aligner_full_text,
                aligner_url,
                queries=queries,
                aligner_backend=request.aligner_backend,
                aligner_language=request.aligner_language,
            )
            if should_smooth_continuation_boundaries and has_prefix:
                splice_time = _query_boundary_time(
                    alignment_response,
                    query_count=len(queries),
                    query_index=0,
                )
                smoothing_boundary_positions.append(splice_time)
                step_info["splice_time"] = splice_time
                logger.info(f"  Splice point at {splice_time:.3f}s")
            boundary_time = _query_boundary_time(
                alignment_response,
                query_count=len(queries),
                query_index=(
                    1 if should_smooth_continuation_boundaries and has_prefix else 0
                ),
            )
            logger.info(f"  Boundary at {boundary_time:.3f}s")

            # --- Proportional token truncation ---
            # Instead of truncating audio and re-extracting tokens (lossy),
            # we truncate tokens proportionally based on the boundary position.
            if full_tokens is not None:
                # Get total audio duration for proportion calculation
                waveform, sr = torchaudio.load(step_wav)
                total_duration = waveform.shape[-1] / sr

                if total_duration > 0:
                    boundary_fraction = boundary_time / total_duration
                    num_tokens_at_boundary = int(len(full_tokens) * boundary_fraction)
                    # Clamp to valid range
                    num_tokens_at_boundary = max(
                        1, min(num_tokens_at_boundary, len(full_tokens))
                    )
                    current_prefix_tokens = full_tokens[:num_tokens_at_boundary]
                    current_prefix_text = boundary_text
                    logger.info(
                        f"  Token truncation: {len(full_tokens)} → "
                        f"{num_tokens_at_boundary} tokens "
                        f"(boundary fraction: {boundary_fraction:.3f})"
                    )
                else:
                    # Fallback: use all tokens
                    current_prefix_tokens = full_tokens
                    current_prefix_text = boundary_text
                    logger.warning("  Zero-duration audio; using all tokens as prefix")
            else:
                # Fallback: extract tokens from truncated audio (legacy path)
                logger.warning(
                    "  No tokens from infer(); falling back to audio extraction"
                )
                truncated_wav = os.path.join(
                    work_dir,
                    f"step{i:02d}_{emo_name or 'neutral'}_truncated.wav",
                )
                _trim_audio_at_time(step_wav, boundary_time, truncated_wav)
                current_prefix_tokens, _ = TTS_MODEL.extract_semantic_tokens(
                    truncated_wav
                )
                current_prefix_text = boundary_text

            step_info["action"] = "truncate"
            step_info["boundary_time"] = boundary_time
            step_info["num_tokens_truncated"] = len(current_prefix_tokens)
            step_metadatas.append(step_info)

        # --- Build final output ---
        # For single-segment: the infer() output (step 0) is the complete result.
        # For multi-segment: the last infer() output contains all segments
        # synthesized through a single AR+vocode pass with prefix tokens,
        # producing seamless transitions.
        final_step_idx = len(step_metadatas) - 1
        final_wav_path = os.path.join(
            work_dir,
            f"step{final_step_idx:02d}_"
            f"{step_metadatas[final_step_idx]['emotion']}_full.wav",
        )
        final_waveform, final_sr = torchaudio.load(final_wav_path)

        # Convert to int16
        if final_waveform.dtype == torch.int16:
            wav_data = final_waveform.squeeze().numpy()
        elif final_waveform.abs().max() <= 1.0:
            wav_data = (
                (final_waveform.squeeze().numpy() * 32767.0)
                .clip(-32767, 32767)
                .astype(np.int16)
            )
        else:
            wav_data = (
                final_waveform.squeeze().numpy().clip(-32767, 32767).astype(np.int16)
            )

        logger.info(
            f"Continuation chain complete: {len(step_metadatas)} steps, "
            f"{len(wav_data) / final_sr:.2f}s"
        )

        if should_smooth_continuation_boundaries:
            logger.info(
                "Smoothing %d continuation splice points%s",
                len(smoothing_boundary_positions),
                (
                    f" with explicit radius {request.smooth_radius_sec:.3f}s"
                    if request.smooth_radius_sec is not None
                    else " with smoother default regen margin"
                ),
            )
            final_sr, wav_data = _smooth_continuation_boundaries(
                request=request,
                wav_data=wav_data,
                sample_rate=final_sr,
                segments=segments,
                boundary_positions=smoothing_boundary_positions,
            )

        _save_continuation_debug_artifacts(
            debug_dir=continuation_debug_dir,
            step_metadatas=step_metadatas,
            sample_rate=final_sr,
            final_wav_path=final_wav_path,
            request_summary=request_summary,
        )

        return final_sr, wav_data, None

    finally:
        # Cleanup intermediate files
        try:
            if not preserve_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup work_dir {work_dir}: {e}")


def _check_smoother_health(smoother_url: str):
    """Check the Boundary Pause Smoother service is reachable."""
    import requests as req

    try:
        resp = req.get(f"{smoother_url}/health", timeout=5)
        data = resp.json()
        if data.get("status") != "healthy":
            raise RuntimeError(f"Smoother unhealthy: {data}")
    except req.ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Boundary Pause Smoother service unavailable at {smoother_url}: {e}",
        )


def _smooth_continuation_boundaries(
    request: "SynthesisRequest",
    wav_data: np.ndarray,
    sample_rate: int,
    segments: List[EmotionTaggedSegment],
    boundary_positions: List[float],
) -> Tuple[int, np.ndarray]:
    """Smooth continuation splice points via the smoother regen-only fast path."""
    import base64
    import requests as req
    import soundfile as sf

    if len(boundary_positions) != len(segments) - 1:
        raise RuntimeError(
            "Continuation smoothing boundary count mismatch: "
            f"got {len(boundary_positions)} positions for {len(segments)} segments"
        )

    audio_float = wav_data.astype(np.float32)
    if wav_data.dtype == np.int16:
        audio_float = audio_float / 32767.0

    buf = io.BytesIO()
    sf.write(buf, audio_float, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    points = []
    for idx, boundary_sec in enumerate(boundary_positions):
        point = {
            "mode": "smooth_only",
            "position_sec": boundary_sec,
        }
        if request.smooth_radius_sec is not None:
            point["regen_margin_sec"] = request.smooth_radius_sec
        points.append(point)

    payload = {
        "audio": {"data": audio_b64, "format": "wav"},
        "text": _join_segment_texts([segment.text for segment in segments]),
        "points": points,
        "output_format": "wav",
        "f5_seed": request.seed,
        "debug": request.smoother_debug,
        "aligner_backend": request.aligner_backend,
    }
    if request.aligner_language is not None:
        payload["language"] = request.aligner_language
    if request.smoother_debug_dir is not None:
        payload["debug_dir"] = request.smoother_debug_dir
    if request.trace_id is not None:
        payload["trace_id"] = request.trace_id

    resp = req.post(f"{request.smoother_url}/smooth", json=payload, timeout=300)
    if resp.status_code != 200:
        logger.error(
            f"Continuation boundary smoother failed ({resp.status_code}): {resp.text[:300]}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Continuation boundary smoothing failed: {resp.text[:200]}",
        )

    smoothed_buf = io.BytesIO(resp.content)
    smoothed_audio, smoothed_sr = sf.read(smoothed_buf, dtype="float32")
    if smoothed_audio.ndim > 1:
        smoothed_audio = smoothed_audio.mean(axis=1)
    smoothed_wav = (smoothed_audio * 32767.0).clip(-32767, 32767).astype(np.int16)
    return smoothed_sr, smoothed_wav


# -----------------------------------------
# Endpoints
# -----------------------------------------


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "IndexTTS Synthesis Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if TTS_MODEL is not None else "not_ready",
        model_loaded=TTS_MODEL is not None,
        device=MODEL_CONFIG.get("device", "unknown"),
        model_version=(
            str(getattr(TTS_MODEL, "model_version", None)) if TTS_MODEL else None
        ),
        config=MODEL_CONFIG,
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """Get model and service information."""
    return InfoResponse(
        model_version=(
            str(getattr(TTS_MODEL, "model_version", None)) if TTS_MODEL else None
        ),
        sampling_rate=SAMPLING_RATE,
    )


@app.post("/synthesize")
@serialized_model_request
def synthesize(request: SynthesisRequest):
    """
    Synthesize speech from text with voice cloning and emotion control.

    Speaker prompt (required):
        - spk_audio_prompt: Reference audio for voice cloning

    Emotion control (optional, mutually exclusive):
        - emo_audio_prompt: Reference audio for emotion
        - emo_text: Text for automatic emotion detection
        - emo_vector: Direct 8-dim emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        - use_emotion_tags: Parse inline emotion tags in text like '<happy>Hello!</happy><calm>World.</calm>'
          Each tagged segment is synthesized separately with its emotion and concatenated.

    Returns:
        - If output_format='wav': Binary WAV file response
        - If output_format='numpy': JSON with encoded waveform array
    """
    _t0 = time.time()
    if TTS_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    temp_files = []

    tokens_out = None
    cache_pydantic = None

    try:
        # Decode speaker audio prompt
        spk_audio_path = decode_audio_input(
            request.spk_audio_prompt, TEMP_DIR, prefix="spk"
        )
        if isinstance(request.spk_audio_prompt, EncodedAudio):
            temp_files.append(spk_audio_path)

        # Decode emotion audio prompt if provided
        emo_audio_path = None
        if request.emo_audio_prompt is not None:
            emo_audio_path = decode_audio_input(
                request.emo_audio_prompt, TEMP_DIR, prefix="emo"
            )
            if isinstance(request.emo_audio_prompt, EncodedAudio):
                temp_files.append(emo_audio_path)

        # --- Emotion-tagged text mode ---
        if request.use_emotion_tags:
            segments = parse_emotion_tags(request.text)
            if not segments:
                raise HTTPException(
                    status_code=400,
                    detail="use_emotion_tags=True but no text segments found. "
                    "Use tags like <happy>text</happy>.",
                )
            if _has_untagged_emotion_text(request.text):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "use_emotion_tags=True requires the full text to be explicitly tagged. "
                        "Wrap neutral spans with <none>...</none> instead of leaving them untagged."
                    ),
                )
            logger.info(
                f"Emotion-tagged synthesis: {len(segments)} segments, "
                f"tags={[segment.emotion or 'neutral' for segment in segments]}, "
                f"paradigm={request.generation_paradigm}"
            )

            if request.generation_paradigm == "continuation":
                # --- Continuation paradigm ---
                _check_aligner_health(request.aligner_url)
                if request.smooth_continuation_boundaries and len(segments) > 1:
                    _check_smoother_health(request.smoother_url)
                sample_rate, wav_data, _ = _synthesize_emotion_tagged_continuation(
                    request,
                    spk_audio_path,
                    segments,
                )
                if wav_data.dtype == np.int16:
                    wav_data_float = wav_data.astype(np.float32) / 32767.0
                else:
                    wav_data_float = wav_data.astype(np.float32)

            else:
                # --- Independent paradigm (original behavior) ---
                wav_parts = []
                silence_samples = int(SAMPLING_RATE * request.interval_silence / 1000)
                silence = np.zeros(silence_samples, dtype=np.float32)

                for seg_idx, segment in enumerate(segments):
                    emo_name = segment.emotion
                    seg_text = segment.text
                    seg_kwargs = {
                        "spk_audio_prompt": spk_audio_path,
                        "text": seg_text,
                        "output_path": None,
                        "interval_silence": 0,
                        "verbose": False,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                        "temperature": request.temperature,
                        "repetition_penalty": request.repetition_penalty,
                        "max_mel_tokens": request.max_mel_tokens,
                        "max_text_tokens_per_segment": request.max_text_tokens_per_segment,
                    }
                    if emo_name and emo_name in EMO_PRESETS:
                        seg_alpha = _resolve_effective_segment_alpha(
                            request.emo_alpha, segment
                        )
                        seg_kwargs.update(
                            _build_emotion_infer_kwargs(
                                emo_audio_path=None,
                                emo_text=None,
                                emo_vector=EMO_PRESETS[emo_name],
                                emo_alpha=seg_alpha,
                                text_for_emo=seg_text,
                            )
                        )
                    elif emo_name:
                        logger.warning(
                            f"Unknown emotion tag '{emo_name}', using neutral"
                        )

                    logger.info(
                        f"  Segment {seg_idx}: <{emo_name or 'neutral'}> '{seg_text[:40]}...'"
                    )
                    with INFERENCE_LOCK:
                        if request.seed is not None:
                            _set_seed(request.seed + seg_idx)
                        seg_result = TTS_MODEL.infer(**seg_kwargs)
                    if seg_result is None:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Segment {seg_idx} ({emo_name}) inference returned no result",
                        )
                    _, seg_wav = seg_result
                    if seg_wav.ndim == 2:
                        seg_wav = seg_wav.squeeze()
                    if seg_wav.dtype == np.int16:
                        seg_wav = seg_wav.astype(np.float32) / 32767.0
                    else:
                        seg_wav = seg_wav.astype(np.float32)

                    if wav_parts and silence_samples > 0:
                        wav_parts.append(silence)
                    wav_parts.append(seg_wav)

                sample_rate = SAMPLING_RATE
                wav_data_float = np.concatenate(wav_parts)
                wav_data = (
                    (wav_data_float * 32767.0).clip(-32767, 32767).astype(np.int16)
                )
        else:
            # --- Standard (non-tagged) synthesis ---
            # Prepare inference kwargs
            infer_kwargs = {
                "spk_audio_prompt": spk_audio_path,
                "text": request.text,
                "output_path": None,  # Return audio directly
                "interval_silence": request.interval_silence,
                "verbose": False,
                # Generation parameters
                "top_p": request.top_p,
                "top_k": request.top_k,
                "temperature": request.temperature,
                "repetition_penalty": request.repetition_penalty,
                "max_mel_tokens": request.max_mel_tokens,
                "max_text_tokens_per_segment": request.max_text_tokens_per_segment,
            }

            infer_kwargs.update(
                _build_emotion_infer_kwargs(
                    emo_audio_path=emo_audio_path,
                    emo_text=request.emo_text,
                    emo_vector=request.emo_vector,
                    emo_alpha=request.emo_alpha,
                    text_for_emo=request.text,
                )
            )

            # Deserialize cache from request if provided
            if request.cache is not None:
                deserialized = _deserialize_cache(request.cache, TTS_MODEL.device)
                infer_kwargs["speaker_cache"] = deserialized["speaker"]
                infer_kwargs["emotion_cache"] = deserialized.get("emotion")
                # Continuation tokens from previous generation
                cont = deserialized.get("continuation")
                if cont is not None:
                    infer_kwargs["prefix_tokens"] = cont["tokens"]
                    infer_kwargs["prefix_text"] = cont["text"]

            infer_kwargs["return_cache"] = request.return_cache
            infer_kwargs["return_tokens"] = request.return_tokens
            infer_kwargs["skip_vocode"] = request.skip_vocode

            # Run inference
            logger.info(
                f"Running synthesis: text='{request.text[:50]}...', emotion_mode={_get_emotion_mode(request)}"
            )

            with INFERENCE_LOCK:
                if request.seed is not None:
                    _set_seed(request.seed)
                result = TTS_MODEL.infer(**infer_kwargs)

            # Unpack result — format depends on return_cache, return_tokens, skip_vocode
            if result is None:
                raise HTTPException(
                    status_code=500, detail="Inference returned no result"
                )

            # Layer 1: strip cache wrapper
            if request.return_cache:
                inner_result, cache_out = result
                cache_pydantic = _serialize_cache(cache_out)
            else:
                inner_result = result

            # Layer 2: handle skip_vocode (tokens-only dict) vs normal
            if request.skip_vocode:
                # inner_result is {"tokens": np.ndarray, "token_rate": float}
                if isinstance(inner_result, dict) and "tokens" in inner_result:
                    tokens_out = inner_result
                    # No audio to return — respond with tokens JSON
                    _elapsed = time.time() - _t0
                    tokens_resp = {
                        "status": "success",
                        "tokens": encode_numpy(
                            tokens_out["tokens"].astype(np.int64)
                        ).model_dump(),
                        "token_rate": tokens_out.get("token_rate", 50.0),
                        "response_time": round(_elapsed, 3),
                    }
                    if request.trace_id:
                        tokens_resp["trace_id"] = request.trace_id
                    if cache_pydantic is not None:
                        tokens_resp["cache"] = json.loads(
                            cache_pydantic.model_dump_json()
                        )
                    return JSONResponse(content=tokens_resp)
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"skip_vocode=True but got unexpected result type: {type(inner_result)}",
                    )

            # Layer 3: handle return_tokens (tuple of audio + tokens)
            if request.return_tokens:
                if isinstance(inner_result, tuple) and len(inner_result) == 2:
                    audio_result, tokens_out = inner_result
                    sample_rate, wav_data = audio_result
                else:
                    # Shouldn't happen, but gracefully degrade
                    sample_rate, wav_data = inner_result
            else:
                sample_rate, wav_data = inner_result

            # wav_data is numpy array of shape (n_samples,) or (n_samples, 1)
            if wav_data.ndim == 2:
                wav_data = wav_data.squeeze()

            # Convert to float32 for consistency
            if wav_data.dtype == np.int16:
                wav_data_float = wav_data.astype(np.float32) / 32767.0
            else:
                wav_data_float = wav_data.astype(np.float32)

        duration_seconds = len(wav_data) / sample_rate

        logger.info(f"Synthesis complete: {duration_seconds:.2f}s audio generated")

        _elapsed = time.time() - _t0

        # For emotion-tagged modes, cache is not supported
        if request.use_emotion_tags:
            cache_pydantic = None

        # Return based on output format
        if request.output_format == "wav":
            # Convert to WAV bytes
            wav_tensor = torch.from_numpy(wav_data).unsqueeze(0)
            if wav_tensor.dtype == torch.int16:
                wav_tensor = wav_tensor.float() / 32767.0

            buffer = io.BytesIO()
            torchaudio.save(buffer, wav_tensor, sample_rate, format="wav")
            buffer.seek(0)
            wav_bytes = buffer.read()

            headers = {
                "Content-Disposition": "attachment; filename=synthesized.wav",
                "X-Sample-Rate": str(sample_rate),
                "X-Duration-Seconds": f"{duration_seconds:.3f}",
                "X-Response-Time": f"{_elapsed:.3f}",
            }
            if request.trace_id:
                headers["X-Trace-Id"] = request.trace_id

            # Determine if we need multipart (cache and/or tokens)
            has_tokens = tokens_out is not None
            has_cache = cache_pydantic is not None
            need_multipart = has_cache or has_tokens
            if need_multipart:
                boundary = "----IndexTTSCacheBoundary"
                parts = []
                # Part 1: WAV audio
                parts.append(
                    f"--{boundary}\r\n"
                    f"Content-Type: audio/wav\r\n"
                    f"Content-Disposition: attachment; filename=synthesized.wav\r\n"
                    f"\r\n".encode()
                )
                parts.append(wav_bytes)

                if has_cache and not has_tokens:
                    # Backward-compatible: Part 2 = raw cache JSON
                    # (existing clients expect this format)
                    parts.append(
                        f"\r\n--{boundary}\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Disposition: inline; name=cache\r\n"
                        f"\r\n".encode()
                    )
                    parts.append(cache_pydantic.model_dump_json().encode())
                else:
                    # New format: Part 2 = metadata JSON with cache and/or tokens
                    meta: Dict[str, Any] = {}
                    if has_cache:
                        meta["cache"] = json.loads(cache_pydantic.model_dump_json())
                    if has_tokens:
                        meta["tokens"] = encode_numpy(
                            tokens_out["tokens"].astype(np.int64)
                        ).model_dump()
                        meta["token_rate"] = tokens_out.get("token_rate", 50.0)
                    parts.append(
                        f"\r\n--{boundary}\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Disposition: inline; name=metadata\r\n"
                        f"\r\n".encode()
                    )
                    parts.append(json.dumps(meta).encode())

                parts.append(f"\r\n--{boundary}--\r\n".encode())

                return Response(
                    content=b"".join(parts),
                    media_type=f"multipart/mixed; boundary={boundary}",
                    headers=headers,
                )

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers=headers,
            )

        elif request.output_format == "numpy":
            # Encode tokens if available
            tokens_encoded = None
            token_rate_val = None
            if tokens_out is not None:
                tokens_encoded = encode_numpy(tokens_out["tokens"].astype(np.int64))
                token_rate_val = tokens_out.get("token_rate", 50.0)

            return SynthesisResponse(
                status="success",
                output_format="numpy",
                tensor=encode_numpy(wav_data_float),
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                response_time=round(_elapsed, 3),
                trace_id=request.trace_id,
                message=f"Generated waveform shape: {list(wav_data_float.shape)}",
                cache=cache_pydantic,
                tokens=tokens_encoded,
                token_rate=token_rate_val,
            )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
    finally:
        # Cleanup temp files
        cleanup_temp_files(temp_files)


@app.post("/extract_tokens", response_model=TokenExtractResponse)
@serialized_model_request
def extract_tokens(request: TokenExtractRequest):
    """
    Extract 50Hz semantic tokens from audio.

    Returns the discrete codes as a 1-D int64 encoded tensor, plus token_rate.
    """
    if TTS_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    temp_files = []
    try:
        audio_path = decode_audio_input(request.audio, TEMP_DIR, prefix="tok")
        if isinstance(request.audio, EncodedAudio):
            temp_files.append(audio_path)

        logger.info(f"Extracting tokens from: {audio_path}")
        with INFERENCE_LOCK:
            codes_np, token_rate = TTS_MODEL.extract_semantic_tokens(audio_path)
        logger.info(f"Extracted {len(codes_np)} tokens at {token_rate} Hz")

        return TokenExtractResponse(
            status="success",
            codes=encode_numpy(codes_np),
            token_rate=token_rate,
            num_tokens=len(codes_np),
            message=f"Extracted {len(codes_np)} tokens at {token_rate} Hz",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Token extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Token extraction failed: {e}")
    finally:
        cleanup_temp_files(temp_files)


@app.post("/synthesize_from_tokens")
@serialized_model_request
def synthesize_from_tokens(request: SynthesizeFromTokensRequest):
    """
    Synthesize audio from pre-computed 50Hz semantic tokens.

    The GPT forward pass is used to produce the latent representation, then
    s2mel + BigVGAN decodes to audio. This endpoint is useful for
    token-level manipulations (e.g. speed change via token stretching).

    Returns:
        - If output_format='wav': Binary WAV file response
        - If output_format='numpy': JSON with encoded waveform array
    """
    if TTS_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    temp_files = []
    try:
        # Decode codes
        codes_np = decode_numpy(request.codes).astype(np.int64)

        # Decode speaker audio prompt
        spk_audio_path = decode_audio_input(
            request.spk_audio_prompt, TEMP_DIR, prefix="spk"
        )
        if isinstance(request.spk_audio_prompt, EncodedAudio):
            temp_files.append(spk_audio_path)

        # Decode emotion audio prompt if provided
        emo_audio_path = None
        if request.emo_audio_prompt is not None:
            emo_audio_path = decode_audio_input(
                request.emo_audio_prompt, TEMP_DIR, prefix="emo"
            )
            if isinstance(request.emo_audio_prompt, EncodedAudio):
                temp_files.append(emo_audio_path)

        # Decode per-pathway override audio sources if provided
        mel_audio_path = None
        if request.mel_audio_source is not None:
            mel_audio_path = decode_audio_input(
                request.mel_audio_source, TEMP_DIR, prefix="mel_src"
            )
            if isinstance(request.mel_audio_source, EncodedAudio):
                temp_files.append(mel_audio_path)

        style_audio_path = None
        if request.style_audio_source is not None:
            style_audio_path = decode_audio_input(
                request.style_audio_source, TEMP_DIR, prefix="style_src"
            )
            if isinstance(request.style_audio_source, EncodedAudio):
                temp_files.append(style_audio_path)

        spk_cond_audio_path = None
        if request.spk_cond_audio_source is not None:
            spk_cond_audio_path = decode_audio_input(
                request.spk_cond_audio_source, TEMP_DIR, prefix="spk_cond_src"
            )
            if isinstance(request.spk_cond_audio_source, EncodedAudio):
                temp_files.append(spk_cond_audio_path)

        # Prepare kwargs
        infer_kwargs = {
            "codes_np": codes_np,
            "text": request.text,
            "spk_audio_prompt": spk_audio_path,
            "interval_silence": request.interval_silence,
            "max_text_tokens_per_segment": request.max_text_tokens_per_segment,
        }

        infer_kwargs.update(
            _build_emotion_infer_kwargs(
                emo_audio_path=emo_audio_path,
                emo_text=request.emo_text,
                emo_vector=request.emo_vector,
                emo_alpha=request.emo_alpha,
                text_for_emo=request.text,
            )
        )

        # Deserialize cache from request if provided
        if request.cache is not None:
            deserialized = _deserialize_cache(request.cache, TTS_MODEL.device)
            infer_kwargs["speaker_cache"] = deserialized["speaker"]
            infer_kwargs["emotion_cache"] = deserialized.get("emotion")

        # Apply per-pathway audio source overrides when any override is set.
        # Strategy: compute full speaker conditioning from spk_audio_prompt as base,
        # then selectively replace pathways from their override sources.
        any_override = any([mel_audio_path, style_audio_path, spk_cond_audio_path])
        if any_override and request.cache is None:
            # Compute base conditioning from spk_audio_prompt
            base_cond = TTS_MODEL.compute_speaker_conditioning(spk_audio_path)

            # Cache override conditioning to avoid redundant model calls for same path
            _override_cache: dict = {}

            def _get_override_cond(path: str) -> dict:
                if path not in _override_cache:
                    _override_cache[path] = TTS_MODEL.compute_speaker_conditioning(path)
                return _override_cache[path]

            # Override mel spectrogram if mel_audio_source is provided
            if mel_audio_path is not None:
                base_cond["mel"] = _get_override_cond(mel_audio_path)["mel"]

            # Override CAM++ style vector if style_audio_source is provided
            if style_audio_path is not None:
                base_cond["s2mel_style"] = _get_override_cond(style_audio_path)[
                    "s2mel_style"
                ]

            # Override spk_cond + s2mel_prompt if spk_cond_audio_source is provided
            # (these two are always kept together due to architectural coupling)
            if spk_cond_audio_path is not None:
                oc = _get_override_cond(spk_cond_audio_path)
                base_cond["spk_cond"] = oc["spk_cond"]
                base_cond["s2mel_prompt"] = oc["s2mel_prompt"]

            infer_kwargs["speaker_cache"] = base_cond
            # spk_audio_prompt is kept in infer_kwargs but will be ignored because
            # speaker_cache takes precedence inside synthesize_from_tokens
        elif any_override and request.cache is not None:
            logger.warning(
                "Per-pathway audio source overrides are ignored when cache is provided. "
                "The cache takes full precedence."
            )

        infer_kwargs["return_cache"] = request.return_cache

        logger.info(
            f"Synthesizing from {len(codes_np)} tokens, text='{request.text[:50]}...'"
        )

        with INFERENCE_LOCK:
            if request.seed is not None:
                _set_seed(request.seed)
            result = TTS_MODEL.synthesize_from_tokens(**infer_kwargs)
        if result is None:
            raise HTTPException(status_code=500, detail="Inference returned no result")

        # Unpack result
        cache_pydantic = None
        if request.return_cache:
            audio_result, cache_out = result
            cache_pydantic = _serialize_cache(cache_out)
            sample_rate, wav_data = audio_result
        else:
            sample_rate, wav_data = result

        if wav_data.ndim == 2:
            wav_data = wav_data.squeeze()

        if wav_data.dtype == np.int16:
            wav_data_float = wav_data.astype(np.float32) / 32767.0
        else:
            wav_data_float = wav_data.astype(np.float32)

        duration_seconds = len(wav_data) / sample_rate
        logger.info(
            f"Token synthesis complete: {duration_seconds:.2f}s audio generated"
        )

        if request.output_format == "wav":
            wav_tensor = torch.from_numpy(wav_data).unsqueeze(0)
            if wav_tensor.dtype == torch.int16:
                wav_tensor = wav_tensor.float() / 32767.0
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav_tensor, sample_rate, format="wav")
            buffer.seek(0)
            wav_bytes = buffer.read()

            headers = {
                "Content-Disposition": "attachment; filename=synthesized_from_tokens.wav",
                "X-Sample-Rate": str(sample_rate),
                "X-Duration-Seconds": f"{duration_seconds:.3f}",
            }

            if cache_pydantic is not None:
                boundary = "----IndexTTSCacheBoundary"
                parts = []
                parts.append(
                    f"--{boundary}\r\n"
                    f"Content-Type: audio/wav\r\n"
                    f"Content-Disposition: attachment; filename=synthesized_from_tokens.wav\r\n"
                    f"\r\n".encode()
                )
                parts.append(wav_bytes)
                parts.append(
                    f"\r\n--{boundary}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Disposition: inline; name=cache\r\n"
                    f"\r\n".encode()
                )
                parts.append(cache_pydantic.model_dump_json().encode())
                parts.append(f"\r\n--{boundary}--\r\n".encode())

                return Response(
                    content=b"".join(parts),
                    media_type=f"multipart/mixed; boundary={boundary}",
                    headers=headers,
                )

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers=headers,
            )
        elif request.output_format == "numpy":
            return SynthesisResponse(
                status="success",
                output_format="numpy",
                tensor=encode_numpy(wav_data_float),
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                message=f"Generated waveform shape: {list(wav_data_float.shape)}",
                cache=cache_pydantic,
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Token synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Token synthesis failed: {e}")
    finally:
        cleanup_temp_files(temp_files)


def _get_emotion_mode(request: SynthesisRequest) -> str:
    """Get string description of emotion mode for logging."""
    if request.use_emotion_tags:
        return "emotion_tags"
    elif request.emo_audio_prompt is not None:
        return "audio"
    elif request.emo_text is not None:
        return (
            f"text('{request.emo_text[:20]}...')"
            if len(request.emo_text) > 20
            else f"text('{request.emo_text}')"
        )
    elif request.emo_vector is not None:
        return f"vector({request.emo_vector})"
    else:
        return "none"


# -----------------------------------------
# Main Entry Point
# -----------------------------------------


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS Synthesis Service")

    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port to bind to (default: 8002)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    # Model settings - defaults use CHECKPOINTS_DIR environment variable
    parser.add_argument(
        "--cfg-path",
        type=str,
        default=os.path.join(_CHECKPOINTS_DIR, "config.yaml") if _CHECKPOINTS_DIR else None,
        required=_CHECKPOINTS_DIR is None,
        help=f"Path to config file (default: {_CHECKPOINTS_DIR}/config.yaml)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=_CHECKPOINTS_DIR,
        required=_CHECKPOINTS_DIR is None,
        help=f"Path to model directory (default: {_CHECKPOINTS_DIR})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g., 'cuda:0', 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 inference (default: False)",
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        default=False,
        help="Use DeepSpeed optimization (default: False)",
    )
    parser.add_argument(
        "--accel",
        action="store_true",
        default=False,
        help="Use AccelInferenceEngine with FlashAttention + CUDA graphs for GPT2 (default: False). Requires flash_attn.",
    )
    parser.add_argument(
        "--cuda-kernel",
        action="store_true",
        default=False,
        help="Use custom CUDA kernel for BigVGAN (default: False)",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Use torch.compile optimization (default: False)",
    )

    args = parser.parse_args()

    # Setup temp directory
    global TEMP_DIR
    TEMP_DIR = tempfile.mkdtemp(prefix="indextts_service_")
    logger.info(f"Temp directory: {TEMP_DIR}")
    logger.info(f"Local rank: {os.environ.get('LOCAL_RANK', 'N/A')}")

    # Load model before starting server
    load_model(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        device=args.device,
        use_fp16=args.fp16,
        use_deepspeed=args.deepspeed,
        use_accel=args.accel,
        use_cuda_kernel=args.cuda_kernel,
        use_torch_compile=args.torch_compile,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
