"""
F5-TTS Synthesis Service: Direct model.sample() inference endpoint.

A lightweight service that wraps model_obj.sample() for both TTS and editing use cases.
Supports tensor transmission via Base64-encoded numpy arrays.

Usage:
    python -m f5_service.synthesis_service --port 8001

    # Server runs on http://localhost:8001
    # API docs at http://localhost:8001/docs

Example client usage:
    import requests
    import numpy as np
    import base64

    # Encode tensor
    def encode_tensor(arr: np.ndarray) -> dict:
        return {
            "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    # Make request
    response = requests.post("http://localhost:8001/sample", json={
        "cond": encode_tensor(mel_spectrogram),
        "text": ["你好世界。"],
        "duration": 1000,
        "output_format": "wav",
    })
"""

import base64
import io
import logging
import math
import threading
import time
from importlib.resources import files
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import uvicorn
from cached_path import cached_path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from hydra.utils import get_class
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from f5_service.crossfade_utils import (
    apply_masked_edit_splice,
    expand_false_regions,
    expand_spans_sec,
    spans_sec_from_edit_mask,
)
from f5_service.core import (
    cfg_strength as default_cfg_strength,
    device as default_device,
    load_model,
    load_vocoder,
    mel_spec_type as default_mel_spec_type,
    nfe_step as default_nfe_step,
    sway_sampling_coef as default_sway_sampling_coef,
    target_sample_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------
# Global State
# -----------------------------------------

MODEL_OBJ: Optional[Any] = None
VOCODER: Optional[Any] = None
VOCODER_NAME: str = default_mel_spec_type
DEVICE: str = default_device
MAX_F5_EDIT_SECONDS = 30.0
F5_INFERENCE_LOCK = threading.Lock()


# -----------------------------------------
# Pydantic Models for Tensor Serialization
# -----------------------------------------


class EncodedTensor(BaseModel):
    """Base64-encoded numpy array with metadata."""

    data: str = Field(..., description="Base64-encoded tensor bytes")
    shape: List[int] = Field(..., description="Tensor shape, e.g., [1, 1000, 100]")
    dtype: str = Field("float32", description="Numpy dtype string, e.g., 'float32', 'bool'")


class SampleRequest(BaseModel):
    """
    Request for model.sample() inference.

    ## TTS vs Editing Mode

    The mode is determined by whether edit_mask or edit_spans_sec is provided:

    **TTS Mode** (no edit_mask/edit_spans_sec):
    - cond serves as speaker/style prompt
    - Model generates new speech matching reference style
    - duration can be longer than cond length
    - text is the concatenated string or list of the prompt and the generation text

    **Editing Mode** (edit_mask or edit_spans_sec provided):
    - cond should contain the full audio to edit
    - edit_mask[i]=True means keep frame i, False means regenerate
    - edit_spans_sec converts time ranges to edit_mask internally
    - Only unmasked frames are regenerated; masked frames are preserved exactly
    - text is the full text corresponding to the entire cond audio (including both masked and unmasked regions)

    Frame/Time Conversion (F5-TTS config):
        - Sample rate: 24000 Hz, Hop length: 256 samples
        - start_frame = floor(time_sec * 24000 / 256)
        - end_frame = ceil(time_sec * 24000 / 256)
    """

    # Required inputs
    cond: EncodedTensor = Field(
        ...,
        description="Conditioning tensor: either raw audio [b, nw] or mel spectrogram [b, n, d]. "
        "If 2D, model internally converts to mel.",
    )
    text: List[Union[str, List[str]]] = Field(
        ...,
        description="List of text inputs. Each item can be a raw string or a pre-tokenized list of pinyin tokens.",
    )
    duration: Optional[int] = Field(
        None,
        description=(
            "Target duration in internal mel frames. Optional for raw-waveform "
            "edit_spans_sec requests, where the service derives it from cond."
        ),
    )
    duration_sec: Optional[float] = Field(
        None,
        description=(
            "Target duration in seconds. Converted to internal mel frames inside "
            "the service when duration is omitted."
        ),
    )
    input_sample_rate: int = Field(
        target_sample_rate,
        description="Sample rate of raw waveform cond. Ignored for mel cond.",
    )

    # Editing inputs (mutually exclusive)
    edit_mask: Optional[EncodedTensor] = Field(
        None,
        description="Edit mask tensor [b, n] of bool type. True = keep original, False = generate. "
        "Mutually exclusive with edit_spans_sec.",
    )
    edit_spans_sec: Optional[List[List[float]]] = Field(
        None,
        description="List of [start_sec, end_sec] time spans to regenerate. "
        "Requires cond to be raw waveform [b, nw]. "
        "Example: [[1.5, 2.5], [3.0, 4.0]] regenerates those time regions. "
        "Mutually exclusive with edit_mask.",
    )

    # Waveform preservation controls
    preserve_unmasked_waveform: bool = Field(
        False,
        description="If True, splice generated regions into a preserve waveform and return that waveform length/sample rate.",
    )
    preserve_waveform: Optional[EncodedTensor] = Field(
        None,
        description="Optional waveform used only for final waveform splice. Never used as model conditioning.",
    )
    preserve_waveform_sample_rate: int = Field(
        target_sample_rate,
        description="Sample rate of preserve_waveform. If preserve_waveform is omitted, input_sample_rate is used.",
    )
    preserve_spans_sec: Optional[List[List[float]]] = Field(
        None,
        description="Core regenerate spans on the preserve waveform timeline. Defaults to edit spans/mask false-runs.",
    )
    source_crossfade_sec: float = Field(
        0.0,
        description="Crossfade duration outside core regenerate spans, in preserved/source regions.",
    )
    regen_crossfade_sec: float = Field(
        0.0,
        description="Crossfade duration inside core regenerate spans.",
    )

    # Inference parameters (with defaults)
    steps: int = Field(default_nfe_step, description="Number of denoising steps (NFE)")
    cfg_strength: float = Field(default_cfg_strength, description="Classifier-free guidance strength")
    sway_sampling_coef: float = Field(default_sway_sampling_coef, description="Sway sampling coefficient")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    # Output format
    output_format: str = Field(
        "mel",
        description="Output format: 'mel' (mel spectrogram), 'numpy' (waveform array), 'wav' (audio bytes), 'both' (mel + numpy)",
    )

    # Tracing
    trace_id: Optional[str] = Field(
        None,
        description="Caller-supplied trace ID for request tracing and profiling.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "cond": {
                    "data": "<base64-encoded-bytes>",
                    "shape": [1, 500, 100],
                    "dtype": "float32",
                },
                "text": ["Some text to synthesize."],
                "duration": 1000,
                "steps": 32,
                "cfg_strength": 2.0,
                "sway_sampling_coef": -1.0,
                "seed": 42,
                "output_format": "wav",
            }
        }


class SampleBatchRequest(BaseModel):
    """Request for batch inference (multiple utterances).

    Future extension for dynamic batching support.
    """

    requests: List[SampleRequest] = Field(..., description="List of sample requests to process")


class TensorResponse(BaseModel):
    """Response containing an encoded tensor."""

    tensor: EncodedTensor
    sample_rate: Optional[int] = Field(None, description="Sample rate if output is audio")


class SampleResponse(BaseModel):
    """Response for sample request."""

    status: str = Field("success", description="Request status")
    output_format: str = Field(..., description="Output format returned")

    # For mel/numpy output
    tensor: Optional[EncodedTensor] = Field(None, description="Output tensor (for mel/numpy formats)")
    sample_rate: Optional[int] = Field(None, description="Sample rate (for numpy/wav formats)")

    # For 'both' format - return mel and waveform together
    mel_tensor: Optional[EncodedTensor] = Field(None, description="Mel spectrogram tensor (for 'both' format)")
    waveform_tensor: Optional[EncodedTensor] = Field(None, description="Waveform tensor (for 'both' format)")

    # Timing and tracing
    response_time: Optional[float] = Field(None, description="Server-side processing time in seconds")
    trace_id: Optional[str] = Field(None, description="Echo of caller-supplied trace ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Diagnostic metadata for edit requests")

    # For wav output - not included in JSON, returned as binary
    message: Optional[str] = Field(None, description="Additional message")


class SampleBatchResponse(BaseModel):
    """Response for batch inference."""

    status: str
    results: List[SampleResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    vocoder_loaded: bool
    vocoder_name: str
    device: str


# -----------------------------------------
# Tensor Encoding/Decoding Utilities
# -----------------------------------------


def decode_tensor(encoded: EncodedTensor, device: str = "cpu") -> torch.Tensor:
    """Decode Base64-encoded tensor to PyTorch tensor."""
    try:
        # Decode base64 to bytes
        tensor_bytes = base64.b64decode(encoded.data)

        # Convert to numpy array
        np_array = np.frombuffer(tensor_bytes, dtype=np.dtype(encoded.dtype))
        np_array = np_array.reshape(encoded.shape)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(np_array.copy()).to(device)
        return tensor
    except Exception as e:
        raise ValueError(f"Failed to decode tensor: {e}")


def decode_numpy(encoded: EncodedTensor) -> np.ndarray:
    """Decode Base64-encoded tensor to numpy."""
    try:
        tensor_bytes = base64.b64decode(encoded.data)
        np_array = np.frombuffer(tensor_bytes, dtype=np.dtype(encoded.dtype))
        return np_array.reshape(encoded.shape).copy()
    except Exception as e:
        raise ValueError(f"Failed to decode tensor: {e}")


def encode_tensor(tensor: torch.Tensor) -> EncodedTensor:
    """Encode PyTorch tensor to Base64-encoded format."""
    # Convert to numpy
    if tensor.is_cuda:
        np_array = tensor.cpu().numpy()
    else:
        np_array = tensor.numpy()

    # Encode to base64
    tensor_bytes = np_array.tobytes()
    data_b64 = base64.b64encode(tensor_bytes).decode("utf-8")

    return EncodedTensor(
        data=data_b64,
        shape=list(np_array.shape),
        dtype=str(np_array.dtype),
    )


def encode_numpy(arr: np.ndarray) -> EncodedTensor:
    """Encode numpy array to Base64-encoded format."""
    tensor_bytes = arr.tobytes()
    data_b64 = base64.b64encode(tensor_bytes).decode("utf-8")

    return EncodedTensor(
        data=data_b64,
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def resample_waveform_np(waveform: np.ndarray, orig_sr: int, target_sr_val: int) -> np.ndarray:
    """Resample mono or batched waveform with soxr."""
    if orig_sr == target_sr_val:
        return waveform.astype(np.float32, copy=False)

    import soxr

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        return soxr.resample(arr, orig_sr, target_sr_val).astype(np.float32)
    if arr.ndim == 2:
        return np.stack(
            [soxr.resample(row, orig_sr, target_sr_val).astype(np.float32) for row in arr],
            axis=0,
        )
    raise ValueError(f"Expected mono or batched waveform for resampling, got shape {list(arr.shape)}")


def waveform_to_mono_np(waveform: np.ndarray) -> np.ndarray:
    """Normalize [n] or [1,n] waveform to mono [n]."""
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"preserve waveform splice supports mono batch size 1 only, got shape {list(arr.shape)}")


# -----------------------------------------
# Edit Mask Generation
# -----------------------------------------

# F5-TTS audio parameters
HOP_LENGTH = 256
SAMPLE_RATE = target_sample_rate  # 24000


def time_to_start_frame(time_sec: float) -> int:
    """Convert a start time in seconds to an inclusive frame index."""
    return int(math.floor(time_sec * SAMPLE_RATE / HOP_LENGTH))


def time_to_end_frame(time_sec: float) -> int:
    """Convert an end time in seconds to an exclusive frame index."""
    return int(math.ceil(time_sec * SAMPLE_RATE / HOP_LENGTH))


def frame_to_time(frame_idx: int) -> float:
    """Convert frame index to time in seconds."""
    return frame_idx * HOP_LENGTH / SAMPLE_RATE


def create_edit_mask_from_spans(
    total_frames: int,
    edit_spans_sec: List[List[float]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create edit mask from time spans.

    Args:
        total_frames: Total number of mel frames
        edit_spans_sec: List of [start_sec, end_sec] time spans to regenerate
        device: Target device for the mask tensor

    Returns:
        Boolean mask tensor [1, total_frames]. True = keep original, False = regenerate.
    """
    # Start with all True (keep everything)
    mask = torch.ones(1, total_frames, dtype=torch.bool, device=device)

    for span in edit_spans_sec:
        if len(span) != 2:
            raise ValueError(f"Each edit span must be [start_sec, end_sec], got {span}")

        start_sec, end_sec = span

        # Convert to [start, end) frames. The end uses ceil so a non-aligned
        # span fully covers its right boundary.
        start_frame = time_to_start_frame(start_sec)
        end_frame = time_to_end_frame(end_sec)

        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        # Mark region for regeneration (False)
        if start_frame < end_frame:
            mask[:, start_frame:end_frame] = False

    return mask


def waveform_to_mel_frames(num_samples: int) -> int:
    """Calculate number of mel frames from waveform samples."""
    # This matches the mel_spec calculation in F5-TTS
    # For vocos: n_frames = (num_samples // hop_length) + 1 approximately
    return (num_samples // HOP_LENGTH) + 1


def vocode_mel_to_numpy(generated: torch.Tensor) -> np.ndarray:
    """Decode [b,n,d] mel to a mono numpy waveform."""
    if VOCODER is None:
        raise HTTPException(status_code=503, detail="Vocoder not loaded")

    gen_mel = generated.permute(0, 2, 1)  # [b, d, n]
    if VOCODER_NAME == "vocos":
        waveform = VOCODER.decode(gen_mel)
    elif VOCODER_NAME == "bigvgan":
        waveform = VOCODER(gen_mel)
    else:
        raise ValueError(f"Unsupported vocoder: {VOCODER_NAME}")

    return waveform.squeeze().detach().cpu().numpy().astype(np.float32)


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    title="F5-TTS Synthesis Service",
    description="Direct model.sample() inference endpoint for TTS and editing",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Load model and vocoder on startup."""
    global MODEL_OBJ, VOCODER, VOCODER_NAME, DEVICE

    logger.info("Starting F5-TTS Synthesis Service...")

    model_name = "F5TTS_v1_Base"
    vocoder_name = default_mel_spec_type
    device_val = default_device

    # Load model config
    model_cfg_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    # Get checkpoint
    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))

    logger.info(f"Loading {model_name} on {device_val}...")
    MODEL_OBJ = load_model(
        model_cls,
        model_arc,
        ckpt_file,
        mel_spec_type=vocoder_name,
        vocab_file="",
        device=device_val,
    )

    logger.info(f"Loading vocoder ({vocoder_name})...")
    VOCODER = load_vocoder(vocoder_name=vocoder_name, device=device_val)

    VOCODER_NAME = vocoder_name
    DEVICE = device_val

    logger.info("✓ Model and vocoder loaded successfully")
    logger.info("Service ready")


# -----------------------------------------
# Endpoints
# -----------------------------------------


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "F5-TTS Synthesis Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL_OBJ is not None else "not_ready",
        model_loaded=MODEL_OBJ is not None,
        vocoder_loaded=VOCODER is not None,
        vocoder_name=VOCODER_NAME,
        device=DEVICE,
    )


@app.post("/sample")
async def sample(request: SampleRequest):
    """
    Run model.sample() inference.

    Supports both TTS (cond=audio) and editing (cond=mel + edit_mask) use cases.

    **Editing with edit_spans_sec**:
    When edit_spans_sec is provided with a raw waveform cond [b, nw], the service:
    1. Calculates the number of mel frames from waveform length
    2. Creates an edit_mask from the time spans
    3. The model auto-converts waveform to mel internally
    4. Regenerates only the specified regions

    **Input validation**:
    - edit_mask and edit_spans_sec are mutually exclusive
    - edit_spans_sec requires cond to be 2D (raw waveform)

    Returns:
        - If output_format='mel': JSON with encoded mel spectrogram
        - If output_format='numpy': JSON with encoded waveform array
        - If output_format='wav': Binary WAV file response
        - If output_format='both': JSON with mel and waveform
    """
    _t0 = time.time()

    try:
        # Validate mutual exclusivity
        if request.edit_mask is not None and request.edit_spans_sec is not None:
            raise ValueError("edit_mask and edit_spans_sec are mutually exclusive. Provide only one.")
        if request.output_format not in {"mel", "numpy", "wav", "both"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid output_format: {request.output_format}. Must be 'mel', 'numpy', 'wav', or 'both'.",
            )
        if request.input_sample_rate <= 0 or request.preserve_waveform_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        if request.source_crossfade_sec < 0 or request.regen_crossfade_sec < 0:
            raise ValueError("crossfade durations must be non-negative")

        preserve_mode = request.preserve_unmasked_waveform
        editing_mode = request.edit_mask is not None or request.edit_spans_sec is not None
        if preserve_mode:
            if not editing_mode:
                raise ValueError("preserve_unmasked_waveform requires edit_mask or edit_spans_sec")
            if request.output_format == "mel":
                raise ValueError("preserve_unmasked_waveform is only supported for waveform outputs")

        # Decode conditioning tensor
        cond_np = decode_numpy(request.cond)
        cond_is_waveform = cond_np.ndim == 2
        if cond_is_waveform and request.input_sample_rate != SAMPLE_RATE:
            cond_model_np = resample_waveform_np(cond_np, request.input_sample_rate, SAMPLE_RATE)
        else:
            cond_model_np = cond_np.astype(np.float32, copy=False)
        cond = torch.from_numpy(cond_model_np.copy()).to(DEVICE)
        logger.debug(f"Decoded cond tensor: shape={cond.shape}, dtype={cond.dtype}")

        if cond.ndim == 2:
            cond_len_frames = waveform_to_mel_frames(cond.shape[1])
        elif cond.ndim == 3:
            cond_len_frames = cond.shape[1]
        else:
            raise ValueError(f"cond must be raw audio [b,nw] or mel [b,n,d], got shape {list(cond.shape)}")

        if request.duration is not None:
            duration_frames = int(request.duration)
        elif request.duration_sec is not None:
            duration_frames = waveform_to_mel_frames(
                int(round(float(request.duration_sec) * SAMPLE_RATE))
            )
        elif request.edit_spans_sec is not None and cond.ndim == 2:
            duration_frames = cond_len_frames
        else:
            raise ValueError(
                "duration is required unless raw-waveform edit_spans_sec is provided"
            )
        if duration_frames <= 0:
            raise ValueError("duration must be positive")

        requested_duration_sec = duration_frames * HOP_LENGTH / SAMPLE_RATE
        if editing_mode and requested_duration_sec > MAX_F5_EDIT_SECONDS:
            raise ValueError(
                "F5-TTS edit request exceeds the single-call duration limit: "
                f"{requested_duration_sec:.2f}s > {MAX_F5_EDIT_SECONDS:.2f}s "
                f"({duration_frames} frames). Reduce the input audio length."
            )

        # Keep cheap request validation ahead of model readiness so malformed
        # edit requests consistently return 400 even during model startup.
        if MODEL_OBJ is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        preserve_waveform = None
        preserve_sample_rate = target_sample_rate
        if preserve_mode:
            if request.preserve_waveform is not None:
                preserve_waveform = waveform_to_mono_np(decode_numpy(request.preserve_waveform))
                preserve_sample_rate = request.preserve_waveform_sample_rate
            elif cond_is_waveform:
                preserve_waveform = waveform_to_mono_np(cond_np)
                preserve_sample_rate = request.input_sample_rate
            else:
                raise ValueError("preserve_waveform is required when cond is mel")

            if cond.shape[0] != 1:
                raise ValueError("preserve waveform splice supports batch size 1 only")
            if duration_frames < cond_len_frames:
                raise ValueError(
                    f"duration ({duration_frames}) must be >= conditioning length ({cond_len_frames}) "
                    "when preserve_unmasked_waveform=True"
                )

        edit_metadata = None

        # Handle edit_spans_sec
        edit_mask = None
        core_spans_sec = request.preserve_spans_sec
        if request.edit_spans_sec is not None:
            if cond.ndim != 2:
                raise ValueError(
                    f"edit_spans_sec requires cond to be 2D raw waveform [b, nw], got shape {list(cond.shape)}"
                )

            # Calculate number of mel frames from waveform
            num_frames = cond_len_frames

            # Ensure duration matches
            if duration_frames != num_frames:
                logger.warning(
                    f"duration ({duration_frames}) != calculated frames ({num_frames}). "
                    f"Using calculated frames for mask generation."
                )

            edit_spans_for_mask = request.edit_spans_sec
            if preserve_mode and request.source_crossfade_sec > 0:
                edit_spans_for_mask = expand_spans_sec(
                    request.edit_spans_sec,
                    request.source_crossfade_sec,
                    num_frames * HOP_LENGTH / SAMPLE_RATE,
                )
            if core_spans_sec is None:
                core_spans_sec = request.edit_spans_sec

            # Create edit mask from time spans
            edit_mask = create_edit_mask_from_spans(
                total_frames=num_frames,
                edit_spans_sec=edit_spans_for_mask,
                device=DEVICE,
            )

            num_edit_frames = (~edit_mask).sum().item()
            logger.info(
                f"Created edit mask: {num_frames} frames, "
                f"{num_edit_frames} frames to regenerate ({num_edit_frames / num_frames * 100:.1f}%)"
            )

        # Decode edit_mask if provided directly
        elif request.edit_mask is not None:
            edit_mask_np = decode_numpy(request.edit_mask).astype(bool)
            if preserve_mode:
                if edit_mask_np.shape[-1] > duration_frames:
                    raise ValueError(
                        f"edit_mask length ({edit_mask_np.shape[-1]}) must be <= duration ({duration_frames}) "
                        "when preserve_unmasked_waveform=True"
                    )
                if edit_mask_np.shape[-1] != cond_len_frames:
                    raise ValueError(
                        f"edit_mask length ({edit_mask_np.shape[-1]}) must equal conditioning length "
                        f"({cond_len_frames}) when preserve_unmasked_waveform=True"
                    )
                if core_spans_sec is None:
                    core_spans_sec = spans_sec_from_edit_mask(edit_mask_np, HOP_LENGTH, SAMPLE_RATE)
                expand_frames = int(np.ceil(request.source_crossfade_sec * SAMPLE_RATE / HOP_LENGTH))
                edit_mask_np = expand_false_regions(edit_mask_np, expand_frames)
            edit_mask = torch.from_numpy(edit_mask_np.copy()).to(DEVICE)
            logger.debug(f"Decoded edit_mask: shape={edit_mask.shape}, dtype={edit_mask.dtype}")

        if editing_mode and edit_mask is not None:
            edit_metadata = {
                "internal_sample_rate": SAMPLE_RATE,
                "hop_length": HOP_LENGTH,
                "mask_spans_sec": spans_sec_from_edit_mask(
                    edit_mask.detach().cpu().numpy(),
                    HOP_LENGTH,
                    SAMPLE_RATE,
                ),
                "core_spans_sec": core_spans_sec or [],
                "source_crossfade_sec": request.source_crossfade_sec,
                "regen_crossfade_sec": request.regen_crossfade_sec,
                "input_sample_rate": request.input_sample_rate,
                "duration_frames": duration_frames,
                "duration_source": (
                    "duration"
                    if request.duration is not None
                    else "duration_sec"
                    if request.duration_sec is not None
                    else "cond_waveform"
                ),
            }

        # Auto-convert text: if any item is a raw string (not pre-tokenized list),
        # apply convert_char_to_pinyin for Chinese character support
        text_input = request.text
        if any(isinstance(t, str) for t in text_input):
            from f5_tts.model.utils import convert_char_to_pinyin

            text_input = convert_char_to_pinyin([t if isinstance(t, str) else "".join(t) for t in text_input])
            logger.debug(f"Auto-converted text to pinyin: {text_input}")

        with F5_INFERENCE_LOCK:
            # Run inference and vocoder work under one service-process lock.
            with torch.inference_mode():
                generated, _ = MODEL_OBJ.sample(
                    cond=cond,
                    text=text_input,
                    duration=duration_frames,
                    steps=request.steps,
                    cfg_strength=request.cfg_strength,
                    sway_sampling_coef=request.sway_sampling_coef,
                    seed=request.seed,
                    edit_mask=edit_mask,
                    # Upstream clamps its automatic extension to max_duration.
                    **({"max_duration": duration_frames} if preserve_mode else {}),
                )

                generated = generated.to(torch.float32)  # [b, n, d] mel spectrogram

            _elapsed = time.time() - _t0

            def build_waveform_output() -> tuple[np.ndarray, int]:
                waveform_np = vocode_mel_to_numpy(generated)
                output_sample_rate = target_sample_rate
                if preserve_mode:
                    assert preserve_waveform is not None
                    assert core_spans_sec is not None
                    if preserve_sample_rate != target_sample_rate:
                        waveform_np = resample_waveform_np(waveform_np, target_sample_rate, preserve_sample_rate)
                    waveform_np = apply_masked_edit_splice(
                        source_waveform=preserve_waveform,
                        generated_waveform=waveform_np,
                        spans_sec=core_spans_sec,
                        sample_rate=preserve_sample_rate,
                        source_crossfade_sec=request.source_crossfade_sec,
                        regen_crossfade_sec=request.regen_crossfade_sec,
                    )
                    output_sample_rate = preserve_sample_rate
                if edit_metadata is not None:
                    edit_metadata["output_sample_rate"] = output_sample_rate
                return waveform_np.astype(np.float32, copy=False), output_sample_rate

            # Return based on output format
            if request.output_format == "mel":
                # Return mel spectrogram
                return SampleResponse(
                    status="success",
                    output_format="mel",
                    tensor=encode_tensor(generated),
                    response_time=round(_elapsed, 3),
                    trace_id=request.trace_id,
                    metadata=edit_metadata,
                    message=f"Generated mel spectrogram shape: {list(generated.shape)}",
                )

            elif request.output_format == "numpy":
                waveform_np, output_sample_rate = build_waveform_output()

                _elapsed = time.time() - _t0
                return SampleResponse(
                    status="success",
                    output_format="numpy",
                    tensor=encode_numpy(waveform_np),
                    sample_rate=output_sample_rate,
                    response_time=round(_elapsed, 3),
                    trace_id=request.trace_id,
                    metadata=edit_metadata,
                    message=f"Generated waveform shape: {list(waveform_np.shape)}",
                )

            elif request.output_format == "wav":
                waveform_np, output_sample_rate = build_waveform_output()

                # Write to bytes buffer
                buffer = io.BytesIO()
                sf.write(buffer, waveform_np, output_sample_rate, format="WAV")
                buffer.seek(0)
                wav_bytes = buffer.read()

                _elapsed = time.time() - _t0
                headers = {
                    "Content-Disposition": "attachment; filename=generated.wav",
                    "X-Sample-Rate": str(output_sample_rate),
                    "X-Duration-Samples": str(len(waveform_np)),
                    "X-Response-Time": f"{_elapsed:.3f}",
                    "X-F5-Internal-Sample-Rate": str(SAMPLE_RATE),
                    "X-F5-Hop-Length": str(HOP_LENGTH),
                }
                if request.trace_id:
                    headers["X-Trace-Id"] = request.trace_id

                return Response(
                    content=wav_bytes,
                    media_type="audio/wav",
                    headers=headers,
                )

            elif request.output_format == "both":
                waveform_np, output_sample_rate = build_waveform_output()

                _elapsed = time.time() - _t0
                return SampleResponse(
                    status="success",
                    output_format="both",
                    mel_tensor=encode_tensor(generated),
                    waveform_tensor=encode_numpy(waveform_np),
                    sample_rate=output_sample_rate,
                    response_time=round(_elapsed, 3),
                    trace_id=request.trace_id,
                    metadata=edit_metadata,
                    message=f"Generated mel shape: {list(generated.shape)}, waveform shape: {list(waveform_np.shape)}",
                )

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid output_format: {request.output_format}. Must be 'mel', 'numpy', 'wav', or 'both'.",
                )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/sample_batch", response_model=SampleBatchResponse)
async def sample_batch(request: SampleBatchRequest):
    """
    Batch inference endpoint (placeholder for future dynamic batching).

    Currently processes requests sequentially. Future versions may implement
    true batching for improved GPU utilization.
    """
    if MODEL_OBJ is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if any(item.output_format == "wav" for item in request.requests):
        raise HTTPException(400, "Batch requests require a JSON output format: mel, numpy, or both")
    results = []
    for single_request in request.requests:
        result = await sample(single_request)
        if isinstance(result, Response):
            raise HTTPException(500, "Expected a JSON batch result")
        results.append(result)

    return SampleBatchResponse(
        status="completed",
        results=results,
    )


# -----------------------------------------
# Main Entry Point
# -----------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F5-TTS Synthesis Service")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to (default: 8001)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers (default: 1, each loads its own model copy)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Note: When using workers > 1, we must use the string import path
    uvicorn.run(
        "f5_service.synthesis_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
