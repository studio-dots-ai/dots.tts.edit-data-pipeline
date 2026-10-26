"""
SeedVC Voice Conversion Service: FastAPI endpoint for voice conversion.

A lightweight service wrapping SeedVC for voice conversion tasks.
Converts source audio to match the speaker characteristics of a reference audio.

Usage:
    source envs/pipe-gpu/.venv/bin/activate
    python seed_vc_service.py --port 8003

    # Server runs on http://localhost:8003
    # API docs at http://localhost:8003/docs

Example client usage:
    import requests
    import base64

    # Encode audio file to base64
    def encode_audio_file(path: str) -> dict:
        with open(path, "rb") as f:
            audio_bytes = f.read()
        return {
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": "wav",
        }

    # Make request
    response = requests.post("http://localhost:8003/convert", json={
        "source_audio": encode_audio_file("source.wav"),
        "reference_audio": encode_audio_file("reference.wav"),
        "output_format": "wav",
    })

    # Save response
    with open("output.wav", "wb") as f:
        f.write(response.content)
"""

import base64
import io
import logging
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sys
_TOOL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOL_ROOT.parents[1]))
from tools._upstream import activate
UPSTREAM_ROOT = activate(_TOOL_ROOT, ("modules", "hf_utils"))

import numpy as np
import torch
import torchaudio
import librosa
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

# Honor the configured model cache; otherwise retain Hugging Face defaults.
if os.environ.get("CHECKPOINTS_DIR"):
    os.environ.setdefault("HF_HUB_CACHE", os.environ["CHECKPOINTS_DIR"])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------
# Global State
# -----------------------------------------

MODEL_COMPONENTS: Optional[Dict[str, Any]] = None
MODEL_CONFIG: Dict[str, Any] = {}
TEMP_DIR: Optional[str] = None


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


class ConvertRequest(BaseModel):
    """Request for voice conversion."""

    source_audio: Union[EncodedAudio, str] = Field(
        ...,
        description="Source audio to convert. Either Base64-encoded audio object or file path string.",
    )
    reference_audio: Union[EncodedAudio, str] = Field(
        ...,
        description="Reference audio for target speaker. Either Base64-encoded audio object or file path string.",
    )

    # Conversion parameters
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible inference. Sets torch, numpy, and python random seeds.",
    )
    diffusion_steps: int = Field(
        30, ge=1, le=100, description="Number of diffusion steps"
    )
    length_adjust: float = Field(
        1.0, ge=0.5, le=2.0, description="Length adjustment factor"
    )
    inference_cfg_rate: float = Field(
        0.7, ge=0.0, le=1.0, description="Classifier-free guidance rate"
    )

    # Output format
    output_format: str = Field(
        "wav",
        description="Output format: 'wav' (audio bytes) or 'numpy' (encoded waveform array)",
    )

    @model_validator(mode="after")
    def check_output_format(self):
        if self.output_format not in ("wav", "numpy"):
            raise ValueError(
                f"output_format must be 'wav' or 'numpy', got '{self.output_format}'"
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "source_audio": "/path/to/source.wav",
                "reference_audio": "/path/to/reference.wav",
                "diffusion_steps": 30,
                "output_format": "wav",
            }
        }


class ConvertResponse(BaseModel):
    """Response for conversion request (for numpy output format)."""

    status: str = Field("success", description="Request status")
    output_format: str = Field(..., description="Output format returned")
    tensor: Optional[EncodedTensor] = Field(None, description="Output waveform tensor")
    sample_rate: int = Field(..., description="Audio sample rate")
    duration_seconds: Optional[float] = Field(
        None, description="Audio duration in seconds"
    )
    message: Optional[str] = Field(None, description="Additional message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    config: Dict[str, Any]


# -----------------------------------------
# Audio Utilities
# -----------------------------------------


def decode_audio_input(
    audio_input: Union[EncodedAudio, str],
    temp_dir: str,
    prefix: str = "audio",
) -> str:
    """Decode audio input to a file path."""
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        return audio_input

    try:
        audio_bytes = base64.b64decode(audio_input.data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 audio: {e}")

    file_ext = audio_input.format.lower()
    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"
    temp_path = os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{file_ext}")

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


def cleanup_temp_files(paths: List[str]):
    """Clean up temporary files."""
    for path in paths:
        try:
            if path and os.path.exists(path) and TEMP_DIR and path.startswith(TEMP_DIR):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {path}: {e}")


# -----------------------------------------
# FastAPI App
# -----------------------------------------

app = FastAPI(
    title="SeedVC Voice Conversion Service",
    description="Voice conversion service using SeedVC model.",
    version="1.0.0",
)


def load_models(
    device: Optional[str],
    fp16: bool,
    f0_condition: bool,
    checkpoint: Optional[str],
    config: Optional[str],
):
    """Load SeedVC models."""
    global MODEL_COMPONENTS, MODEL_CONFIG

    # Import here to avoid import errors
    import yaml
    from modules.commons import recursive_munch, build_model, load_checkpoint
    from huggingface_hub import hf_hub_download

    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("CHECKPOINTS_DIR")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    logger.info(f"Loading SeedVC models on {device}...")

    # Load model config
    if not f0_condition:
        if checkpoint is None:
            dit_checkpoint_path = hf_hub_download(
                "Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", cache_dir=cache_dir,
            )
            dit_config_path = hf_hub_download(
                "Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml", cache_dir=cache_dir,
            )
        else:
            dit_checkpoint_path = checkpoint
            dit_config_path = config
        f0_fn = None
    else:
        if checkpoint is None:
            dit_checkpoint_path = hf_hub_download(
                "Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth", cache_dir=cache_dir,
            )
            dit_config_path = hf_hub_download(
                "Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml", cache_dir=cache_dir,
            )
        else:
            dit_checkpoint_path = checkpoint
            dit_config_path = config
        from modules.rmvpe import RMVPE

        model_path = hf_hub_download(
            "lj1995/VoiceConversionWebUI", "rmvpe.pt", cache_dir=cache_dir
        )
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        f0_fn = f0_extractor.infer_from_audio

    config_dict = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config_dict["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config_dict["preprocess_params"]["spect_params"]["hop_length"]
    sr = config_dict["preprocess_params"]["sr"]

    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load CAMPPlus
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = hf_hub_download(
        "funasr/campplus", "campplus_cn_common.bin", cache_dir=cache_dir
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    # Load vocoder
    vocoder_type = model_params.vocoder.type
    if vocoder_type == "bigvgan":
        from modules.bigvgan import bigvgan

        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            bigvgan_name, use_cuda_kernel=False
        )
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == "hifigan":
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor

        hift_config = yaml.safe_load(open(UPSTREAM_ROOT / "configs/hifigan.yml", "r"))
        hift_gen = HiFTGenerator(
            **hift_config["hift"],
            f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
        )
        hift_path = hf_hub_download(
            "FunAudioLLM/CosyVoice-300M", "hift.pt", cache_dir=cache_dir
        )
        hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    # Load speech tokenizer
    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == "whisper":
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(
            whisper_name, torch_dtype=torch.float16
        ).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

    # Mel spectrogram function
    mel_fn_args = {
        "n_fft": config_dict["preprocess_params"]["spect_params"]["n_fft"],
        "win_size": config_dict["preprocess_params"]["spect_params"]["win_length"],
        "hop_size": config_dict["preprocess_params"]["spect_params"]["hop_length"],
        "num_mels": config_dict["preprocess_params"]["spect_params"]["n_mels"],
        "sampling_rate": sr,
        "fmin": config_dict["preprocess_params"]["spect_params"].get("fmin", 0),
        "fmax": None
        if config_dict["preprocess_params"]["spect_params"].get("fmax", "None")
        == "None"
        else 8000,
        "center": False,
    }
    from modules.audio import mel_spectrogram

    mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

    MODEL_COMPONENTS = {
        "model": model,
        "semantic_fn": semantic_fn,
        "f0_fn": f0_fn,
        "vocoder_fn": vocoder_fn,
        "campplus_model": campplus_model,
        "mel_fn": mel_fn,
        "sr": sr,
        "hop_length": hop_length,
        "device": device,
        "fp16": fp16,
        "f0_condition": f0_condition,
    }

    MODEL_CONFIG = {
        "device": str(device),
        "fp16": fp16,
        "f0_condition": f0_condition,
        "sr": sr,
    }

    logger.info("✓ SeedVC models loaded successfully")


@torch.no_grad()
def run_voice_conversion(
    source_path: str,
    reference_path: str,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
) -> tuple:
    """Run voice conversion and return (waveform_numpy, sample_rate)."""

    if MODEL_COMPONENTS is None:
        raise RuntimeError("Model not loaded")

    model = MODEL_COMPONENTS["model"]
    semantic_fn = MODEL_COMPONENTS["semantic_fn"]
    f0_fn = MODEL_COMPONENTS["f0_fn"]
    vocoder_fn = MODEL_COMPONENTS["vocoder_fn"]
    campplus_model = MODEL_COMPONENTS["campplus_model"]
    mel_fn = MODEL_COMPONENTS["mel_fn"]
    sr = MODEL_COMPONENTS["sr"]
    hop_length = MODEL_COMPONENTS["hop_length"]
    device = MODEL_COMPONENTS["device"]
    fp16 = MODEL_COMPONENTS["fp16"]
    f0_condition = MODEL_COMPONENTS["f0_condition"]

    # Load audio
    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(reference_path, sr=sr)[0]

    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    # Process semantic features
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[
                    :, traversed_time : traversed_time + 16000 * 30
                ]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        converted_waves_16k[
                            :,
                            traversed_time : traversed_time
                            + 16000 * (30 - overlapping_time),
                        ],
                    ],
                    dim=-1,
                )
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += (
                30 * 16000
                if traversed_time == 0
                else chunk.size(-1) - 16000 * overlapping_time
            )
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # F0 processing (if enabled)
    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        shifted_log_f0_alt = log_f0_alt.clone()
        shifted_log_f0_alt[F0_alt > 1] = (
            log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    else:
        F0_ori = None
        shifted_f0_alt = None

    # Length regulation
    cond, _, codes, _, _ = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
    )

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []

    def crossfade(chunk1, chunk2, overlap):
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in[: len(chunk2)]
                + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
            )
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16 if fp16 else torch.float32
        ):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]

        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]

        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
            )
            generated_wave_chunks.append(output_wave)
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    vc_wave = np.concatenate(generated_wave_chunks)
    return vc_wave, sr


# -----------------------------------------
# Endpoints
# -----------------------------------------


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "SeedVC Voice Conversion Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if MODEL_COMPONENTS is not None else "not_ready",
        model_loaded=MODEL_COMPONENTS is not None,
        device=MODEL_CONFIG.get("device", "unknown"),
        config=MODEL_CONFIG,
    )


@app.post("/convert")
def convert(request: ConvertRequest):
    """
    Convert source audio to match the speaker of reference audio.

    Returns:
        - If output_format='wav': Binary WAV file response
        - If output_format='numpy': JSON with encoded waveform array
    """
    if MODEL_COMPONENTS is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    temp_files = []

    try:
        # Decode audio inputs
        source_path = decode_audio_input(request.source_audio, TEMP_DIR, prefix="src")
        if isinstance(request.source_audio, EncodedAudio):
            temp_files.append(source_path)

        reference_path = decode_audio_input(
            request.reference_audio, TEMP_DIR, prefix="ref"
        )
        if isinstance(request.reference_audio, EncodedAudio):
            temp_files.append(reference_path)

        logger.info(
            f"Running voice conversion: source={source_path}, reference={reference_path}"
        )

        # Set seed for reproducible inference
        if request.seed is not None:
            _set_seed(request.seed)

        # Run conversion
        waveform, sample_rate = run_voice_conversion(
            source_path=source_path,
            reference_path=reference_path,
            diffusion_steps=request.diffusion_steps,
            length_adjust=request.length_adjust,
            inference_cfg_rate=request.inference_cfg_rate,
        )

        duration_seconds = len(waveform) / sample_rate
        logger.info(f"Conversion complete: {duration_seconds:.2f}s audio generated")

        if request.output_format == "wav":
            wav_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav_tensor, sample_rate, format="wav")
            buffer.seek(0)
            wav_bytes = buffer.read()

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=converted.wav",
                    "X-Sample-Rate": str(sample_rate),
                    "X-Duration-Seconds": f"{duration_seconds:.3f}",
                },
            )

        elif request.output_format == "numpy":
            return ConvertResponse(
                status="success",
                output_format="numpy",
                tensor=encode_numpy(waveform.astype(np.float32)),
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                message=f"Generated waveform shape: {list(waveform.shape)}",
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        cleanup_temp_files(temp_files)


# -----------------------------------------
# Main Entry Point
# -----------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SeedVC Voice Conversion Service")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16")
    parser.add_argument(
        "--f0-condition", action="store_true", default=False, help="Use F0 conditioning"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config")

    args = parser.parse_args()

    global TEMP_DIR
    TEMP_DIR = tempfile.mkdtemp(prefix="seedvc_service_")
    logger.info(f"Temp directory: {TEMP_DIR}")

    load_models(
        device=args.device,
        fp16=args.fp16,
        f0_condition=args.f0_condition,
        checkpoint=args.checkpoint,
        config=args.config,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
