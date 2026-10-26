"""
Qwen3-TTS Core: Model loading and pure inference functions.

Loads CustomVoice (preset speakers + instruct), and Base (voice
cloning) models, plus the 12Hz tokenizer for token encode/decode.

Design note: model loading is isolated in init functions so the generate() calls
can later be swapped for vLLM inference with minimal changes.
"""

from pathlib import Path
import sys
_TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOL_ROOT.parents[1]))
from tools._upstream import activate
UPSTREAM_ROOT = activate(_TOOL_ROOT, ("qwen_tts",))

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Result of a TTS synthesis call."""
    audio: np.ndarray          # float32 waveform
    sample_rate: int
    duration: float            # seconds


@dataclass
class TokenEncodeResult:
    """Result of tokenizer encode."""
    codes: List[np.ndarray]    # list of (T, Q) int64 arrays
    token_rate_hz: float       # tokens per second
    durations: List[float]     # per-item durations in seconds


@dataclass
class TokenDecodeResult:
    """Result of tokenizer decode."""
    audios: List[np.ndarray]   # list of float32 waveforms
    sample_rate: int


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_custom_voice_model = None      # Qwen3TTSModel  (CustomVoice)
_base_model = None              # Qwen3TTSModel  (Base / voice clone)
_tokenizer = None               # Qwen3TTSTokenizer (12Hz)

_model_info: Dict[str, Any] = {}


def get_models():
    """Return (custom_voice_model, base_model, tokenizer)."""
    return _custom_voice_model, _base_model, _tokenizer




def get_model_info() -> Dict[str, Any]:
    return dict(_model_info)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_models(
    custom_voice_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    base_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    tokenizer_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    load_custom_voice: bool = True,
    load_base: bool = True,
    load_tokenizer: bool = True,
):
    """
    Load Qwen3-TTS models.  Each flag can be toggled independently.

    Args:
        custom_voice_path: HF repo id or local path for CustomVoice model.
        base_path: HF repo id or local path for Base (voice clone) model.
        tokenizer_path: HF repo id or local path for 12Hz tokenizer.
        device: torch device string.
        dtype: "bfloat16" or "float16" or "float32".
        load_custom_voice: whether to load the CustomVoice model.
        load_base: whether to load the Base model.
        load_tokenizer: whether to load the 12Hz tokenizer.
    """
    global _custom_voice_model, _base_model, _tokenizer, _model_info

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    load_kwargs = dict(
        device_map=device,
        torch_dtype=torch_dtype,
    )

    # Select the attention implementation before loading the model.
    try:
        import flash_attn  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using flash_attention_2")
    except ImportError:
        logger.info("flash-attn not available, using default attention")

    if load_custom_voice:
        logger.info(f"Loading CustomVoice model from {custom_voice_path} ...")
        from qwen_tts import Qwen3TTSModel
        _custom_voice_model = Qwen3TTSModel.from_pretrained(custom_voice_path, **load_kwargs)
        logger.info("  CustomVoice model loaded.")

    if load_base:
        logger.info(f"Loading Base (voice clone) model from {base_path} ...")
        from qwen_tts import Qwen3TTSModel
        _base_model = Qwen3TTSModel.from_pretrained(base_path, **load_kwargs)
        logger.info("  Base model loaded.")

    if load_tokenizer:
        logger.info(f"Loading 12Hz tokenizer from {tokenizer_path} ...")
        from qwen_tts import Qwen3TTSTokenizer
        _tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_path, device_map=device, torch_dtype=torch_dtype)
        logger.info("  Tokenizer loaded.")

    _model_info = {
        "voice_design_path": None,
        "custom_voice_path": custom_voice_path if load_custom_voice else None,
        "base_path": base_path if load_base else None,
        "tokenizer_path": tokenizer_path if load_tokenizer else None,
        "device": device,
        "dtype": dtype,
        "voice_design_loaded": False,
        "custom_voice_loaded": _custom_voice_model is not None,
        "base_loaded": _base_model is not None,
        "tokenizer_loaded": _tokenizer is not None,
    }

    if _custom_voice_model is not None:
        speakers = _custom_voice_model.get_supported_speakers()
        _model_info["supported_speakers"] = speakers or []
        languages = _custom_voice_model.get_supported_languages()
        _model_info["supported_languages"] = languages or []

    logger.info("Qwen3-TTS init complete.")


# ---------------------------------------------------------------------------
# CustomVoice synthesis  (preset speakers + optional instruct)
# ---------------------------------------------------------------------------

def synthesize_custom_voice(
    text: str,
    speaker: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    **gen_kwargs,
) -> SynthesisResult:
    """
    Synthesize speech with a preset speaker, optionally controlled by instruct text.

    Args:
        text: Text to synthesize.
        speaker: One of the 9 preset speaker names.
        language: Language code or "Auto".
        instruct: Optional instruction, e.g. "Very happy." or "用特别愤怒的语气说".
        **gen_kwargs: Forwarded to model.generate_custom_voice().

    Returns:
        SynthesisResult with audio waveform, sample rate, and duration.
    """
    if _custom_voice_model is None:
        raise RuntimeError("CustomVoice model not loaded. Call init_models() first.")

    wavs, sr = _custom_voice_model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        **gen_kwargs,
    )
    audio = wavs[0]
    return SynthesisResult(
        audio=audio.astype(np.float32),
        sample_rate=sr,
        duration=len(audio) / sr,
    )


# ---------------------------------------------------------------------------
# Voice clone synthesis  (Base model)
# ---------------------------------------------------------------------------

def synthesize_voice_clone(
    text: str,
    language: str = "Auto",
    ref_audio: Any = None,
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
    **gen_kwargs,
) -> SynthesisResult:
    """
    Clone a speaker from reference audio and synthesize new text.

    Args:
        text: Text to synthesize.
        language: Language code or "Auto".
        ref_audio: Reference audio (path, base64 str, (ndarray, sr) tuple).
        ref_text: Reference transcript. Required unless x_vector_only_mode=True.
        x_vector_only_mode: If True, only speaker embedding is used.
        **gen_kwargs: Forwarded to model.generate_voice_clone().

    Returns:
        SynthesisResult with audio waveform, sample rate, and duration.
    """
    if _base_model is None:
        raise RuntimeError("Base (voice-clone) model not loaded. Call init_models() first.")

    wavs, sr = _base_model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
        **gen_kwargs,
    )
    audio = wavs[0]
    return SynthesisResult(
        audio=audio.astype(np.float32),
        sample_rate=sr,
        duration=len(audio) / sr,
    )


# ---------------------------------------------------------------------------
# Tokenizer encode / decode
# ---------------------------------------------------------------------------

def encode_tokens(
    audio: Any,
    sr: Optional[int] = None,
) -> TokenEncodeResult:
    """
    Encode audio into 12Hz discrete codes.

    Args:
        audio: path(s), base64 str(s), or numpy array(s).
        sr: sample rate (required for numpy input).

    Returns:
        TokenEncodeResult with codes, token rate, and durations.
    """
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded. Call init_models() first.")

    enc = _tokenizer.encode(audio, sr=sr)
    codes_list = enc.audio_codes  # List[Tensor (T, Q)]

    token_rate = _tokenizer.get_input_sample_rate() / _tokenizer.get_encode_downsample_rate()

    durations = []
    np_codes = []
    for c in codes_list:
        c_np = c.cpu().numpy().astype(np.int64)
        np_codes.append(c_np)
        durations.append(c_np.shape[0] / token_rate)

    return TokenEncodeResult(
        codes=np_codes,
        token_rate_hz=float(token_rate),
        durations=durations,
    )


def decode_tokens(
    codes: Union[List[np.ndarray], np.ndarray],
) -> TokenDecodeResult:
    """
    Decode 12Hz discrete codes back to audio.

    Args:
        codes: int64 array(s) of shape (T, Q) or list thereof.

    Returns:
        TokenDecodeResult with audio waveforms and sample rate.
    """
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded. Call init_models() first.")

    if isinstance(codes, np.ndarray):
        codes = [codes]

    dicts = [{"audio_codes": torch.from_numpy(c.astype(np.int64))} for c in codes]
    wavs, sr = _tokenizer.decode(dicts)
    audios = [w.astype(np.float32) for w in wavs]

    return TokenDecodeResult(audios=audios, sample_rate=sr)
