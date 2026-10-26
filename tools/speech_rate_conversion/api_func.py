"""
Speech Rate Conversion Function API: Direct function interface.

Provides the same interface as the HTTP service but as direct function calls.
Use this in the data agent pipeline for local execution without HTTP overhead.

Supported methods:
  - "wsola": SoX WSOLA time stretching (default)
  - "token": Token-level nearest-neighbor resampling (requires external tokenizer)
"""

import io
import logging
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

from .core import (
    stretch_audio,
    stretch_audio_to_bytes,
    stretch_tokens,
    StretchResult,
    TokenStretchResult,
    METHODS,
)

logger = logging.getLogger(__name__)


def stretch(
    audio: Union[np.ndarray, bytes, str],
    rate: float,
    sample_rate: Optional[int] = None,
    output_format: str = "wav",
    method: str = "wsola",
) -> Union[bytes, Tuple[np.ndarray, int]]:
    """
    Apply time stretching to audio (function API).
    
    Args:
        audio: Input audio as numpy array, bytes (wav), or file path
        rate: Speed factor (>1 = faster/shorter, <1 = slower/longer)
        sample_rate: Sample rate (required if audio is numpy array)
        output_format: "wav" returns bytes, "numpy" returns (array, sample_rate)
        method: "wsola" for signal-level or "token" is not supported here
                (use stretch_tokens_func for token-level)
    
    Returns:
        If output_format="wav": WAV bytes
        If output_format="numpy": Tuple of (numpy array, sample_rate)
    
    Note:
        Token-based stretching requires external services (IndexTTS / Qwen3-TTS)
        for encode/decode. Use ``token_stretch`` for the token-only step.
    """
    if method == "token":
        raise ValueError(
            "Token-based stretching via this function is not supported. "
            "Use token_stretch() for token manipulation, then decode via "
            "the appropriate service (IndexTTS or Qwen3-TTS)."
        )
    if method != "wsola":
        raise ValueError(f"Unsupported method: {method}. Supported: {METHODS}")

    if output_format == "wav":
        return stretch_audio_to_bytes(audio, rate, sample_rate)
    elif output_format == "numpy":
        result = stretch_audio(audio, rate, sample_rate)
        return result.audio, result.sample_rate
    else:
        raise ValueError(f"output_format must be 'wav' or 'numpy', got '{output_format}'")


def token_stretch(
    codes: np.ndarray,
    rate: float,
    token_rate: float = 50.0,
) -> TokenStretchResult:
    """
    Stretch a discrete token sequence via nearest-neighbor resampling.

    This is the pure token-manipulation step. Encoding (audio → tokens) and
    decoding (tokens → audio) should be done by the caller via the
    appropriate service (IndexTTS at 50Hz, Qwen3-TTS at 12Hz).

    Args:
        codes: 1-D int64 array of discrete speech tokens.
        rate: Speed factor (>1 = faster/fewer tokens, <1 = slower/more tokens).
        token_rate: Token rate in Hz (50 for IndexTTS, 12 for Qwen3-TTS).

    Returns:
        TokenStretchResult with original and stretched codes.
    """
    return stretch_tokens(codes, rate, token_rate)


def stretch_with_metadata(
    audio: Union[np.ndarray, bytes, str],
    rate: float,
    sample_rate: Optional[int] = None,
) -> StretchResult:
    """
    Apply WSOLA time stretching and return full metadata.
    """
    return stretch_audio(audio, rate, sample_rate)
