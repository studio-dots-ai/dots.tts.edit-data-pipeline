"""
Speech Rate Conversion Core: WSOLA-based time stretching using SoX,
plus token-based time stretching via nearest-neighbor resampling.

Core implementation without HTTP/service dependencies.
Can be used directly as a function or wrapped by api_service/api_func.

Supported methods:
  - "wsola": SoX tempo -s (WSOLA algorithm for speech)
  - "token": Nearest-neighbor resampling of discrete speech tokens
"""

import io
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import sox

logger = logging.getLogger(__name__)

# Valid method names
METHODS = ("wsola", "token")


@dataclass
class StretchResult:
    """Result of time stretching operation."""
    audio: np.ndarray  # Output waveform
    sample_rate: int
    original_duration: float
    stretched_duration: float
    rate: float
    method: str = "wsola"


@dataclass
class TokenStretchResult:
    """Result of token-level stretching (before decoding to audio)."""
    original_codes: np.ndarray   # 1-D int64 original codes
    stretched_codes: np.ndarray  # 1-D int64 stretched codes
    token_rate: float            # Hz (50 for IndexTTS, 12 for Qwen3-TTS)
    rate: float                  # applied speed factor
    original_num_tokens: int
    stretched_num_tokens: int


# ------------------------------------
# WSOLA (signal-level) methods
# ------------------------------------


def time_stretch_wsola(
    input_path: str,
    output_path: str,
    rate: float,
) -> None:
    """
    Apply WSOLA-based time stretching using SoX tempo -s.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        rate: Speed factor (>1 = faster, <1 = slower)
    
    SoX tempo effect with -s flag uses WSOLA algorithm optimized for speech.
    """
    tfm = sox.Transformer()
    tfm.tempo(rate, audio_type='s')  # 's' = speech (uses WSOLA)
    tfm.build(input_path, output_path)


def stretch_audio(
    audio: Union[np.ndarray, bytes, str],
    rate: float,
    sample_rate: Optional[int] = None,
    temp_dir: Optional[str] = None,
) -> StretchResult:
    """
    Apply WSOLA time stretching to audio.
    
    Args:
        audio: Input audio as numpy array, bytes (wav), or file path
        rate: Speed factor (>1 = faster/shorter, <1 = slower/longer)
        sample_rate: Sample rate (required if audio is numpy array)
        temp_dir: Temporary directory for intermediate files
    
    Returns:
        StretchResult with stretched audio and metadata
    """
    if not 0.1 < rate < 10.0:
        raise ValueError(f"Rate must be between 0.1 and 10.0, got {rate}")
    
    cleanup_temp = temp_dir is None
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="stretch_")
    
    temp_files = []
    
    try:
        if isinstance(audio, str):
            if not os.path.exists(audio):
                raise ValueError(f"Audio file not found: {audio}")
            input_path = audio
        elif isinstance(audio, bytes):
            input_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.wav")
            with open(input_path, "wb") as f:
                f.write(audio)
            temp_files.append(input_path)
        elif isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio is numpy array")
            input_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.wav")
            sf.write(input_path, audio, sample_rate)
            temp_files.append(input_path)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        info = sf.info(input_path)
        original_duration = info.duration
        sr = info.samplerate
        
        output_path = os.path.join(temp_dir, f"stretched_{uuid.uuid4().hex}.wav")
        temp_files.append(output_path)
        
        time_stretch_wsola(input_path, output_path, rate)
        
        stretched, sr = sf.read(output_path)
        stretched_duration = len(stretched) / sr
        
        logger.debug(f"WSOLA stretch: {original_duration:.2f}s -> {stretched_duration:.2f}s (rate={rate})")
        
        return StretchResult(
            audio=stretched.astype(np.float32),
            sample_rate=sr,
            original_duration=original_duration,
            stretched_duration=stretched_duration,
            rate=rate,
            method="wsola",
        )
    
    finally:
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")
        
        if cleanup_temp and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass


def stretch_audio_to_bytes(
    audio: Union[np.ndarray, bytes, str],
    rate: float,
    sample_rate: Optional[int] = None,
    output_format: str = "wav",
) -> bytes:
    """
    Apply WSOLA time stretching and return result as bytes.
    """
    result = stretch_audio(audio, rate, sample_rate)
    
    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format=output_format.upper())
    buffer.seek(0)
    return buffer.read()


# ------------------------------------
# Token-level stretching
# ------------------------------------


def stretch_tokens(
    codes: np.ndarray,
    rate: float,
    token_rate: float = 50.0,
) -> TokenStretchResult:
    """
    Stretch a discrete token sequence via nearest-neighbor resampling along the
    time axis (axis 0).

    Args:
        codes: int64 numpy array of discrete speech tokens.
               Can be 1-D (T,) or 2-D (T, D) for multi-codebook models.
        rate: Speed factor. >1 = faster (fewer tokens), <1 = slower (more tokens).
        token_rate: Token rate in Hz (50 for IndexTTS, 12 for Qwen3-TTS).

    Returns:
        TokenStretchResult with original and stretched codes (same ndim as input).
    """
    if not 0.1 < rate < 10.0:
        raise ValueError(f"Rate must be between 0.1 and 10.0, got {rate}")
    
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim == 1:
        n_orig = len(codes)
    elif codes.ndim == 2:
        n_orig = codes.shape[0]  # time axis
    else:
        raise ValueError(f"Expected 1-D or 2-D codes, got {codes.ndim}-D")

    if n_orig == 0:
        raise ValueError("Empty token sequence")

    # Target length: slower → more tokens, faster → fewer tokens
    n_target = max(1, int(round(n_orig / rate)))

    # Nearest-neighbor resampling along time axis
    indices = np.round(np.linspace(0, n_orig - 1, n_target)).astype(int)
    stretched = codes[indices]  # works for both 1-D and 2-D (fancy indexing on axis 0)

    logger.debug(
        f"Token stretch: {n_orig} -> {n_target} tokens "
        f"(rate={rate}, token_rate={token_rate}Hz, shape={codes.shape}->{stretched.shape})"
    )

    return TokenStretchResult(
        original_codes=codes,
        stretched_codes=stretched,
        token_rate=token_rate,
        rate=rate,
        original_num_tokens=n_orig,
        stretched_num_tokens=n_target,
    )
