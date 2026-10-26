"""
Volume Normalizer Function API: Direct function interface.

Provides the same interface as the HTTP service but as direct function calls.
Use this in the data agent pipeline for local execution without HTTP overhead.
"""

import io
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from .core import (
    normalize_audio,
    normalize_audio_to_bytes,
    normalize_multiple,
    normalize_to_match,
    NormalizeResult,
)

logger = logging.getLogger(__name__)


def normalize(
    audio: Union[np.ndarray, bytes, str],
    target_db: float = -20.0,
    sample_rate: Optional[int] = None,
    method: str = "rms",
    headroom_db: float = -1.0,
    output_format: str = "wav",
) -> Union[bytes, Tuple[np.ndarray, int]]:
    """
    Normalize audio loudness (function API, same interface as HTTP service).
    
    Args:
        audio: Input audio as numpy array, bytes (wav), or file path
        target_db: Target loudness level in dB (default: -20 dB)
        sample_rate: Sample rate (required if audio is numpy array)
        method: Normalization method: "rms" or "peak"
        headroom_db: Maximum peak level to prevent clipping (default: -1 dB)
        output_format: "wav" returns bytes, "numpy" returns (array, sample_rate)
    
    Returns:
        If output_format="wav": WAV bytes
        If output_format="numpy": Tuple of (numpy array, sample_rate)
    
    Example:
        >>> from tools.volume_normalizer import normalize
        >>> # From file
        >>> wav_bytes = normalize("input.wav", target_db=-18.0)
        >>> # From bytes
        >>> wav_bytes = normalize(audio_bytes, target_db=-20.0)
        >>> # From numpy
        >>> audio, sr = normalize(audio_array, sample_rate=22050, output_format="numpy")
    """
    if output_format == "wav":
        return normalize_audio_to_bytes(audio, target_db, sample_rate, method, headroom_db)
    
    elif output_format == "numpy":
        result = normalize_audio(audio, target_db, sample_rate, method, headroom_db)
        return result.audio, result.sample_rate
    
    else:
        raise ValueError(f"output_format must be 'wav' or 'numpy', got '{output_format}'")


def normalize_with_metadata(
    audio: Union[np.ndarray, bytes, str],
    target_db: float = -20.0,
    sample_rate: Optional[int] = None,
    method: str = "rms",
    headroom_db: float = -1.0,
) -> NormalizeResult:
    """
    Normalize audio and return full metadata.
    
    Args:
        audio: Input audio as numpy array, bytes (wav), or file path
        target_db: Target loudness level in dB
        sample_rate: Sample rate (required if audio is numpy array)
        method: Normalization method
        headroom_db: Maximum peak level
    
    Returns:
        NormalizeResult with audio, sample_rate, original_rms, normalized_rms, gain_db
    """
    return normalize_audio(audio, target_db, sample_rate, method, headroom_db)


def normalize_batch(
    audios: List[Union[np.ndarray, bytes, str]],
    target_db: float = -20.0,
    sample_rates: Optional[List[int]] = None,
    method: str = "rms",
    headroom_db: float = -1.0,
    output_format: str = "wav",
) -> List[Union[bytes, Tuple[np.ndarray, int]]]:
    """
    Normalize multiple audio files to the same target level.
    
    Args:
        audios: List of audio inputs
        target_db: Target loudness level in dB
        sample_rates: Sample rates (required for numpy arrays)
        method: Normalization method
        headroom_db: Maximum peak level
        output_format: "wav" or "numpy"
    
    Returns:
        List of normalized audio in specified format
    """
    results = normalize_multiple(audios, target_db, sample_rates, method, headroom_db)
    
    if output_format == "wav":
        outputs = []
        for result in results:
            buffer = io.BytesIO()
            sf.write(buffer, result.audio, result.sample_rate, format="WAV")
            buffer.seek(0)
            outputs.append(buffer.read())
        return outputs
    
    elif output_format == "numpy":
        return [(result.audio, result.sample_rate) for result in results]
    
    else:
        raise ValueError(f"output_format must be 'wav' or 'numpy', got '{output_format}'")


def normalize_to_reference(
    source_audio: Union[np.ndarray, bytes, str],
    reference_audio: Union[np.ndarray, bytes, str],
    source_sr: Optional[int] = None,
    reference_sr: Optional[int] = None,
    method: str = "rms",
    output_format: str = "wav",
) -> Union[bytes, Tuple[np.ndarray, int]]:
    """
    Normalize source audio to match reference audio's loudness.
    
    Args:
        source_audio: Audio to normalize
        reference_audio: Reference audio for target loudness
        source_sr: Source sample rate (required if numpy)
        reference_sr: Reference sample rate (required if numpy)
        method: Loudness measurement method
        output_format: "wav" or "numpy"
    
    Returns:
        Normalized audio in specified format
    """
    result = normalize_to_match(source_audio, reference_audio, source_sr, reference_sr, method)
    
    if output_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, result.audio, result.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
    
    elif output_format == "numpy":
        return result.audio, result.sample_rate
    
    else:
        raise ValueError(f"output_format must be 'wav' or 'numpy', got '{output_format}'")
