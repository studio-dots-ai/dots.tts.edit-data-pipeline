"""
Volume Normalizer Core: Audio loudness normalization.

Core implementation for normalizing audio loudness to a target level.
Uses RMS-based normalization by default, with optional peak normalization.
"""

import io
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class NormalizeResult:
    """Result of volume normalization."""
    audio: np.ndarray  # Normalized waveform
    sample_rate: int
    original_rms: float  # Original RMS level (dB)
    normalized_rms: float  # Normalized RMS level (dB)
    gain_db: float  # Applied gain in dB


def compute_rms_db(audio: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute RMS level in dB.
    
    Args:
        audio: Audio waveform
        eps: Small value to avoid log(0)
    
    Returns:
        RMS level in dB
    """
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    return 20.0 * np.log10(rms + eps)


def compute_peak_db(audio: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute peak level in dB.
    
    Args:
        audio: Audio waveform
        eps: Small value to avoid log(0)
    
    Returns:
        Peak level in dB
    """
    peak = np.max(np.abs(audio))
    return 20.0 * np.log10(peak + eps)


def normalize_audio(
    audio: Union[np.ndarray, bytes, str],
    target_db: float = -20.0,
    sample_rate: Optional[int] = None,
    method: str = "rms",
    headroom_db: float = -1.0,
) -> NormalizeResult:
    """
    Normalize audio loudness to target level.
    
    Args:
        audio: Input audio as numpy array, bytes (wav), or file path
        target_db: Target loudness level in dB (default: -20 dB)
        sample_rate: Sample rate (required if audio is numpy array)
        method: Normalization method: "rms" or "peak"
        headroom_db: Maximum peak level to prevent clipping (default: -1 dB)
    
    Returns:
        NormalizeResult with normalized audio and metadata
    """
    # Load audio
    if isinstance(audio, str):
        if not os.path.exists(audio):
            raise ValueError(f"Audio file not found: {audio}")
        audio_data, sr = sf.read(audio)
    elif isinstance(audio, bytes):
        audio_data, sr = sf.read(io.BytesIO(audio))
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is numpy array")
        audio_data = audio.copy()
        sr = sample_rate
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")
    
    # Ensure float
    audio_data = audio_data.astype(np.float32)
    
    # Compute original level
    if method == "rms":
        original_level = compute_rms_db(audio_data)
    elif method == "peak":
        original_level = compute_peak_db(audio_data)
    else:
        raise ValueError(f"Unknown method: {method}, must be 'rms' or 'peak'")
    
    # Compute gain
    gain_db = target_db - original_level
    
    # Apply gain
    gain_linear = 10.0 ** (gain_db / 20.0)
    normalized = audio_data * gain_linear
    
    # Check for clipping and apply headroom
    peak_after = compute_peak_db(normalized)
    if peak_after > headroom_db:
        # Reduce gain to maintain headroom
        reduction_db = peak_after - headroom_db
        reduction_linear = 10.0 ** (-reduction_db / 20.0)
        normalized = normalized * reduction_linear
        gain_db -= reduction_db
        logger.debug(f"Applied headroom reduction: {reduction_db:.2f} dB")
    
    # Compute final RMS
    normalized_rms = compute_rms_db(normalized)
    
    return NormalizeResult(
        audio=normalized,
        sample_rate=sr,
        original_rms=original_level,
        normalized_rms=normalized_rms,
        gain_db=gain_db,
    )


def normalize_audio_to_bytes(
    audio: Union[np.ndarray, bytes, str],
    target_db: float = -20.0,
    sample_rate: Optional[int] = None,
    method: str = "rms",
    headroom_db: float = -1.0,
    output_format: str = "wav",
) -> bytes:
    """
    Normalize audio and return as bytes.
    
    Args:
        audio: Input audio as numpy array, bytes (wav), or file path
        target_db: Target loudness level in dB
        sample_rate: Sample rate (required if audio is numpy array)
        method: Normalization method: "rms" or "peak"
        headroom_db: Maximum peak level to prevent clipping
        output_format: Output format (currently only "wav")
    
    Returns:
        WAV bytes of normalized audio
    """
    result = normalize_audio(audio, target_db, sample_rate, method, headroom_db)
    
    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format=output_format.upper())
    buffer.seek(0)
    return buffer.read()


def normalize_multiple(
    audios: List[Union[np.ndarray, bytes, str]],
    target_db: float = -20.0,
    sample_rates: Optional[List[int]] = None,
    method: str = "rms",
    headroom_db: float = -1.0,
) -> List[NormalizeResult]:
    """
    Normalize multiple audio files to the same target level.
    
    Args:
        audios: List of audio inputs
        target_db: Target loudness level in dB
        sample_rates: Sample rates (required for numpy arrays)
        method: Normalization method
        headroom_db: Maximum peak level
    
    Returns:
        List of NormalizeResult objects
    """
    results = []
    for i, audio in enumerate(audios):
        sr = sample_rates[i] if sample_rates else None
        result = normalize_audio(audio, target_db, sr, method, headroom_db)
        results.append(result)
    return results


def normalize_to_match(
    source_audio: Union[np.ndarray, bytes, str],
    reference_audio: Union[np.ndarray, bytes, str],
    source_sr: Optional[int] = None,
    reference_sr: Optional[int] = None,
    method: str = "rms",
) -> NormalizeResult:
    """
    Normalize source audio to match reference audio's loudness.
    
    Args:
        source_audio: Audio to normalize
        reference_audio: Reference audio for target loudness
        source_sr: Source sample rate (required if numpy)
        reference_sr: Reference sample rate (required if numpy)
        method: Loudness measurement method
    
    Returns:
        NormalizeResult with source normalized to match reference
    """
    # Get reference level
    if isinstance(reference_audio, str):
        ref_data, _ = sf.read(reference_audio)
    elif isinstance(reference_audio, bytes):
        ref_data, _ = sf.read(io.BytesIO(reference_audio))
    else:
        ref_data = reference_audio
    
    if method == "rms":
        target_db = compute_rms_db(ref_data)
    else:
        target_db = compute_peak_db(ref_data)
    
    return normalize_audio(source_audio, target_db, source_sr, method)
