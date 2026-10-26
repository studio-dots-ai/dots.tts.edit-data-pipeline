"""
VAD API Functions: High-level API for voice activity detection.

This module provides easy-to-use functions for common VAD tasks:
- Detecting speech segments
- Trimming silence from audio
- Getting speech boundaries
"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .core import (
        VADResult,
        TrimResult,
        SpeechSegment,
        detect_speech_segments,
        trim_silence,
        get_speech_boundaries,
        waveform_to_wav_bytes,
    )
except ImportError:
    from core import (
        VADResult,
        TrimResult,
        SpeechSegment,
        detect_speech_segments,
        trim_silence,
        get_speech_boundaries,
        waveform_to_wav_bytes,
    )

logger = logging.getLogger(__name__)


def detect(
    audio: Union[str, Path, bytes, np.ndarray],
    min_silence_len: int = 300,
    silence_thresh: int = -40,
    seek_step: int = 10,
    sample_rate: int = 24000,
) -> VADResult:
    """
    Detect speech segments in audio.

    Args:
        audio: Audio file path, bytes, or numpy array (float32 mono)
        min_silence_len: Minimum silence length in ms to consider as break
        silence_thresh: Silence threshold in dBFS (e.g., -40)
        seek_step: Search step size in ms
        sample_rate: Sample rate (only used for numpy input)

    Returns:
        VADResult containing:
            - segments: List[SpeechSegment] with start, end, duration
            - total_duration: Total audio duration in seconds
            - speech_duration: Total speech duration in seconds
            - silence_duration: Total silence duration in seconds
            - num_segments: Number of detected segments

    Example:
        >>> result = detect("audio.wav")
        >>> for seg in result.segments:
        ...     print(f"Speech: {seg.start:.2f}s - {seg.end:.2f}s")
    """
    return detect_speech_segments(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
        sample_rate=sample_rate,
    )


def detect_to_dict(
    audio: Union[str, Path, bytes, np.ndarray],
    min_silence_len: int = 300,
    silence_thresh: int = -40,
    seek_step: int = 10,
    sample_rate: int = 24000,
) -> Dict:
    """
    Detect speech segments and return as dictionary.

    Args:
        Same as detect()

    Returns:
        Dictionary with segments and statistics

    Example:
        >>> result = detect_to_dict("audio.wav")
        >>> print(f"Found {result['num_segments']} segments")
    """
    result = detect(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
        sample_rate=sample_rate,
    )

    return {
        "segments": [
            {"start": seg.start, "end": seg.end, "duration": seg.duration}
            for seg in result.segments
        ],
        "total_duration": result.total_duration,
        "speech_duration": result.speech_duration,
        "silence_duration": result.silence_duration,
        "num_segments": result.num_segments,
    }


def trim(
    audio: Union[str, Path, bytes, np.ndarray],
    target_start_silence: float = 0.05,
    target_end_silence: float = 0.05,
    min_silence_len: int = 100,
    silence_thresh: int = -40,
    sample_rate: int = 24000,
    output_format: str = "waveform",
) -> Union[np.ndarray, bytes, TrimResult]:
    """
    Trim silence from audio boundaries.

    Args:
        audio: Audio file path, bytes, or numpy array
        target_start_silence: Target silence at start in seconds (default 50ms)
        target_end_silence: Target silence at end in seconds (default 50ms)
        min_silence_len: Minimum silence length for detection in ms
        silence_thresh: Silence threshold in dBFS
        sample_rate: Sample rate (for numpy input)
        output_format: Output format - 'waveform', 'wav', or 'full'

    Returns:
        - If output_format='waveform': numpy array (float32 mono)
        - If output_format='wav': WAV bytes
        - If output_format='full': TrimResult with all metadata

    Example:
        >>> waveform = trim("audio.wav", target_start_silence=0.1)
        >>> # Or get WAV bytes
        >>> wav_bytes = trim("audio.wav", output_format="wav")
    """
    return_waveform = output_format in ("waveform", "full")

    result = trim_silence(
        audio,
        target_start_silence=target_start_silence,
        target_end_silence=target_end_silence,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        sample_rate=sample_rate,
        return_waveform=return_waveform,
    )

    if output_format == "waveform":
        return result.waveform
    elif output_format == "wav":
        return waveform_to_wav_bytes(result.audio, result.sample_rate)
    else:
        return result


def trim_with_metadata(
    audio: Union[str, Path, bytes, np.ndarray],
    target_start_silence: float = 0.05,
    target_end_silence: float = 0.05,
    min_silence_len: int = 100,
    silence_thresh: int = -40,
    sample_rate: int = 24000,
) -> Tuple[np.ndarray, Dict]:
    """
    Trim silence and return waveform with metadata.

    Args:
        Same as trim()

    Returns:
        Tuple of (waveform, metadata_dict)

    Example:
        >>> waveform, meta = trim_with_metadata("audio.wav")
        >>> print(f"Trimmed: {meta['original_duration']:.2f}s -> {meta['trimmed_duration']:.2f}s")
    """
    result = trim_silence(
        audio,
        target_start_silence=target_start_silence,
        target_end_silence=target_end_silence,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        sample_rate=sample_rate,
        return_waveform=True,
    )

    metadata = {
        "sample_rate": result.sample_rate,
        "original_duration": result.original_duration,
        "trimmed_duration": result.trimmed_duration,
        "speech_start": result.speech_start,
        "speech_end": result.speech_end,
    }

    return result.waveform, metadata


def get_boundaries(
    audio: Union[str, Path, bytes, np.ndarray],
    min_silence_len: int = 100,
    silence_thresh: int = -40,
    sample_rate: int = 24000,
) -> Tuple[float, float]:
    """
    Get the overall speech boundaries.

    Args:
        audio: Audio file path, bytes, or numpy array
        min_silence_len: Minimum silence length in ms
        silence_thresh: Silence threshold in dBFS
        sample_rate: Sample rate (for numpy input)

    Returns:
        Tuple of (speech_start_sec, speech_end_sec)

    Raises:
        ValueError: If no speech is detected

    Example:
        >>> start, end = get_boundaries("audio.wav")
        >>> print(f"Speech region: {start:.2f}s - {end:.2f}s")
    """
    return get_speech_boundaries(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        sample_rate=sample_rate,
    )


def get_segment_times(
    audio: Union[str, Path, bytes, np.ndarray],
    min_silence_len: int = 300,
    silence_thresh: int = -40,
    sample_rate: int = 24000,
) -> List[Tuple[float, float]]:
    """
    Get list of speech segment times.

    Args:
        audio: Audio file path, bytes, or numpy array
        min_silence_len: Minimum silence length in ms
        silence_thresh: Silence threshold in dBFS
        sample_rate: Sample rate (for numpy input)

    Returns:
        List of (start, end) tuples in seconds

    Example:
        >>> times = get_segment_times("audio.wav")
        >>> for start, end in times:
        ...     print(f"Segment: {start:.2f}s - {end:.2f}s")
    """
    result = detect(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        sample_rate=sample_rate,
    )

    return [(seg.start, seg.end) for seg in result.segments]
