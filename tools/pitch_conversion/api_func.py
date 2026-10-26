"""
Pitch Conversion Function API: Direct function interface.

Provides the same interface as the HTTP service but as direct function calls.
Use this in the data agent pipeline for local execution without HTTP overhead.

Only the pure PSOLA method is available here.  Codec refinement methods
("psola+qwen3_codec", "psola+indextts2_cco", "psola+mimo_codec") require
external GPU services and are only available through the HTTP service
(api_service.py).
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np

from .core import (
    shift_audio,
    shift_audio_to_bytes,
    PitchShiftResult,
    CORE_METHOD,
    ALL_METHODS,
)

logger = logging.getLogger(__name__)


def pitch_shift(
    audio: Union[np.ndarray, bytes, str],
    semitones: float,
    sample_rate: Optional[int] = None,
    output_format: str = "wav",
    method: str = "psola",
) -> Union[bytes, Tuple[np.ndarray, int]]:
    """
    Apply pitch shifting to audio (function API).

    Args:
        audio: Input audio as numpy array, bytes (wav), or file path.
        semitones: Pitch shift in semitones (+/- up to 24).
        sample_rate: Sample rate (required if audio is numpy array).
        output_format: "wav" returns bytes, "numpy" returns (array, sample_rate).
        method: Only "psola" is supported here.  Use the HTTP service for
                codec refinement methods.

    Returns:
        If output_format="wav": WAV bytes
        If output_format="numpy": Tuple of (numpy array, sample_rate)
    """
    if method != CORE_METHOD:
        raise ValueError(
            f"Only '{CORE_METHOD}' method is supported in the function API. "
            f"For codec refinement methods ({', '.join(ALL_METHODS[1:])}), "
            f"use the HTTP service."
        )

    if output_format == "wav":
        return shift_audio_to_bytes(audio, semitones, sample_rate)
    elif output_format == "numpy":
        result = shift_audio(audio, semitones, sample_rate)
        return result.audio, result.sample_rate
    else:
        raise ValueError(
            f"output_format must be 'wav' or 'numpy', got '{output_format}'"
        )


def pitch_shift_with_metadata(
    audio: Union[np.ndarray, bytes, str],
    semitones: float,
    sample_rate: Optional[int] = None,
) -> PitchShiftResult:
    """
    Apply PSOLA pitch shifting and return full metadata.
    """
    return shift_audio(audio, semitones, sample_rate)
