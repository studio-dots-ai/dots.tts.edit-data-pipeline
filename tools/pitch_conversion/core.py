"""
Pitch Conversion Core: PSOLA-based pitch shifting using Praat (parselmouth).

Core implementation without HTTP/service dependencies.
Can be used directly as a function or wrapped by api_service/api_func.

The pitch shift is performed via Praat's Manipulation object + overlap-add
resynthesis.  This shifts pitch without explicitly correcting formants:
formants move proportionally with pitch (acceptable for small shifts,
audible at large shifts — handled by optional codec refinement in the
service layer).

Supported methods (core only implements the signal-level shift):
  - "psola": Pure PSOLA pitch shift (no refinement)

Codec refinement ("psola+qwen3_codec", "psola+indextts2_cco",
"psola+mimo_codec") is handled by the api_service layer via external
HTTP calls.
"""

import io
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import parselmouth
import soundfile as sf

logger = logging.getLogger(__name__)

# Valid method names (core only handles "psola"; codec methods are service-level)
CORE_METHOD = "psola"
ALL_METHODS = (
    "psola",
    "psola+qwen3_codec",
    "psola+indextts2_cco",
    "psola+mimo_codec",
)


@dataclass
class PitchShiftResult:
    """Result of pitch shifting operation."""

    audio: np.ndarray  # Output waveform (float32)
    sample_rate: int
    original_duration: float
    shifted_duration: float
    semitones: float
    method: str = "psola"


# ------------------------------------
# PSOLA pitch shifting
# ------------------------------------


def pitch_shift_psola(
    input_path: str,
    output_path: str,
    semitones: float,
    time_step: float = 0.01,
    min_pitch: float = 75.0,
    max_pitch: float = 600.0,
) -> None:
    """
    Pitch shift using Praat PSOLA via Manipulation + overlap-add resynthesis.

    Args:
        input_path: Path to input audio file.
        output_path: Path to write output WAV file.
        semitones: Pitch shift in semitones (positive = higher, negative = lower).
        time_step: Analysis time step in seconds (default 0.01).
        min_pitch: Minimum pitch for analysis in Hz (default 75).
        max_pitch: Maximum pitch for analysis in Hz (default 600).

    The pitch ratio is computed as 2^(semitones/12).  All detected pitch
    points are multiplied by this ratio.
    """
    snd = parselmouth.Sound(input_path)
    manipulation = parselmouth.praat.call(
        snd, "To Manipulation", time_step, min_pitch, max_pitch
    )
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    pitch_ratio = 2 ** (semitones / 12)
    parselmouth.praat.call(
        pitch_tier, "Multiply frequencies", snd.xmin, snd.xmax, pitch_ratio
    )
    parselmouth.praat.call([manipulation, pitch_tier], "Replace pitch tier")
    shifted = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    shifted.save(output_path, "WAV")


def shift_audio(
    audio: Union[np.ndarray, bytes, str],
    semitones: float,
    sample_rate: Optional[int] = None,
    temp_dir: Optional[str] = None,
) -> PitchShiftResult:
    """
    Apply PSOLA pitch shifting to audio.

    Args:
        audio: Input audio as numpy array, bytes (wav), or file path.
        semitones: Pitch shift in semitones (+/- up to 24).
        sample_rate: Sample rate (required if audio is numpy array).
        temp_dir: Temporary directory for intermediate files.

    Returns:
        PitchShiftResult with shifted audio and metadata.
    """
    if not -24.0 <= semitones <= 24.0:
        raise ValueError(f"semitones must be between -24 and 24, got {semitones}")

    cleanup_temp = temp_dir is None
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="pitch_")

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

        output_path = os.path.join(temp_dir, f"shifted_{uuid.uuid4().hex}.wav")
        temp_files.append(output_path)

        pitch_shift_psola(input_path, output_path, semitones)

        shifted, sr = sf.read(output_path)
        shifted_duration = len(shifted) / sr

        logger.debug(
            f"PSOLA pitch shift: {original_duration:.2f}s -> {shifted_duration:.2f}s "
            f"(semitones={semitones:+.1f})"
        )

        return PitchShiftResult(
            audio=shifted.astype(np.float32),
            sample_rate=sr,
            original_duration=original_duration,
            shifted_duration=shifted_duration,
            semitones=semitones,
            method="psola",
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


def shift_audio_to_bytes(
    audio: Union[np.ndarray, bytes, str],
    semitones: float,
    sample_rate: Optional[int] = None,
    output_format: str = "wav",
) -> bytes:
    """
    Apply PSOLA pitch shifting and return result as bytes.
    """
    result = shift_audio(audio, semitones, sample_rate)

    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format=output_format.upper())
    buffer.seek(0)
    return buffer.read()
