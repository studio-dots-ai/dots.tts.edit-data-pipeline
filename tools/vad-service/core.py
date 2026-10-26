"""
VAD Core: Core voice activity detection logic using soundfile + numpy.

This module provides the fundamental VAD functionality:
- Detecting non-silent (speech) regions in audio
- Trimming silence from audio boundaries

Uses soundfile for audio I/O (no ffmpeg subprocess, thread-safe).
Implements energy-based silence detection in pure numpy.
"""

import io
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start: float  # seconds
    end: float    # seconds
    duration: float  # seconds


@dataclass
class VADResult:
    """Result of VAD detection."""
    segments: List[SpeechSegment]
    total_duration: float  # seconds
    speech_duration: float  # seconds
    silence_duration: float  # seconds
    
    @property
    def num_segments(self) -> int:
        return len(self.segments)


@dataclass
class TrimResult:
    """Result of silence trimming."""
    audio: np.ndarray  # float32 mono waveform
    waveform: Optional[np.ndarray]  # same as audio (for backward compat)
    sample_rate: int
    original_duration: float  # seconds
    trimmed_duration: float  # seconds
    speech_start: float  # seconds, in original audio
    speech_end: float  # seconds, in original audio


# ---------------------------------------------------------------------------
# Audio loading (pure soundfile, no subprocess)
# ---------------------------------------------------------------------------

def load_audio_sf(
    audio_input: Union[str, Path, bytes, np.ndarray],
    sample_rate: int = 24000,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from various input formats.  Returns (float32 mono waveform, sr).
    """
    if isinstance(audio_input, np.ndarray):
        arr = audio_input.astype(np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr, sample_rate
    if isinstance(audio_input, (str, Path)):
        data, sr = sf.read(str(audio_input), dtype="float32", always_2d=False)
    elif isinstance(audio_input, bytes):
        data, sr = sf.read(io.BytesIO(audio_input), dtype="float32", always_2d=False)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Energy-based silence detection (pure numpy, replaces pydub.silence)
# ---------------------------------------------------------------------------

def _rms_db(samples: np.ndarray) -> float:
    """Compute RMS level in dBFS for int16-range samples."""
    if len(samples) == 0:
        return -np.inf
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms < 1e-10:
        return -np.inf
    # dBFS relative to int16 full scale (32768)
    return 20.0 * np.log10(rms / 32768.0)


def _detect_nonsilent_np(
    samples_i16: np.ndarray,
    sr: int,
    min_silence_len: int = 300,
    silence_thresh: float = -40.0,
    seek_step: int = 10,
) -> List[Tuple[int, int]]:
    """
    Detect non-silent ranges.  Returns list of (start_ms, end_ms).

    Mirrors pydub.silence.detect_nonsilent semantics:
      - Walk through audio in *seek_step* ms windows.
      - Windows below *silence_thresh* dBFS are silent.
      - Non-silent regions separated by >= *min_silence_len* ms of silence
        are reported as separate segments.
    """
    step_samples = int(sr * seek_step / 1000)
    if step_samples < 1:
        step_samples = 1
    min_silence_windows = max(1, int(min_silence_len / seek_step))

    n = len(samples_i16)
    # Compute per-window loudness
    num_windows = max(1, (n + step_samples - 1) // step_samples)
    is_silent = np.ones(num_windows, dtype=bool)
    for i in range(num_windows):
        start = i * step_samples
        end = min(start + step_samples, n)
        is_silent[i] = _rms_db(samples_i16[start:end]) < silence_thresh

    # Merge: find non-silent runs separated by >= min_silence_windows silent windows
    nonsilent_ranges: List[Tuple[int, int]] = []
    seg_start: Optional[int] = None
    silence_count = 0

    for i, silent in enumerate(is_silent):
        if not silent:
            if seg_start is None:
                seg_start = i
            silence_count = 0
        else:
            silence_count += 1
            if seg_start is not None and silence_count >= min_silence_windows:
                seg_end = i - silence_count + 1
                nonsilent_ranges.append((seg_start * seek_step, seg_end * seek_step))
                seg_start = None

    if seg_start is not None:
        # Find last non-silent window
        last_nonsil = num_windows - 1
        while last_nonsil >= seg_start and is_silent[last_nonsil]:
            last_nonsil -= 1
        nonsilent_ranges.append((seg_start * seek_step, (last_nonsil + 1) * seek_step))

    return nonsilent_ranges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_speech_segments(
    audio_input: Union[str, Path, bytes, np.ndarray],
    min_silence_len: int = 300,
    silence_thresh: int = -40,
    seek_step: int = 10,
    sample_rate: int = 24000,
) -> VADResult:
    """
    Detect speech segments in audio.
    
    Args:
        audio_input: Audio file path, bytes, or numpy array
        min_silence_len: Minimum silence length in milliseconds to consider as a break
        silence_thresh: Silence threshold in dBFS (e.g., -40)
        seek_step: Search step size in milliseconds
        sample_rate: Sample rate (used if audio_input is numpy array)
        
    Returns:
        VADResult with detected segments and statistics
    """
    waveform, sr = load_audio_sf(audio_input, sample_rate)
    total_duration = len(waveform) / sr

    # Convert to int16 range for dBFS computation
    samples_i16 = (waveform * 32768.0).clip(-32768, 32767).astype(np.int16)

    nonsilent_ranges = _detect_nonsilent_np(
        samples_i16, sr,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
    )

    segments = []
    speech_duration = 0.0
    for start_ms, end_ms in nonsilent_ranges:
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        duration = end_sec - start_sec
        segments.append(SpeechSegment(start=start_sec, end=end_sec, duration=duration))
        speech_duration += duration

    return VADResult(
        segments=segments,
        total_duration=total_duration,
        speech_duration=speech_duration,
        silence_duration=total_duration - speech_duration,
    )


def trim_silence(
    audio_input: Union[str, Path, bytes, np.ndarray],
    target_start_silence: float = 0.05,
    target_end_silence: float = 0.05,
    min_silence_len: int = 100,
    silence_thresh: int = -40,
    sample_rate: int = 24000,
    return_waveform: bool = False,
) -> TrimResult:
    """
    Trim silence from audio boundaries.
    
    Detects speech region and adjusts silence at start/end to target durations.
    """
    waveform, sr = load_audio_sf(audio_input, sample_rate)
    original_duration = len(waveform) / sr

    samples_i16 = (waveform * 32768.0).clip(-32768, 32767).astype(np.int16)

    nonsilent_ranges = _detect_nonsilent_np(
        samples_i16, sr,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=10,
    )

    if not nonsilent_ranges:
        raise ValueError("No speech detected in audio")

    speech_start_ms = nonsilent_ranges[0][0]
    speech_end_ms = nonsilent_ranges[-1][1]
    speech_start = speech_start_ms / 1000.0
    speech_end = speech_end_ms / 1000.0

    trim_start_s = max(0.0, speech_start - target_start_silence)
    trim_end_s = min(original_duration, speech_end + target_end_silence)

    start_idx = int(trim_start_s * sr)
    end_idx = min(len(waveform), int(trim_end_s * sr))
    trimmed = waveform[start_idx:end_idx]
    trimmed_duration = len(trimmed) / sr

    return TrimResult(
        audio=trimmed,
        waveform=trimmed if return_waveform else None,
        sample_rate=sr,
        original_duration=original_duration,
        trimmed_duration=trimmed_duration,
        speech_start=speech_start,
        speech_end=speech_end,
    )


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 mono waveform to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def get_speech_boundaries(
    audio_input: Union[str, Path, bytes, np.ndarray],
    min_silence_len: int = 100,
    silence_thresh: int = -40,
    sample_rate: int = 24000,
) -> Tuple[float, float]:
    """
    Get the overall speech boundaries (first speech start, last speech end).
    """
    result = detect_speech_segments(
        audio_input,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        sample_rate=sample_rate,
    )
    if not result.segments:
        raise ValueError("No speech detected in audio")
    return result.segments[0].start, result.segments[-1].end
