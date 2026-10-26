"""Audio timeline helpers for tagged text edit synthesis."""

from __future__ import annotations

import io
from typing import Sequence, Tuple

import numpy as np
from pydub import AudioSegment, silence as pydub_silence
import soundfile as sf


def wav_bytes_to_mono_float(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def mono_float_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


def nonsilent_speech_duration(
    audio_bytes: bytes,
    *,
    min_silence_len: int = 200,
    silence_thresh: int = -40,
    seek_step: int = 10,
) -> float:
    """Return nonsilent speech duration for tagged edit speech-rate estimates."""
    aseg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    total_duration = len(aseg) / 1000.0
    nonsilent_ranges = pydub_silence.detect_nonsilent(
        aseg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
    )
    if not nonsilent_ranges:
        return total_duration
    return sum(end - start for start, end in nonsilent_ranges) / 1000.0


def _samples(sec: float, sample_rate: int) -> int:
    return max(0, int(round(float(sec) * sample_rate)))


def build_target_condition_waveform(
    source_waveform: np.ndarray,
    sample_rate: int,
    source_spans: Sequence[Sequence[float]],
    target_spans: Sequence[Sequence[float]],
    target_duration: float,
) -> np.ndarray:
    """Copy unedited source regions onto the target timeline and zero edits."""
    source = np.asarray(source_waveform, dtype=np.float32).squeeze()
    if source.ndim != 1:
        raise ValueError("source_waveform must be mono")
    if len(source_spans) != len(target_spans):
        raise ValueError("source_spans and target_spans must have equal length")

    target_len = max(1, _samples(target_duration, sample_rate))
    target = np.zeros(target_len, dtype=np.float32)

    source_cursor = 0
    target_cursor = 0
    source_len = len(source)
    for source_span, target_span in sorted(
        zip(source_spans, target_spans), key=lambda item: (item[0][0], item[0][1])
    ):
        source_start = min(source_len, _samples(source_span[0], sample_rate))
        source_end = min(source_len, _samples(source_span[1], sample_rate))
        target_start = min(target_len, _samples(target_span[0], sample_rate))
        copy_len = max(0, min(source_start - source_cursor, target_start - target_cursor))
        if copy_len:
            target[target_cursor : target_cursor + copy_len] = source[
                source_cursor : source_cursor + copy_len
            ]
        source_cursor = max(source_cursor, source_end)
        target_cursor = min(target_len, _samples(target_span[1], sample_rate))

    copy_len = max(0, min(source_len - source_cursor, target_len - target_cursor))
    if copy_len:
        target[target_cursor : target_cursor + copy_len] = source[
            source_cursor : source_cursor + copy_len
        ]
    return target
