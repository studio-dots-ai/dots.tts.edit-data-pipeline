from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from speech_edit.pipelines.emotion.emotion_acoustic_effect import (
    AcousticEffectConfig,
    F0Track,
    compute_effect_from_tracks,
    compute_emotion_acoustic_effect,
    extract_f0_track,
)


def _segments(
    *,
    mismatch: bool = False,
    zero_duration: bool = False,
    multiple_regions: bool = False,
):
    source_tokens = [
        {"char": "hello", "start": 0.0, "end": 0.2},
        {"char": "world", "start": 0.2, "end": 0.4},
    ]
    target_tokens = [
        {
            "char": "HELLO" if not mismatch else "different",
            "start": 0.0,
            "end": 0.4,
        },
        {
            "char": "world",
            "start": 0.4,
            "end": 0.4 if zero_duration else 0.8,
        },
    ]
    source_spans = [{"text": "hello world", "span": [0.0, 0.4]}]
    target_spans = [{"text": "hello world", "span": [0.0, 0.8]}]
    if multiple_regions:
        source_spans = [
            {"text": "hello", "span": [0.0, 0.2]},
            {"text": "world", "span": [0.2, 0.4]},
        ]
        target_spans = [
            {"text": "hello", "span": [0.0, 0.4]},
            {"text": "world", "span": [0.4, 0.8]},
        ]
    return [
        {
            "chunk_id": "src",
            "edit_time_spans": source_spans,
            "alignment": {
                "status": "success",
                "level": "word",
                "alignments": source_tokens,
            },
        },
        {
            "chunk_id": "tgt",
            "edit_time_spans": target_spans,
            "alignment": {
                "status": "success",
                "level": "word",
                "alignments": target_tokens,
            },
        },
    ]


def _track(duration: float, frequency: float) -> F0Track:
    times = np.arange(0.01, duration, 0.01)
    return F0Track(times, np.full(times.shape, frequency, dtype=np.float64))


def _changed_segments(*, multiple_regions: bool = False):
    if multiple_regions:
        return [
            {"segment_idx": 0, "segment_text": "hello"},
            {"segment_idx": 1, "segment_text": "world"},
        ]
    return [{"segment_idx": 0, "segment_text": "hello world"}]


def _wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def test_compute_effect_matches_reference_payload_values():
    source_audio = np.full(6400, 0.1, dtype=np.float64)
    target_audio = np.full(12800, 0.2, dtype=np.float64)
    result = compute_effect_from_tracks(
        segments=_segments(),
        changed_segments=_changed_segments(),
        source_track=_track(0.4, 100.0),
        target_track=_track(0.8, 200.0),
        source_audio=source_audio,
        source_sample_rate=16000,
        target_audio=target_audio,
        target_sample_rate=16000,
    )

    assert result["status"] == "ok"
    assert "all_intensity_only" not in result
    assert "pass" not in result
    assert result["num_regions"] == 1
    assert result["num_token_pairs"] == 2
    assert result["num_duration_pairs"] == 2
    assert result["num_pitch_pairs"] == 2
    assert result["num_energy_pairs"] == 2
    assert result["duration_median_abs_log2_ratio"] == pytest.approx(1.0)
    assert result["pitch_level_median_abs_semitones"] == pytest.approx(12.0)
    assert result["pitch_range_median_abs_delta_semitones"] == pytest.approx(0.0)
    assert result["energy_level_median_abs_db"] == pytest.approx(
        20.0 * np.log10(2.0)
    )
    assert result["pitch_direction_coherence"] == pytest.approx(1.0)
    assert result["energy_direction_coherence"] == pytest.approx(1.0)
    assert result["coherence"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(1.5)
    tokens = result["regions"][0]["tokens"]
    assert len(tokens) == 2
    assert tokens[0]["source_duration_sec"] == pytest.approx(0.2)
    assert tokens[0]["target_duration_sec"] == pytest.approx(0.4)
    assert tokens[0]["source_f0"]["median_semitones"] == pytest.approx(
        12.0 * np.log2(100.0)
    )
    assert tokens[0]["target_energy_rms_db"] == pytest.approx(
        20.0 * np.log10(0.2)
    )


def test_compute_effect_supports_multiple_edit_regions():
    result = compute_effect_from_tracks(
        segments=_segments(multiple_regions=True),
        changed_segments=_changed_segments(multiple_regions=True),
        source_track=_track(0.4, 100.0),
        target_track=_track(0.8, 120.0),
        source_audio=np.full(6400, 0.1, dtype=np.float64),
        source_sample_rate=16000,
        target_audio=np.full(12800, 0.15, dtype=np.float64),
        target_sample_rate=16000,
    )

    assert result["status"] == "ok"
    assert result["num_regions"] == 2
    assert [region["segment_index"] for region in result["regions"]] == [0, 1]
    assert all(region["num_token_pairs"] == 1 for region in result["regions"])


def test_compute_effect_skips_zero_duration_token():
    result = compute_effect_from_tracks(
        segments=_segments(zero_duration=True),
        changed_segments=_changed_segments(),
        source_track=_track(0.4, 100.0),
        target_track=_track(0.8, 100.0),
        source_audio=np.full(6400, 0.1, dtype=np.float64),
        source_sample_rate=16000,
        target_audio=np.full(12800, 0.1, dtype=np.float64),
        target_sample_rate=16000,
    )

    assert result["num_token_pairs"] == 2
    assert result["num_duration_pairs"] == 1
    assert result["duration_median_abs_log2_ratio"] == pytest.approx(1.0)


def test_compute_effect_rejects_alignment_token_mismatch():
    with pytest.raises(ValueError, match="alignment token mismatch"):
        compute_effect_from_tracks(
            segments=_segments(mismatch=True),
            changed_segments=_changed_segments(),
            source_track=_track(0.4, 100.0),
            target_track=_track(0.8, 100.0),
            source_audio=np.full(6400, 0.1, dtype=np.float64),
            source_sample_rate=16000,
            target_audio=np.full(12800, 0.1, dtype=np.float64),
            target_sample_rate=16000,
        )


def test_compute_effect_allows_unvoiced_tokens_without_fallback():
    source_audio = np.zeros(6400, dtype=np.float64)
    target_audio = np.zeros(12800, dtype=np.float64)
    result = compute_effect_from_tracks(
        segments=_segments(),
        changed_segments=_changed_segments(),
        source_track=_track(0.4, 0.0),
        target_track=_track(0.8, 0.0),
        source_audio=source_audio,
        source_sample_rate=16000,
        target_audio=target_audio,
        target_sample_rate=16000,
    )

    assert result["status"] == "ok"
    assert result["num_pitch_pairs"] == 0
    assert result["pitch_level_median_abs_semitones"] is None
    assert result["score"] is None
    assert result["num_energy_pairs"] == 2


def test_compute_wrapper_rejects_missing_spans():
    segments = _segments()
    source_alignment = segments[0]["alignment"]
    target_alignment = segments[1]["alignment"]
    with pytest.raises(ValueError, match='edit_time_spans is missing'):
        result = compute_emotion_acoustic_effect(
            source_audio_bytes=_wav_bytes(np.zeros(6400, dtype=np.float32)),
            target_audio_bytes=_wav_bytes(np.zeros(12800, dtype=np.float32)),
            source_alignment=source_alignment,
            target_alignment=target_alignment,
            source_edit_spans=[],
            target_edit_spans=[],
            changed_segments=_changed_segments(),
        )


def test_extract_f0_track_directly_uses_parselmouth():
    sample_rate = 16000
    times = np.arange(sample_rate, dtype=np.float64) / sample_rate
    audio = 0.1 * np.sin(2.0 * np.pi * 200.0 * times)

    track = extract_f0_track(audio, sample_rate, AcousticEffectConfig())
    voiced = track.frequencies_hz[track.frequencies_hz > 0]

    assert voiced.size > 50
    assert np.median(voiced) == pytest.approx(200.0, abs=2.0)
