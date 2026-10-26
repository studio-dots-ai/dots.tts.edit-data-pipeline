"""Pair-level acoustic-effect metrics for emotion/style edits.

The metric measures whether edited regions changed acoustically. It does not
classify emotion labels or judge whether a change moved in the requested
emotion direction.
"""

from __future__ import annotations

import io
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

METRIC_KEY = "emotion_acoustic_effect_v1"
METRIC_VERSION = "v1"


@dataclass(frozen=True)
class AcousticEffectConfig:
    frame_step_sec: float = 0.01
    f0_min_hz: float = 50.0
    f0_max_hz: float = 600.0
    min_token_duration_sec: float = 0.04
    min_voiced_frames: int = 3
    min_voiced_fraction: float = 0.25
    pitch_score_scale_semitones: float = 1.1
    energy_score_scale_db: float = 3.0
    duration_score_scale_log2: float = 0.15
    score_component_cap: float = 1.5


@dataclass(frozen=True)
class F0Track:
    times_sec: np.ndarray
    frequencies_hz: np.ndarray

    def __post_init__(self) -> None:
        if self.times_sec.ndim != 1 or self.frequencies_hz.ndim != 1:
            raise ValueError("F0Track arrays must be one-dimensional")
        if self.times_sec.shape != self.frequencies_hz.shape:
            raise ValueError("F0Track arrays must have equal shapes")


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.median(present)) if present else None


def _signed_coherence(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    magnitude = _median([abs(value) for value in present])
    if not present or magnitude is None or magnitude <= 0.0:
        return None
    return abs(float(np.median(present))) / magnitude


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float64",
        always_2d=False,
    )
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio, dtype=np.float64), int(sample_rate)


def extract_f0_track(
    audio: np.ndarray,
    sample_rate: int,
    config: AcousticEffectConfig,
) -> F0Track:
    """Extract one Praat F0 track for a complete utterance."""
    import parselmouth

    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(
        time_step=config.frame_step_sec,
        pitch_floor=config.f0_min_hz,
        pitch_ceiling=config.f0_max_hz,
    )
    return F0Track(
        times_sec=np.asarray(pitch.xs(), dtype=np.float64),
        frequencies_hz=np.asarray(
            pitch.selected_array["frequency"],
            dtype=np.float64,
        ),
    )


def _alignment_items(segment: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    alignment = segment.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("alignment is missing")
    status = str(alignment.get("status") or "success").strip().lower()
    if status != "success":
        raise ValueError("alignment is not successful")
    items = alignment.get("alignments")
    if not isinstance(items, list) or not items:
        raise ValueError("alignment has no token entries")
    if not all(isinstance(item, Mapping) for item in items):
        raise ValueError("alignment token entry is not a mapping")
    return items


def _edit_spans(segment: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    spans = segment.get("edit_time_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("edit_time_spans is missing")
    if not all(isinstance(item, Mapping) for item in spans):
        raise ValueError("edit_time_spans entry is not a mapping")
    return spans


def _parse_span(item: Mapping[str, Any]) -> tuple[float, float]:
    raw = item.get("span")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 2
    ):
        raise ValueError("invalid edit span")
    start = _optional_float(raw[0])
    end = _optional_float(raw[1])
    if start is None or end is None or end <= start:
        raise ValueError("invalid edit span bounds")
    return start, end


def _token_bounds(item: Mapping[str, Any]) -> tuple[float, float]:
    start = _optional_float(item.get("start"))
    end = _optional_float(item.get("end"))
    if start is None or end is None or end < start:
        raise ValueError("invalid alignment token bounds")
    return start, end


def _contains_midpoint(
    bounds: tuple[float, float],
    span: tuple[float, float],
    *,
    include_end: bool,
) -> bool:
    midpoint = (bounds[0] + bounds[1]) / 2.0
    if include_end:
        return span[0] <= midpoint <= span[1]
    return span[0] <= midpoint < span[1]


def _token_f0_stats(
    track: F0Track,
    bounds: tuple[float, float],
    config: AcousticEffectConfig,
) -> dict[str, float | int] | None:
    frame_mask = (track.times_sec >= bounds[0]) & (track.times_sec <= bounds[1])
    frames = track.frequencies_hz[frame_mask]
    voiced = frames[frames > 0.0]
    voiced_fraction = float(voiced.size / frames.size) if frames.size else 0.0
    if (
        voiced.size < config.min_voiced_frames
        or voiced_fraction < config.min_voiced_fraction
    ):
        return None
    semitones = 12.0 * np.log2(voiced)
    return {
        "num_frames": int(frames.size),
        "num_voiced_frames": int(voiced.size),
        "voiced_fraction": voiced_fraction,
        "median_semitones": float(np.median(semitones)),
        "p10_semitones": float(np.quantile(semitones, 0.10)),
        "p90_semitones": float(np.quantile(semitones, 0.90)),
        "std_semitones": float(np.std(semitones)),
    }


def _token_energy_rms_db(
    audio: np.ndarray,
    sample_rate: int,
    bounds: tuple[float, float],
) -> float | None:
    start = max(0, int(bounds[0] * sample_rate))
    end = min(int(audio.shape[0]), int(bounds[1] * sample_rate))
    clip = audio[start:end]
    if clip.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


def _effect_score(
    *,
    pitch_level: float | None,
    energy_level: float | None,
    duration: float | None,
    config: AcousticEffectConfig,
) -> float | None:
    if pitch_level is None or energy_level is None or duration is None:
        return None
    components = (
        pitch_level / config.pitch_score_scale_semitones,
        energy_level / config.energy_score_scale_db,
        duration / config.duration_score_scale_log2,
    )
    return float(
        np.mean(
            [
                min(float(value), config.score_component_cap)
                for value in components
            ]
        )
    )


def compute_effect_from_tracks(
    *,
    segments: Sequence[Mapping[str, Any]],
    changed_segments: Sequence[Mapping[str, Any]],
    source_track: F0Track,
    target_track: F0Track,
    source_audio: np.ndarray,
    source_sample_rate: int,
    target_audio: np.ndarray,
    target_sample_rate: int,
    config: AcousticEffectConfig | None = None,
) -> dict[str, Any]:
    """Compute paired token acoustic differences inside edited regions."""
    config = config or AcousticEffectConfig()
    by_side = {
        str(segment.get("chunk_id")): segment
        for segment in segments
        if isinstance(segment, Mapping)
    }
    if set(by_side) != {"src", "tgt"}:
        raise ValueError("segments must contain exactly src and tgt")
    source_segment = by_side["src"]
    target_segment = by_side["tgt"]
    source_tokens = _alignment_items(source_segment)
    target_tokens = _alignment_items(target_segment)
    if len(source_tokens) != len(target_tokens):
        raise ValueError("source and target alignment lengths differ")
    for index, (source_token, target_token) in enumerate(
        zip(source_tokens, target_tokens)
    ):
        if _normalize_text(source_token.get("char")) != _normalize_text(
            target_token.get("char")
        ):
            raise ValueError(f"alignment token mismatch at index {index}")

    source_spans = _edit_spans(source_segment)
    target_spans = _edit_spans(target_segment)
    if len(source_spans) != len(target_spans):
        raise ValueError("source and target edit span counts differ")
    if changed_segments and len(changed_segments) != len(source_spans):
        raise ValueError("changed_segments and edit span counts differ")

    regions: list[dict[str, Any]] = []
    all_pitch_signed: list[float] = []
    all_energy_signed: list[float] = []
    all_energy_abs: list[float] = []
    for region_index, (source_span_item, target_span_item) in enumerate(
        zip(source_spans, target_spans)
    ):
        segment_index = region_index
        if changed_segments:
            segment_index = int(
                changed_segments[region_index].get("segment_idx", region_index)
            )
            expected_text = _normalize_text(
                changed_segments[region_index].get("segment_text")
            )
            if expected_text and expected_text not in {
                _normalize_text(source_span_item.get("text")),
                _normalize_text(target_span_item.get("text")),
            }:
                raise ValueError("changed segment text does not match edit span text")

        source_span = _parse_span(source_span_item)
        target_span = _parse_span(target_span_item)
        include_end = region_index == len(source_spans) - 1
        token_indices = []
        for token_index, (source_token, target_token) in enumerate(
            zip(source_tokens, target_tokens)
        ):
            if _contains_midpoint(
                _token_bounds(source_token),
                source_span,
                include_end=include_end,
            ) and _contains_midpoint(
                _token_bounds(target_token),
                target_span,
                include_end=include_end,
            ):
                token_indices.append(token_index)
        if not token_indices:
            raise ValueError(f"region {region_index} has no paired alignment tokens")

        duration_abs: list[float] = []
        pitch_level_abs: list[float] = []
        pitch_level_signed: list[float] = []
        pitch_range_abs: list[float] = []
        energy_abs: list[float] = []
        energy_signed: list[float] = []
        pitch_pairs = 0
        token_metrics: list[dict[str, Any]] = []
        for token_index in token_indices:
            source_token = source_tokens[token_index]
            target_token = target_tokens[token_index]
            source_bounds = _token_bounds(source_token)
            target_bounds = _token_bounds(target_token)
            source_duration = source_bounds[1] - source_bounds[0]
            target_duration = target_bounds[1] - target_bounds[0]
            duration_signed = None
            if (
                source_duration >= config.min_token_duration_sec
                and target_duration >= config.min_token_duration_sec
            ):
                duration_signed = math.log2(target_duration / source_duration)
                duration_abs.append(abs(duration_signed))

            source_f0 = _token_f0_stats(source_track, source_bounds, config)
            target_f0 = _token_f0_stats(target_track, target_bounds, config)
            pitch_signed = None
            pitch_range_signed = None
            if source_f0 is not None and target_f0 is not None:
                pitch_pairs += 1
                pitch_signed = float(target_f0["median_semitones"]) - float(
                    source_f0["median_semitones"]
                )
                pitch_level_signed.append(pitch_signed)
                pitch_level_abs.append(abs(pitch_signed))
                source_range = float(source_f0["p90_semitones"]) - float(
                    source_f0["p10_semitones"]
                )
                target_range = float(target_f0["p90_semitones"]) - float(
                    target_f0["p10_semitones"]
                )
                pitch_range_signed = target_range - source_range
                pitch_range_abs.append(abs(pitch_range_signed))

            source_energy = _token_energy_rms_db(
                source_audio,
                source_sample_rate,
                source_bounds,
            )
            target_energy = _token_energy_rms_db(
                target_audio,
                target_sample_rate,
                target_bounds,
            )
            energy_delta = None
            if source_energy is not None and target_energy is not None:
                energy_delta = target_energy - source_energy
                energy_signed.append(energy_delta)
                energy_abs.append(abs(energy_delta))

            token_metrics.append(
                {
                    "token_index": token_index,
                    "token": str(source_token.get("char") or ""),
                    "source_bounds_sec": list(source_bounds),
                    "target_bounds_sec": list(target_bounds),
                    "source_duration_sec": source_duration,
                    "target_duration_sec": target_duration,
                    "duration_signed_log2_ratio": duration_signed,
                    "duration_abs_log2_ratio": (
                        None if duration_signed is None else abs(duration_signed)
                    ),
                    "source_f0": source_f0,
                    "target_f0": target_f0,
                    "pitch_level_signed_semitones": pitch_signed,
                    "pitch_level_abs_semitones": (
                        None if pitch_signed is None else abs(pitch_signed)
                    ),
                    "pitch_range_signed_delta_semitones": pitch_range_signed,
                    "pitch_range_abs_delta_semitones": (
                        None
                        if pitch_range_signed is None
                        else abs(pitch_range_signed)
                    ),
                    "source_energy_rms_db": source_energy,
                    "target_energy_rms_db": target_energy,
                    "energy_level_signed_db": energy_delta,
                    "energy_level_abs_db": (
                        None if energy_delta is None else abs(energy_delta)
                    ),
                }
            )

        regions.append(
            {
                "region_index": region_index,
                "segment_index": segment_index,
                "text": str(source_span_item.get("text") or ""),
                "num_token_pairs": len(token_indices),
                "num_duration_pairs": len(duration_abs),
                "num_pitch_pairs": pitch_pairs,
                "num_energy_pairs": len(energy_abs),
                "duration_median_abs_log2_ratio": _median(duration_abs),
                "pitch_level_median_abs_semitones": _median(pitch_level_abs),
                "pitch_range_median_abs_delta_semitones": _median(
                    pitch_range_abs
                ),
                "energy_level_median_abs_db": _median(energy_abs),
                "pitch_direction_coherence": _signed_coherence(
                    pitch_level_signed
                ),
                "energy_direction_coherence": _signed_coherence(energy_signed),
                "tokens": token_metrics,
            }
        )
        all_pitch_signed.extend(pitch_level_signed)
        all_energy_signed.extend(energy_signed)
        all_energy_abs.extend(energy_abs)

    if any(region["num_duration_pairs"] < 1 for region in regions):
        raise ValueError("one or more regions have no valid duration pairs")

    duration_median = _median(
        [region["duration_median_abs_log2_ratio"] for region in regions]
    )
    pitch_level_median = _median(
        [region["pitch_level_median_abs_semitones"] for region in regions]
    )
    pitch_range_median = _median(
        [region["pitch_range_median_abs_delta_semitones"] for region in regions]
    )
    energy_level_median = _median(all_energy_abs)
    pitch_coherence = _signed_coherence(all_pitch_signed)
    energy_coherence = _signed_coherence(all_energy_signed)
    coherence = (
        max(
            value
            for value in (pitch_coherence, energy_coherence)
            if value is not None
        )
        if pitch_coherence is not None or energy_coherence is not None
        else None
    )
    return {
        "version": METRIC_VERSION,
        "status": "ok",
        "alignment_level": source_segment["alignment"].get("level"),
        "num_regions": len(regions),
        "num_token_pairs": sum(region["num_token_pairs"] for region in regions),
        "num_duration_pairs": sum(
            region["num_duration_pairs"] for region in regions
        ),
        "num_pitch_pairs": sum(region["num_pitch_pairs"] for region in regions),
        "num_energy_pairs": sum(
            region["num_energy_pairs"] for region in regions
        ),
        "duration_median_abs_log2_ratio": duration_median,
        "pitch_level_median_abs_semitones": pitch_level_median,
        "pitch_range_median_abs_delta_semitones": pitch_range_median,
        "energy_level_median_abs_db": energy_level_median,
        "pitch_direction_coherence": pitch_coherence,
        "energy_direction_coherence": energy_coherence,
        "coherence": coherence,
        "score": _effect_score(
            pitch_level=pitch_level_median,
            energy_level=energy_level_median,
            duration=duration_median,
            config=config,
        ),
        "regions": regions,
    }


def compute_emotion_acoustic_effect(
    *,
    source_audio_bytes: bytes,
    target_audio_bytes: bytes,
    source_alignment: Mapping[str, Any],
    target_alignment: Mapping[str, Any],
    source_edit_spans: Sequence[Mapping[str, Any]],
    target_edit_spans: Sequence[Mapping[str, Any]],
    changed_segments: Sequence[Mapping[str, Any]],
    config: AcousticEffectConfig | None = None,
) -> dict[str, Any]:
    """Compute the v1 metric payload for one source-target pair."""
    config = config or AcousticEffectConfig()
    source_audio, source_sample_rate = decode_audio_bytes(source_audio_bytes)
    target_audio, target_sample_rate = decode_audio_bytes(target_audio_bytes)
    source_track = extract_f0_track(
        source_audio,
        source_sample_rate,
        config,
    )
    target_track = extract_f0_track(
        target_audio,
        target_sample_rate,
        config,
    )
    return compute_effect_from_tracks(
        segments=[
            {
                "chunk_id": "src",
                "edit_time_spans": list(source_edit_spans),
                "alignment": dict(source_alignment),
            },
            {
                "chunk_id": "tgt",
                "edit_time_spans": list(target_edit_spans),
                "alignment": dict(target_alignment),
            },
        ],
        changed_segments=changed_segments,
        source_track=source_track,
        target_track=target_track,
        source_audio=source_audio,
        source_sample_rate=source_sample_rate,
        target_audio=target_audio,
        target_sample_rate=target_sample_rate,
        config=config,
    )
