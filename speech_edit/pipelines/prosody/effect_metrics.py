"""Effect metrics for pitch/rate prosody edits.

This module defines the effect measurements used by the prosody pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from speech_edit.pipelines.prosody.tag_utils import (
    PITCH_TAG,
    RATE_TAG,
    parse_tags,
)


def safe_round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def mean_present(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not present:
        return None
    return sum(present) / len(present)


def expected_rate_duration(original_duration: float, factor: float) -> float:
    if factor <= 0:
        raise ValueError("Rate factor must be positive")
    return original_duration / factor


def duration_l1(original_duration: float, generated_duration: float, factor: float) -> float:
    return abs(generated_duration - expected_rate_duration(original_duration, factor))


def actual_pitch_shift_semitones(
    original_f0_hz: Optional[float],
    generated_f0_hz: Optional[float],
) -> Optional[float]:
    if original_f0_hz is None or generated_f0_hz is None:
        return None
    if original_f0_hz <= 0.0 or generated_f0_hz <= 0.0:
        return None
    return 12.0 * math.log2(generated_f0_hz / original_f0_hz)


def pitch_l1_semitones(
    original_f0_hz: Optional[float],
    generated_f0_hz: Optional[float],
    requested_semitones: float,
) -> Optional[float]:
    actual = actual_pitch_shift_semitones(original_f0_hz, generated_f0_hz)
    if actual is None:
        return None
    return abs(actual - requested_semitones)


def tagged_regions(tagged_text: str) -> List[Dict[str, Any]]:
    """Return ordered pitch/rate regions with plain-text character spans."""
    regions: List[Dict[str, Any]] = []
    pos = 0
    for segment in parse_tags(tagged_text):
        start = pos
        end = pos + len(segment.text)
        pos = end
        if not segment.is_tagged or not segment.text:
            continue
        if segment.tag_type == PITCH_TAG:
            regions.append(
                {
                    "edit_type": PITCH_TAG,
                    "text": segment.text,
                    "start_char": start,
                    "end_char": end,
                    "semitones": segment.semitones,
                    "factor": None,
                }
            )
        elif segment.tag_type == RATE_TAG:
            regions.append(
                {
                    "edit_type": RATE_TAG,
                    "text": segment.text,
                    "start_char": start,
                    "end_char": end,
                    "semitones": None,
                    "factor": segment.factor,
                }
            )
    return regions


def normalize_time_spans(raw_spans: Any) -> List[Dict[str, Any]]:
    """Normalize span records to ``{"start": float, "end": float, ...}``."""
    spans: List[Dict[str, Any]] = []
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes, bytearray)):
        return spans
    for item in raw_spans:
        if not isinstance(item, Mapping):
            continue
        span_value = item.get("span")
        if isinstance(span_value, Sequence) and not isinstance(span_value, (str, bytes, bytearray)) and len(span_value) == 2:
            start_raw, end_raw = span_value
        else:
            start_raw = item.get("start", item.get("start_sec"))
            end_raw = item.get("end", item.get("end_sec"))
        try:
            start = float(start_raw)
            end = float(end_raw)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "text": str(item.get("text") or item.get("words") or "").strip(),
            }
        )
    return spans


def spans_from_alignment_regions(regions: Any) -> List[Dict[str, Any]]:
    if not isinstance(regions, list):
        return []
    return normalize_time_spans(
        [
            {
                "start": region.get("start_sec"),
                "end": region.get("end_sec"),
                "text": region.get("text") or region.get("words"),
            }
            for region in regions
            if isinstance(region, Mapping)
        ]
    )


def spans_from_applied_edits(edits: Any, span_key: str = "target_span") -> List[Dict[str, Any]]:
    if not isinstance(edits, list):
        return []
    return normalize_time_spans(
        [
            {
                "span": edit.get(span_key),
                "text": edit.get("words"),
            }
            for edit in edits
            if isinstance(edit, Mapping)
        ]
    )


def _span_duration(span: Mapping[str, Any]) -> float:
    return max(0.0, float(span["end"]) - float(span["start"]))


def _f0_values_by_region(
    f0_client: Any,
    audio_bytes: bytes,
    spans: Sequence[Mapping[str, Any]],
    region_indices: Sequence[int],
    *,
    f0_min_hz: float,
    f0_max_hz: float,
) -> tuple[Dict[int, Optional[float]], Dict[int, Dict[str, Any]]]:
    if not region_indices:
        return {}, {}
    if f0_client is None:
        raise ValueError("F0 client is required for pitch evaluation")
    request_spans = [
        {
            "start": float(spans[idx]["start"]),
            "end": float(spans[idx]["end"]),
            "label": str(idx),
        }
        for idx in region_indices
    ]
    response = f0_client.analyze(
        audio_bytes,
        spans=request_spans,
        f0_min_hz=f0_min_hz,
        f0_max_hz=f0_max_hz,
    )
    values: Dict[int, Optional[float]] = {}
    details: Dict[int, Dict[str, Any]] = {}
    results = response["spans"]
    if len(results) != len(region_indices):
        raise ValueError("F0 response span count does not match request")
    for expected_idx, span_result in zip(region_indices, results):
        idx = int(span_result["label"])
        if idx != expected_idx:
            raise ValueError("F0 response labels do not match request")
        median = span_result["median_f0_hz"]
        values[idx] = float(median) if median is not None else None
        details[idx] = {
            "voiced_fraction": span_result["voiced_fraction"],
            "num_frames": span_result["num_frames"],
            "num_voiced_frames": span_result["num_voiced_frames"],
        }
    return values, details


def compute_region_effect_metrics(
    *,
    source_audio_bytes: bytes,
    target_audio_bytes: bytes,
    tagged_text: str,
    source_spans: Sequence[Mapping[str, Any]],
    target_spans: Sequence[Mapping[str, Any]],
    f0_client: Any = None,
    f0_min_hz: float = 50.0,
    f0_max_hz: float = 600.0,
) -> Dict[str, Any]:
    """Compute duration/pitch realization metrics for one ordered pair."""
    regions = tagged_regions(tagged_text)
    if not regions:
        return {
            "region_effects_status": "skipped_no_regions",
            "region_effects": [],
            "duration_l1_mean_sec": None,
            "pitch_l1_mean_semitones": None,
        }
    normalized_source_spans = normalize_time_spans(source_spans)
    normalized_target_spans = normalize_time_spans(target_spans)
    if len(normalized_source_spans) < len(regions) or len(normalized_target_spans) < len(regions):
        raise ValueError(
            "not enough edit spans for region effect metrics: "
            f"regions={len(regions)} source_spans={len(normalized_source_spans)} "
            f"target_spans={len(normalized_target_spans)}"
        )
    for idx, span in enumerate(normalized_source_spans[: len(regions)]):
        if _span_duration(span) <= 0.0:
            text = str(regions[idx].get("text") or "")
            raise ValueError(
                f"Empty source edit region for {text!r}: "
                f"{float(span['start']):.3f}s -> {float(span['end']):.3f}s"
            )

    pitch_indices = [idx for idx, region in enumerate(regions) if region["edit_type"] == PITCH_TAG]
    nonzero_target_pitch_indices = [
        idx for idx in pitch_indices if _span_duration(normalized_target_spans[idx]) > 0.0
    ]
    source_f0, source_f0_details = _f0_values_by_region(
        f0_client,
        source_audio_bytes,
        normalized_source_spans,
        pitch_indices,
        f0_min_hz=f0_min_hz,
        f0_max_hz=f0_max_hz,
    )
    target_f0, target_f0_details = _f0_values_by_region(
        f0_client,
        target_audio_bytes,
        normalized_target_spans,
        nonzero_target_pitch_indices,
        f0_min_hz=f0_min_hz,
        f0_max_hz=f0_max_hz,
    )
    for idx in pitch_indices:
        if _span_duration(normalized_target_spans[idx]) == 0.0:
            target_f0[idx] = source_f0.get(idx)

    region_results: List[Dict[str, Any]] = []
    duration_l1_values: List[Optional[float]] = []
    pitch_l1_values: List[Optional[float]] = []
    for idx, region in enumerate(regions):
        source_span = dict(normalized_source_spans[idx])
        target_span = dict(normalized_target_spans[idx])
        source_duration = _span_duration(source_span)
        target_duration = _span_duration(target_span)
        payload: Dict[str, Any] = {
            "region_index": idx,
            "edit_type": region["edit_type"],
            "text": region["text"],
            "source_span": {"start": source_span["start"], "end": source_span["end"]},
            "target_span": {"start": target_span["start"], "end": target_span["end"]},
            "source_duration_sec": source_duration,
            "target_duration_sec": target_duration,
        }
        if region["edit_type"] == RATE_TAG:
            factor = float(region["factor"])
            l1 = duration_l1(source_duration, target_duration, factor)
            payload.update(
                {
                    "factor": factor,
                    "expected_duration_sec": expected_rate_duration(source_duration, factor),
                    "duration_l1_sec": l1,
                }
            )
            duration_l1_values.append(l1)
        elif region["edit_type"] == PITCH_TAG:
            requested = float(region["semitones"])
            actual = actual_pitch_shift_semitones(source_f0.get(idx), target_f0.get(idx))
            l1 = pitch_l1_semitones(source_f0.get(idx), target_f0.get(idx), requested)
            f0_status = "ok"
            if source_f0.get(idx) is None or target_f0.get(idx) is None:
                f0_status = "unvoiced_or_unmeasurable"
            payload.update(
                {
                    "semitones": requested,
                    "source_median_f0_hz": source_f0.get(idx),
                    "target_median_f0_hz": target_f0.get(idx),
                    "actual_shift_semitones": actual,
                    "pitch_l1_semitones": l1,
                    "f0_status": f0_status,
                }
            )
            if idx in source_f0_details:
                payload["source_f0"] = source_f0_details[idx]
            if idx in target_f0_details:
                payload["target_f0"] = target_f0_details[idx]
            pitch_l1_values.append(l1)
        region_results.append(payload)

    return {
        "region_effects_status": "ok",
        "region_effects": region_results,
        "duration_l1_mean_sec": safe_round(mean_present(duration_l1_values), 4),
        "pitch_l1_mean_semitones": safe_round(mean_present(pitch_l1_values), 4),
    }
