"""Waveform splice helpers for masked edit preservation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


Span = Tuple[int, int]


def normalize_spans_sec(
    spans_sec: Sequence[Sequence[float]],
    sample_rate: int,
    num_samples: int,
) -> List[Span]:
    """Convert second spans to sorted, merged sample spans."""
    spans: List[Span] = []
    for span in spans_sec:
        if len(span) != 2:
            raise ValueError(f"Each span must be [start_sec, end_sec], got {span}")
        start_sec, end_sec = float(span[0]), float(span[1])
        if end_sec <= start_sec:
            continue
        start = max(0, int(start_sec * sample_rate))
        end = min(num_samples, int(end_sec * sample_rate))
        if start < end:
            spans.append((start, end))

    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def spans_sec_from_edit_mask(
    edit_mask: np.ndarray,
    hop_length: int,
    sample_rate: int,
) -> List[List[float]]:
    """Derive regenerate spans from an edit mask where False means regenerate."""
    mask = np.asarray(edit_mask, dtype=bool)
    if mask.ndim == 2:
        if mask.shape[0] != 1:
            raise ValueError("preserve waveform splice supports batch size 1 only")
        mask = mask[0]
    elif mask.ndim != 1:
        raise ValueError(f"edit_mask must be 1D or [1, n], got shape {list(mask.shape)}")

    spans: List[List[float]] = []
    start = None
    for idx, keep in enumerate(mask):
        if not keep and start is None:
            start = idx
        elif keep and start is not None:
            spans.append([start * hop_length / sample_rate, idx * hop_length / sample_rate])
            start = None
    if start is not None:
        spans.append([start * hop_length / sample_rate, len(mask) * hop_length / sample_rate])
    return spans


def expand_spans_sec(
    spans_sec: Sequence[Sequence[float]],
    expand_sec: float,
    total_duration_sec: float,
) -> List[List[float]]:
    """Expand spans in seconds and merge overlaps."""
    if expand_sec <= 0:
        return [[float(start), float(end)] for start, end in spans_sec if end > start]

    expanded = []
    for span in spans_sec:
        if len(span) != 2:
            raise ValueError(f"Each span must be [start_sec, end_sec], got {span}")
        start, end = float(span[0]), float(span[1])
        if end <= start:
            continue
        expanded.append([max(0.0, start - expand_sec), min(total_duration_sec, end + expand_sec)])

    if not expanded:
        return []

    expanded.sort()
    merged = [expanded[0]]
    for start, end in expanded[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def expand_false_regions(
    edit_mask: np.ndarray,
    expand_frames: int,
) -> np.ndarray:
    """Expand False regions in a bool edit mask by a frame margin."""
    mask = np.asarray(edit_mask, dtype=bool).copy()
    if expand_frames <= 0:
        return mask
    if mask.ndim == 1:
        mask_2d = mask.reshape(1, -1)
        squeeze = True
    elif mask.ndim == 2:
        mask_2d = mask
        squeeze = False
    else:
        raise ValueError(f"edit_mask must be 1D or 2D, got shape {list(mask.shape)}")

    expanded = mask_2d.copy()
    for batch_idx in range(mask_2d.shape[0]):
        false_idx = np.where(~mask_2d[batch_idx])[0]
        for idx in false_idx:
            start = max(0, idx - expand_frames)
            end = min(mask_2d.shape[1], idx + expand_frames + 1)
            expanded[batch_idx, start:end] = False
    return expanded[0] if squeeze else expanded


def apply_masked_edit_splice(
    source_waveform: np.ndarray,
    generated_waveform: np.ndarray,
    spans_sec: Sequence[Sequence[float]],
    sample_rate: int,
    *,
    source_crossfade_sec: float = 0.0,
    regen_crossfade_sec: float = 0.0,
) -> np.ndarray:
    """Splice generated regions into source with independent source/regen fades.

    `source_crossfade_sec` fades outside the core regenerate span, in preserved
    source material. `regen_crossfade_sec` fades inside the regenerate span.
    """
    source = np.asarray(source_waveform, dtype=np.float32).squeeze()
    generated = np.asarray(generated_waveform, dtype=np.float32).squeeze()
    if source.ndim != 1:
        raise ValueError(f"source_waveform must be mono, got shape {list(source.shape)}")
    if generated.ndim != 1:
        raise ValueError(f"generated_waveform must be mono, got shape {list(generated.shape)}")

    num_samples = len(source)
    if len(generated) < num_samples:
        generated = np.pad(generated, (0, num_samples - len(generated)))
    elif len(generated) > num_samples:
        generated = generated[:num_samples]

    spans = normalize_spans_sec(spans_sec, sample_rate, num_samples)
    if not spans:
        return source.copy()

    source_cf = max(0, int(source_crossfade_sec * sample_rate))
    regen_cf = max(0, int(regen_crossfade_sec * sample_rate))
    src_left = [min(source_cf, start) for start, _ in spans]
    src_right = [min(source_cf, num_samples - end) for _, end in spans]

    for idx in range(len(spans) - 1):
        gap = spans[idx + 1][0] - spans[idx][1]
        total = src_right[idx] + src_left[idx + 1]
        if gap >= 0 and total > gap:
            right = min(src_right[idx], gap // 2)
            left = min(src_left[idx + 1], gap - right)
            right = min(src_right[idx], gap - left)
            src_right[idx] = right
            src_left[idx + 1] = left

    output = source.copy()
    for idx, (start, end) in enumerate(spans):
        span_len = end - start
        if span_len <= 0:
            continue

        output[start:end] = generated[start:end]

        regen_cap = span_len // 4
        left_regen = min(regen_cf, regen_cap) if start > 0 else 0
        right_regen = min(regen_cf, regen_cap) if end < num_samples else 0

        if left_regen > 0:
            _blend_into_output(
                output,
                source,
                generated,
                start,
                start + left_regen,
                source_to_generated=True,
            )
        if right_regen > 0:
            _blend_into_output(
                output,
                source,
                generated,
                end - right_regen,
                end,
                source_to_generated=False,
            )

        if src_left[idx] > 0:
            _blend_into_output(
                output,
                source,
                generated,
                start - src_left[idx],
                start,
                source_to_generated=True,
            )
        if src_right[idx] > 0:
            _blend_into_output(
                output,
                source,
                generated,
                end,
                end + src_right[idx],
                source_to_generated=False,
            )

    return output.astype(np.float32, copy=False)


def _blend_into_output(
    output: np.ndarray,
    source: np.ndarray,
    generated: np.ndarray,
    start: int,
    end: int,
    *,
    source_to_generated: bool,
) -> None:
    length = end - start
    if length <= 0:
        return
    fade_in = np.linspace(0.0, 1.0, length, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, length, dtype=np.float32)
    if source_to_generated:
        output[start:end] = source[start:end] * fade_out + generated[start:end] * fade_in
    else:
        output[start:end] = generated[start:end] * fade_out + source[start:end] * fade_in
