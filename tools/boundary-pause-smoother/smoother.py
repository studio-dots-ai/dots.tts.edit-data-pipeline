"""Boundary Pause Smoother — Core Module.

Boundary smoothing is expressed in terms of pause intervals rather than a
punctuation/remove-pause policy split.

For each smoothing point, the service:
1. Resolves the native pause interval from either an explicit
   ``pause_interval_sec`` or from aligner/VAD around ``position_sec``.
2. Clamps or expands that native pause to the target duration implied by
   ``min_pause_sec`` and ``max_pause_sec``.
3. Rebuilds the audio by shrinking or expanding pause content around the
   boundary anchor so the adjusted pause reaches the target duration.
4. Regenerates either a single continuous masked region covering
   ``adjusted_pause_interval_sec ± regen_margin_sec`` via F5-TTS, or, for
   ``audio_insert`` mode, two disjoint transition spans on the left and right of
   the insert interval.

This module can be used as a Python function or via the companion FastAPI
service.
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import requests
import soundfile as sf

logger = logging.getLogger(__name__)

# F5-TTS constants
F5_SAMPLE_RATE = 24000
F5_HOP_LENGTH = 256
MAX_F5_EDIT_SECONDS = 30.0


@dataclass
class SmoothingPoint:
    """Specification for a single smoothing point."""

    mode: Literal["smooth_only", "adjust"] = "smooth_only"
    """Processing mode.

    ``smooth_only`` skips aligner, VAD, and pause adjustment, and directly
    regenerates a seam-centered span around ``position_sec``.

    ``adjust`` resolves or consumes a native pause interval, adjusts its
    duration, and then regenerates the adjusted pause region.
    """

    position_sec: Optional[float] = None
    """Boundary anchor in seconds.

    Optional when ``pause_interval_sec`` is provided. If ``pause_interval_sec`` is
    present, ``position_sec`` is normalized to the midpoint of that interval.
    """

    pause_interval_sec: Optional[Tuple[float, float]] = None
    """Optional native pause interval ``(start_sec, end_sec)``.

    When provided, the service skips aligner/VAD resolution for this point and
    treats the midpoint of the interval as the logical boundary anchor.
    """

    max_pause_sec: Optional[float] = None
    """Maximum allowed pause duration.

    If provided, adjusted pause durations longer than this are clamped down to
    ``max_pause_sec``.
    """

    min_pause_sec: Optional[float] = None
    """Minimum required pause duration.

    If provided, adjusted pause durations shorter than this are expanded to
    ``min_pause_sec`` by inserting silence.
    """

    regen_margin_sec: float = 0.15
    """Extra masked context on each side of the adjusted pause interval.

    The final masked interval is
    ``[adjusted_pause_start - regen_margin_sec, adjusted_pause_end + regen_margin_sec]``.
    """

    audio_insert: Optional[np.ndarray] = None
    """Optional mono float32 audio to overwrite the protected interval after F5."""

    audio_insert_intrusion_ratio: float = 0.0
    """Ratio of insert duration exposed to F5 on each boundary side."""


@dataclass
class SmoothingResult:
    """Result from the smoothing operation."""

    audio: np.ndarray
    """Smoothed audio waveform (float32, mono)."""

    sample_rate: int
    """Sample rate of the output audio."""

    points_processed: int
    """Number of smoothing points that were actually processed."""

    details: List[Dict] = field(default_factory=list)
    """Per-point processing details (pause detected, trimmed, regen span, etc.)."""

    debug_dir: Optional[str] = None
    """Resolved debug directory when debug mode is enabled."""


@dataclass
class _NormalizedPoint:
    orig_idx: int
    mode: Literal["smooth_only", "adjust"]
    position_sec: float
    pause_interval_sec: Optional[Tuple[float, float]]
    min_pause_sec: Optional[float]
    max_pause_sec: Optional[float]
    regen_margin_sec: float
    audio_insert: Optional[np.ndarray] = None
    audio_insert_intrusion_ratio: float = 0.0


@dataclass
class _PausePlan:
    orig_idx: int
    position_sec: float
    native_pause: Dict
    native_start_idx: int
    native_position_idx: int
    native_end_idx: int
    native_left_samples: int
    native_right_samples: int
    native_duration_samples: int
    target_duration_samples: int
    target_left_samples: int
    target_right_samples: int
    regen_margin_samples: int
    adjusted_start_idx: int = 0
    adjusted_end_idx: int = 0
    adjusted_position_idx: int = 0
    audio_insert: Optional[np.ndarray] = None
    insert_start_idx: Optional[int] = None
    insert_end_idx: Optional[int] = None
    left_regen_start_idx: Optional[int] = None
    left_regen_end_idx: Optional[int] = None
    right_regen_start_idx: Optional[int] = None
    right_regen_end_idx: Optional[int] = None
    intrusion_samples: int = 0
    left_intrusion_end_idx: Optional[int] = None
    right_intrusion_start_idx: Optional[int] = None


@dataclass
class _RegenerationResult:
    spliced_audio: np.ndarray
    f5_output_audio: np.ndarray
    f5_output_sample_rate: int = F5_SAMPLE_RATE


def _debug_save_wav(debug_dir: str, filename: str, audio: np.ndarray, sr: int):
    """Save audio to a WAV file in the debug directory."""
    import os

    path = os.path.join(debug_dir, filename)
    sf.write(path, audio, sr)
    logger.debug(f"Debug: saved {path} ({len(audio) / sr:.2f}s)")


def _overlay_debug_marker(
    audio: np.ndarray,
    *,
    center_idx: Optional[int],
    sample_rate: int,
    duration_sec: float = 0.1,
    frequency_hz: float = 1000.0,
    amplitude: float = 0.35,
) -> None:
    """Overlay a short marker beep in place without changing audio length."""
    if center_idx is None or center_idx < 0 or center_idx >= len(audio):
        return

    half_len = max(1, int(round(duration_sec * sample_rate / 2.0)))
    start_idx = max(0, center_idx - half_len)
    end_idx = min(len(audio), center_idx + half_len)
    if end_idx <= start_idx:
        return

    marker_len = end_idx - start_idx
    marker = np.sin(
        2.0 * np.pi * frequency_hz * np.arange(marker_len, dtype=np.float32) / sample_rate
    ).astype(np.float32)
    fade_len = min(marker_len // 2, max(1, int(round(sample_rate * 0.01))))
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        marker[:fade_len] *= fade_in
        marker[-fade_len:] *= fade_out

    audio[start_idx:end_idx] = np.clip(
        audio[start_idx:end_idx] + amplitude * marker,
        -1.0,
        1.0,
    )


def _build_debug_mask_visualization(
    adjusted_audio: np.ndarray,
    *,
    merged_spans: List[List[float]],
    plans: List[_PausePlan],
    sample_rate: int,
) -> np.ndarray:
    """Show full masked regions and mark audio-insert boundaries distinctly."""
    mask_audio = adjusted_audio.copy()
    for s, e in merged_spans:
        ms = _sec_to_sample(s, sample_rate, upper_bound=len(mask_audio))
        me = _sec_to_sample(e, sample_rate, upper_bound=len(mask_audio))
        mask_audio[ms:me] = 0.0

    for plan in plans:
        if plan.audio_insert is None:
            continue
        _overlay_debug_marker(
            mask_audio,
            center_idx=plan.left_regen_end_idx,
            sample_rate=sample_rate,
            frequency_hz=880.0,
        )
        _overlay_debug_marker(
            mask_audio,
            center_idx=plan.left_intrusion_end_idx,
            sample_rate=sample_rate,
            frequency_hz=1040.0,
        )
        _overlay_debug_marker(
            mask_audio,
            center_idx=plan.right_regen_start_idx,
            sample_rate=sample_rate,
            frequency_hz=1320.0,
        )
        _overlay_debug_marker(
            mask_audio,
            center_idx=plan.right_intrusion_start_idx,
            sample_rate=sample_rate,
            frequency_hz=1560.0,
        )

    return mask_audio


def _round_sec(value: float) -> float:
    return round(float(value), 4)


def _sec_to_sample(
    sec: float,
    sample_rate: int,
    *,
    upper_bound: Optional[int] = None,
) -> int:
    sample = int(round(float(sec) * sample_rate))
    sample = max(sample, 0)
    if upper_bound is not None:
        sample = min(sample, upper_bound)
    return sample


def _sample_to_sec(sample_idx: int, sample_rate: int) -> float:
    return sample_idx / sample_rate


def _normalize_points(
    points: List[SmoothingPoint],
    total_dur: float,
) -> List[_NormalizedPoint]:
    normalized = []

    for idx, pt in enumerate(points):
        mode = pt.mode
        pause_interval = None
        min_pause_sec = None
        max_pause_sec = None
        if mode == "smooth_only":
            if pt.position_sec is None:
                raise ValueError(
                    f"points[{idx}] with mode='smooth_only' must provide position_sec"
                )
            if pt.pause_interval_sec is not None:
                raise ValueError(
                    f"points[{idx}] with mode='smooth_only' must not provide pause_interval_sec"
                )
            if pt.min_pause_sec is not None or pt.max_pause_sec is not None:
                raise ValueError(
                    f"points[{idx}] with mode='smooth_only' must not provide min_pause_sec or max_pause_sec"
                )
            if pt.audio_insert is not None:
                raise ValueError(
                    f"points[{idx}] with mode='smooth_only' must not provide audio_insert"
                )
            position_sec = float(pt.position_sec)
            if not np.isfinite(position_sec):
                raise ValueError(f"points[{idx}].position_sec must be finite")
            if position_sec < 0.0 or position_sec > total_dur:
                raise ValueError(
                    f"points[{idx}].position_sec must stay within audio duration"
                )
            min_pause_sec = None
            max_pause_sec = None
        elif pt.pause_interval_sec is not None:
            if len(pt.pause_interval_sec) != 2:
                raise ValueError(
                    f"points[{idx}].pause_interval_sec must contain exactly two values"
                )
            pause_start = float(pt.pause_interval_sec[0])
            pause_end = float(pt.pause_interval_sec[1])
            if not np.isfinite(pause_start) or not np.isfinite(pause_end):
                raise ValueError(
                    f"points[{idx}].pause_interval_sec must contain finite values"
                )
            if pause_end < pause_start:
                raise ValueError(
                    f"points[{idx}].pause_interval_sec must satisfy start <= end"
                )
            if pause_start < 0.0 or pause_end > total_dur:
                raise ValueError(
                    f"points[{idx}].pause_interval_sec must stay within audio duration"
                )
            pause_interval = (pause_start, pause_end)
            position_sec = (pause_start + pause_end) / 2.0
            min_pause_sec = None
            if pt.min_pause_sec is not None:
                min_pause_sec = float(pt.min_pause_sec)
                if not np.isfinite(min_pause_sec) or min_pause_sec < 0.0:
                    raise ValueError(
                        f"points[{idx}].min_pause_sec must be a finite >= 0"
                    )

            max_pause_sec = None
            if pt.max_pause_sec is not None:
                max_pause_sec = float(pt.max_pause_sec)
                if not np.isfinite(max_pause_sec) or max_pause_sec < 0.0:
                    raise ValueError(
                        f"points[{idx}].max_pause_sec must be a finite >= 0"
                    )
        elif pt.position_sec is not None:
            position_sec = float(pt.position_sec)
            if not np.isfinite(position_sec):
                raise ValueError(f"points[{idx}].position_sec must be finite")
            if position_sec < 0.0 or position_sec > total_dur:
                raise ValueError(
                    f"points[{idx}].position_sec must stay within audio duration"
                )
            if pt.min_pause_sec is not None:
                min_pause_sec = float(pt.min_pause_sec)
                if not np.isfinite(min_pause_sec) or min_pause_sec < 0.0:
                    raise ValueError(
                        f"points[{idx}].min_pause_sec must be a finite >= 0"
                    )

            if pt.max_pause_sec is not None:
                max_pause_sec = float(pt.max_pause_sec)
                if not np.isfinite(max_pause_sec) or max_pause_sec < 0.0:
                    raise ValueError(
                        f"points[{idx}].max_pause_sec must be a finite >= 0"
                    )
        else:
            raise ValueError(
                f"points[{idx}] must provide either position_sec or pause_interval_sec"
            )

        if (
            pt.audio_insert is None
            and min_pause_sec is not None
            and max_pause_sec is not None
            and min_pause_sec > max_pause_sec
        ):
            raise ValueError(
                f"points[{idx}] must satisfy min_pause_sec <= max_pause_sec"
            )

        regen_margin_sec = float(pt.regen_margin_sec)
        if not np.isfinite(regen_margin_sec) or regen_margin_sec < 0.0:
            raise ValueError(f"points[{idx}].regen_margin_sec must be a finite >= 0")

        audio_insert = None
        if pt.audio_insert is not None:
            if mode != "adjust":
                raise ValueError(
                    f"points[{idx}].audio_insert is only supported in mode='adjust'"
                )
            audio_insert = np.asarray(pt.audio_insert, dtype=np.float32).squeeze()
            if audio_insert.ndim != 1:
                raise ValueError(f"points[{idx}].audio_insert must be mono audio")
            if len(audio_insert) == 0:
                raise ValueError(f"points[{idx}].audio_insert must not be empty")

        audio_insert_intrusion_ratio = float(pt.audio_insert_intrusion_ratio)
        if not np.isfinite(audio_insert_intrusion_ratio):
            raise ValueError(
                f"points[{idx}].audio_insert_intrusion_ratio must be finite"
            )
        if audio_insert_intrusion_ratio < 0.0 or audio_insert_intrusion_ratio > 0.5:
            raise ValueError(
                f"points[{idx}].audio_insert_intrusion_ratio must be between 0 and 0.5"
            )
        if audio_insert is None and audio_insert_intrusion_ratio != 0.0:
            raise ValueError(
                f"points[{idx}].audio_insert_intrusion_ratio requires audio_insert"
            )

        normalized.append(
            _NormalizedPoint(
                orig_idx=idx,
                mode=mode,
                position_sec=position_sec,
                pause_interval_sec=pause_interval,
                min_pause_sec=min_pause_sec,
                max_pause_sec=max_pause_sec,
                regen_margin_sec=regen_margin_sec,
                audio_insert=audio_insert,
                audio_insert_intrusion_ratio=audio_insert_intrusion_ratio,
            )
        )

    return sorted(normalized, key=lambda item: (item.position_sec, item.orig_idx))


def _build_smooth_only_details_and_spans(
    normalized_points: List[_NormalizedPoint],
    *,
    sample_rate: int,
    total_dur: float,
) -> Tuple[List[Dict], List[List[float]], int]:
    details = []
    regen_spans = []
    points_processed = 0

    for point in normalized_points:
        regen_start = max(0.0, point.position_sec - point.regen_margin_sec)
        regen_end = min(total_dur, point.position_sec + point.regen_margin_sec)
        entry = {
            "orig_idx": point.orig_idx,
            "mode": point.mode,
            "position_sec": _round_sec(point.position_sec),
            "position_new_sec": _round_sec(point.position_sec),
            "action": "smooth_only",
            "regen_margin_sec": _round_sec(point.regen_margin_sec),
        }

        if regen_end - regen_start >= 0.02:
            regen_spans.append([regen_start, regen_end])
            entry["regen"] = {
                "start": _round_sec(regen_start),
                "end": _round_sec(regen_end),
                "duration": _round_sec(regen_end - regen_start),
            }
            points_processed += 1
        else:
            entry["regen"] = "skipped_too_short"

        details.append(entry)

    return details, regen_spans, points_processed


def _find_pause_from_vad(
    position_sec: float,
    vad_silences: List[Dict],
    *,
    tolerance_sec: float = 0.05,
) -> Optional[Dict]:
    """Find the VAD silence interval nearest to ``position_sec``."""
    best = None
    best_distance = float("inf")

    for silence in vad_silences:
        start = silence["start"]
        end = silence["end"]
        if start <= position_sec <= end:
            distance = 0.0
        else:
            distance = min(abs(position_sec - start), abs(position_sec - end))

        if distance < best_distance:
            best_distance = distance
            best = silence

    if best is None or best_distance > tolerance_sec:
        return None

    return {
        "pause_start": _round_sec(best["start"]),
        "pause_end": _round_sec(best["end"]),
        "pause_duration": _round_sec(best["end"] - best["start"]),
        "method": "vad",
    }


def _resolve_native_pause(
    point: _NormalizedPoint,
    *,
    alignments: List[Dict],
    vad_silences: List[Dict],
    total_dur: float,
) -> Dict:
    if point.pause_interval_sec is not None:
        pause_start, pause_end = point.pause_interval_sec
        return {
            "pause_start": _round_sec(pause_start),
            "pause_end": _round_sec(pause_end),
            "pause_duration": _round_sec(pause_end - pause_start),
            "method": "provided",
        }

    pause = None
    if alignments:
        pause = _find_pause_from_alignment(alignments, point.position_sec, total_dur)
        if pause["pause_duration"] > 0.01 and vad_silences:
            refined = _refine_pause_with_vad(pause, vad_silences)
            if refined is not None:
                pause = refined
        elif vad_silences:
            vad_pause = _find_pause_from_vad(point.position_sec, vad_silences)
            if vad_pause is not None:
                pause = vad_pause
    elif vad_silences:
        pause = _find_pause_from_vad(point.position_sec, vad_silences)

    if pause is None:
        pause = {
            "pause_start": _round_sec(point.position_sec),
            "pause_end": _round_sec(point.position_sec),
            "pause_duration": 0.0,
            "method": "insertion_point",
        }

    return pause


def _plan_pause_adjustments(
    normalized_points: List[_NormalizedPoint],
    *,
    sample_rate: int,
    audio_num_samples: int,
    total_dur: float,
    alignments: List[Dict],
    vad_silences: List[Dict],
) -> List[_PausePlan]:
    plans: List[_PausePlan] = []
    prev_end_idx = 0

    for point in normalized_points:
        native_pause = _resolve_native_pause(
            point,
            alignments=alignments,
            vad_silences=vad_silences,
            total_dur=total_dur,
        )
        native_start_idx = _sec_to_sample(
            native_pause["pause_start"], sample_rate, upper_bound=audio_num_samples
        )
        native_end_idx = _sec_to_sample(
            native_pause["pause_end"], sample_rate, upper_bound=audio_num_samples
        )
        native_end_idx = max(native_end_idx, native_start_idx)
        native_position_idx = _sec_to_sample(
            point.position_sec,
            sample_rate,
            upper_bound=audio_num_samples,
        )
        native_position_idx = min(max(native_position_idx, native_start_idx), native_end_idx)

        if native_start_idx < prev_end_idx:
            raise ValueError(
                "Resolved native pause intervals overlap; provide non-overlapping "
                "pause_interval_sec values or separate the boundary positions"
            )

        native_left_samples = native_position_idx - native_start_idx
        native_right_samples = native_end_idx - native_position_idx
        native_duration_samples = native_end_idx - native_start_idx
        target_duration_samples = native_duration_samples

        if point.audio_insert is not None:
            target_duration_samples = len(point.audio_insert)
        elif point.max_pause_sec is not None:
            target_duration_samples = min(
                target_duration_samples,
                _sec_to_sample(point.max_pause_sec, sample_rate),
            )
        if point.audio_insert is None and point.min_pause_sec is not None:
            target_duration_samples = max(
                target_duration_samples,
                _sec_to_sample(point.min_pause_sec, sample_rate),
            )

        if target_duration_samples >= native_duration_samples:
            add_total = target_duration_samples - native_duration_samples
            add_left = add_total // 2
            add_right = add_total - add_left
            target_left_samples = native_left_samples + add_left
            target_right_samples = native_right_samples + add_right
        else:
            remove_total = native_duration_samples - target_duration_samples
            remove_left = min(native_left_samples, remove_total // 2)
            remove_right = min(native_right_samples, remove_total - remove_left)
            remaining_remove = remove_total - remove_left - remove_right
            if remaining_remove > 0:
                extra_left = min(native_left_samples - remove_left, remaining_remove)
                remove_left += extra_left
                remaining_remove -= extra_left
            if remaining_remove > 0:
                extra_right = min(native_right_samples - remove_right, remaining_remove)
                remove_right += extra_right
                remaining_remove -= extra_right
            target_left_samples = native_left_samples - remove_left
            target_right_samples = native_right_samples - remove_right
        regen_margin_samples = _sec_to_sample(point.regen_margin_sec, sample_rate)
        intrusion_samples = 0
        if point.audio_insert is not None:
            intrusion_samples = int(
                round(target_duration_samples * point.audio_insert_intrusion_ratio)
            )
            intrusion_samples = min(
                max(intrusion_samples, 0),
                target_duration_samples // 2,
            )

        plans.append(
            _PausePlan(
                orig_idx=point.orig_idx,
                position_sec=point.position_sec,
                native_pause=native_pause,
                native_start_idx=native_start_idx,
                native_position_idx=native_position_idx,
                native_end_idx=native_end_idx,
                native_left_samples=native_left_samples,
                native_right_samples=native_right_samples,
                native_duration_samples=native_duration_samples,
                target_duration_samples=target_duration_samples,
                target_left_samples=target_left_samples,
                target_right_samples=target_right_samples,
                regen_margin_samples=regen_margin_samples,
                audio_insert=point.audio_insert,
                intrusion_samples=intrusion_samples,
            )
        )
        prev_end_idx = native_end_idx

    return plans


def _build_pause_adjusted_audio(
    audio: np.ndarray,
    plans: List[_PausePlan],
) -> np.ndarray:
    if not plans:
        return audio.copy()

    chunks = []
    cursor = 0
    output_cursor = 0

    for plan in plans:
        if plan.native_start_idx < cursor:
            raise ValueError("Native pause intervals must be non-overlapping")

        prefix = audio[cursor : plan.native_start_idx]
        if len(prefix) > 0:
            chunks.append(prefix)
            output_cursor += len(prefix)

        native_left = audio[plan.native_start_idx : plan.native_position_idx]
        native_right = audio[plan.native_position_idx : plan.native_end_idx]

        if plan.target_left_samples <= len(native_left):
            left_adjusted = native_left[: plan.target_left_samples].copy()
        else:
            extra_left = plan.target_left_samples - len(native_left)
            left_adjusted = np.concatenate(
                [native_left, np.zeros(extra_left, dtype=np.float32)]
            )

        if plan.target_right_samples <= len(native_right):
            right_adjusted = native_right[
                len(native_right) - plan.target_right_samples :
            ].copy()
        else:
            extra_right = plan.target_right_samples - len(native_right)
            right_adjusted = np.concatenate(
                [np.zeros(extra_right, dtype=np.float32), native_right]
            )

        plan.adjusted_start_idx = output_cursor
        plan.adjusted_position_idx = output_cursor + len(left_adjusted)
        plan.adjusted_end_idx = plan.adjusted_position_idx + len(right_adjusted)

        if len(left_adjusted) > 0:
            chunks.append(left_adjusted)
        if len(right_adjusted) > 0:
            chunks.append(right_adjusted)
        if len(left_adjusted) > 0 or len(right_adjusted) > 0:
            output_cursor = plan.adjusted_end_idx

        cursor = plan.native_end_idx

    suffix = audio[cursor:]
    if len(suffix) > 0:
        chunks.append(suffix)

    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].copy()
    return np.concatenate(chunks)


def _append_regen_span(
    regen_spans: List[List[float]],
    *,
    start_idx: int,
    end_idx: int,
    sample_rate: int,
) -> Optional[Dict]:
    if end_idx <= start_idx:
        return None

    start_sec = _sample_to_sec(start_idx, sample_rate)
    end_sec = _sample_to_sec(end_idx, sample_rate)
    regen_spans.append([start_sec, end_sec])
    return {
        "start": _round_sec(start_sec),
        "end": _round_sec(end_sec),
        "duration": _round_sec(end_sec - start_sec),
    }


def _subtract_protected_intervals(
    regen_spans: List[List[float]],
    protected_intervals: List[List[float]],
) -> List[List[float]]:
    """Remove unmasked/protected intervals from candidate F5 edit spans."""
    if not regen_spans or not protected_intervals:
        return regen_spans

    protected = sorted(protected_intervals)
    result = []
    for span_start, span_end in regen_spans:
        pieces = [[span_start, span_end]]
        for protect_start, protect_end in protected:
            if protect_end <= span_start or protect_start >= span_end:
                continue
            next_pieces = []
            for piece_start, piece_end in pieces:
                if protect_end <= piece_start or protect_start >= piece_end:
                    next_pieces.append([piece_start, piece_end])
                    continue
                if protect_start > piece_start:
                    next_pieces.append([piece_start, min(protect_start, piece_end)])
                if protect_end < piece_end:
                    next_pieces.append([max(protect_end, piece_start), piece_end])
            pieces = next_pieces
            if not pieces:
                break
        result.extend(piece for piece in pieces if piece[1] > piece[0])

    return result


def _apply_audio_inserts(audio: np.ndarray, plans: List[_PausePlan]) -> np.ndarray:
    if not any(plan.audio_insert is not None for plan in plans):
        return audio

    output = audio.copy()
    for plan in plans:
        if plan.audio_insert is None:
            continue
        if plan.insert_start_idx is None or plan.insert_end_idx is None:
            raise ValueError("audio_insert interval was not computed")
        expected = plan.insert_end_idx - plan.insert_start_idx
        if expected != len(plan.audio_insert):
            raise ValueError("audio_insert length must match computed insert interval")
        output[plan.insert_start_idx : plan.insert_end_idx] = plan.audio_insert
    return output


def _build_f5_input_audio(
    audio: np.ndarray,
    plans: List[_PausePlan],
) -> np.ndarray:
    if not any(plan.audio_insert is not None for plan in plans):
        return audio

    output = audio.copy()
    for plan in plans:
        if plan.audio_insert is None:
            continue
        if plan.insert_start_idx is None or plan.insert_end_idx is None:
            raise ValueError("audio_insert F5 input interval was not computed")
        if plan.insert_end_idx > plan.insert_start_idx:
            # Only the forced-silence insert interval is zeroed. Neighboring
            # regen spans stay intact in the conditioning audio and are edited
            # solely via the F5 mask.
            output[plan.insert_start_idx : plan.insert_end_idx] = 0.0
    return output


def _build_details_and_regen_spans(
    adjusted_audio: np.ndarray,
    plans: List[_PausePlan],
    *,
    sample_rate: int,
) -> Tuple[List[Dict], List[List[float]], List[List[float]], int]:
    details = []
    regen_spans = []
    protected_intervals = []
    points_processed = 0
    audio_num_samples = len(adjusted_audio)

    for plan in plans:
        native_duration_sec = _sample_to_sec(plan.native_duration_samples, sample_rate)
        target_duration_sec = _sample_to_sec(plan.target_duration_samples, sample_rate)
        delta_sec = target_duration_sec - native_duration_sec
        if delta_sec > 1e-9:
            operation = "expanded"
        elif delta_sec < -1e-9:
            operation = "shrunk"
        else:
            operation = "kept"

        adjusted_start_sec = _sample_to_sec(plan.adjusted_start_idx, sample_rate)
        adjusted_end_sec = _sample_to_sec(plan.adjusted_end_idx, sample_rate)
        adjusted_position_sec = _sample_to_sec(plan.adjusted_position_idx, sample_rate)

        regen_start_idx = max(0, plan.adjusted_start_idx - plan.regen_margin_samples)
        regen_end_idx = min(
            audio_num_samples,
            plan.adjusted_end_idx + plan.regen_margin_samples,
        )
        insert_start_idx = None
        insert_end_idx = None
        left_regen_start_idx = None
        left_regen_end_idx = None
        right_regen_start_idx = None
        right_regen_end_idx = None
        left_intrusion_end_idx = None
        right_intrusion_start_idx = None
        if plan.audio_insert is not None and plan.target_duration_samples > 0:
            insert_start_idx = plan.adjusted_start_idx
            insert_end_idx = plan.adjusted_end_idx
            left_regen_start_idx = max(0, insert_start_idx - plan.regen_margin_samples)
            left_regen_end_idx = insert_start_idx
            right_regen_start_idx = insert_end_idx
            right_regen_end_idx = min(
                audio_num_samples,
                insert_end_idx + plan.regen_margin_samples,
            )
            left_intrusion_end_idx = min(
                insert_end_idx,
                insert_start_idx + plan.intrusion_samples,
            )
            right_intrusion_start_idx = max(
                insert_start_idx,
                insert_end_idx - plan.intrusion_samples,
            )
            plan.insert_start_idx = insert_start_idx
            plan.insert_end_idx = insert_end_idx
            plan.left_regen_start_idx = left_regen_start_idx
            plan.left_regen_end_idx = left_regen_end_idx
            plan.right_regen_start_idx = right_regen_start_idx
            plan.right_regen_end_idx = right_regen_end_idx
            plan.left_intrusion_end_idx = left_intrusion_end_idx
            plan.right_intrusion_start_idx = right_intrusion_start_idx

        entry = {
            "orig_idx": plan.orig_idx,
            "mode": "adjust",
            "position_sec": _round_sec(plan.position_sec),
            "position_new_sec": _round_sec(adjusted_position_sec),
            "native_pause": {
                "start": _round_sec(plan.native_pause["pause_start"]),
                "end": _round_sec(plan.native_pause["pause_end"]),
                "duration": _round_sec(native_duration_sec),
                "left_duration": _round_sec(
                    _sample_to_sec(plan.native_left_samples, sample_rate)
                ),
                "right_duration": _round_sec(
                    _sample_to_sec(plan.native_right_samples, sample_rate)
                ),
                "method": plan.native_pause["method"],
            },
            "target_pause": {
                "duration": _round_sec(target_duration_sec),
                "delta": _round_sec(delta_sec),
                "operation": operation,
                "left_duration": _round_sec(
                    _sample_to_sec(plan.target_left_samples, sample_rate)
                ),
                "right_duration": _round_sec(
                    _sample_to_sec(plan.target_right_samples, sample_rate)
                ),
            },
            "adjusted_pause": {
                "start": _round_sec(adjusted_start_sec),
                "end": _round_sec(adjusted_end_sec),
                "duration": _round_sec(target_duration_sec),
            },
            "regen_margin_sec": _round_sec(
                _sample_to_sec(plan.regen_margin_samples, sample_rate)
            ),
        }

        point_regen_spans = []
        if insert_start_idx is not None and insert_end_idx is not None:
            protected_intervals.append(
                [
                    _sample_to_sec(insert_start_idx, sample_rate),
                    _sample_to_sec(insert_end_idx, sample_rate),
                ]
            )
            entry["audio_insert_interval"] = {
                "start": _round_sec(_sample_to_sec(insert_start_idx, sample_rate)),
                "end": _round_sec(_sample_to_sec(insert_end_idx, sample_rate)),
                "duration": _round_sec(
                    _sample_to_sec(insert_end_idx - insert_start_idx, sample_rate)
                ),
                "audio_insert": True,
            }
            entry["audio_insert_intrusion"] = {
                "ratio": _round_sec(
                    plan.intrusion_samples / max(1, plan.target_duration_samples)
                ),
                "duration": _round_sec(
                    _sample_to_sec(plan.intrusion_samples, sample_rate)
                ),
                "left_end": _round_sec(
                    _sample_to_sec(left_intrusion_end_idx, sample_rate)
                ),
                "right_start": _round_sec(
                    _sample_to_sec(right_intrusion_start_idx, sample_rate)
                ),
            }
            left_span = _append_regen_span(
                regen_spans,
                start_idx=left_regen_start_idx,
                end_idx=left_intrusion_end_idx,
                sample_rate=sample_rate,
            )
            right_span = _append_regen_span(
                regen_spans,
                start_idx=right_intrusion_start_idx,
                end_idx=right_regen_end_idx,
                sample_rate=sample_rate,
            )
            point_regen_spans = [
                span for span in [left_span, right_span] if span is not None
            ]
        else:
            span = _append_regen_span(
                regen_spans,
                start_idx=regen_start_idx,
                end_idx=regen_end_idx,
                sample_rate=sample_rate,
            )
            if span is not None:
                point_regen_spans = [span]

        if point_regen_spans:
            if len(point_regen_spans) == 1 and "audio_insert_interval" not in entry:
                entry["regen"] = point_regen_spans[0]
            else:
                entry["regen"] = {"spans": point_regen_spans}
            points_processed += 1
        else:
            entry["regen"] = "skipped_too_short"

        details.append(entry)

    return details, regen_spans, protected_intervals, points_processed


def smooth_audio(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    points: List[SmoothingPoint],
    *,
    f5tts_url: str = "http://localhost:8001",
    aligner_url: str = "http://localhost:8029",
    aligner_backend: str = "auto",
    language: Optional[str] = None,
    vad_url: str = "http://localhost:8005",
    f5_steps: int = 32,
    f5_cfg_strength: float = 2.0,
    f5_sway_sampling_coef: float = -1.0,
    f5_seed: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[str] = None,
) -> SmoothingResult:
    """
    Smooth audio at specified boundary points.

    Single-pass pipeline:
      1. Normalize all points into ``position_sec`` + optional ``pause_interval_sec``
      2. Resolve native pause intervals for points without explicit intervals
      3. Clamp or expand each pause to its target duration
      4. Rebuild the audio by shrinking or expanding each native pause around
         the boundary anchor
      5. Construct masked spans: adjusted_pause ± regen_margin_sec, or, for
         audio_insert mode, two disjoint transition spans around the insert interval
      6. Regenerate all masked regions via one F5-TTS call and apply any
          audio_insert replacements

    Args:
        audio: Input waveform (float32, mono, 1D).
        sample_rate: Sample rate of the input audio.
        text: Full text corresponding to the audio.
        points: List of SmoothingPoint specs, one per boundary.
        f5tts_url: URL of the F5-TTS synthesis service.
        aligner_url: URL of the aligner gateway service, used only when a point
            does not provide ``pause_interval_sec``.
        aligner_backend: Explicit aligner gateway backend ("qwen3" or "auto").
        language: Optional language code used by the aligner gateway.
        vad_url: URL of the VAD service, used only when a point does not provide
            ``pause_interval_sec``.
        f5_steps: Number of F5-TTS denoising steps.
        f5_cfg_strength: F5-TTS classifier-free guidance strength.
        f5_sway_sampling_coef: F5-TTS sway sampling coefficient.
        f5_seed: Random seed for F5-TTS (None = random).
        debug: If True, save intermediate audio files to debug_dir.
        debug_dir: Directory for debug output (auto-generated if None).

    Returns:
        SmoothingResult with the smoothed audio and processing details.
    """
    import json as json_mod
    import os

    if audio.ndim != 1:
        audio = audio.squeeze()
    audio = audio.astype(np.float32)

    # Debug setup
    if debug:
        if debug_dir is None:
            import tempfile

            debug_dir = tempfile.mkdtemp(prefix="smooth_debug_")
        debug_dir = os.path.abspath(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        _debug_save_wav(debug_dir, "00_original.wav", audio, sample_rate)
        logger.info(f"Debug mode: saving artifacts to {debug_dir}")

    total_dur = len(audio) / sample_rate
    normalized_points = _normalize_points(points, total_dur)
    if not normalized_points:
        return SmoothingResult(
            audio=audio,
            sample_rate=sample_rate,
            points_processed=0,
            details=[],
            debug_dir=debug_dir,
        )

    if any(point.mode == "smooth_only" for point in normalized_points) and any(
        point.mode == "adjust" for point in normalized_points
    ):
        raise ValueError(
            "A single request must not mix mode='smooth_only' and mode='adjust' points"
        )

    if all(point.mode == "smooth_only" for point in normalized_points):
        details, regen_spans, points_processed = _build_smooth_only_details_and_spans(
            normalized_points,
            sample_rate=sample_rate,
            total_dur=total_dur,
        )
        adjusted_audio = audio
        adjusted_duration = total_dur
        resolution_mode = "smooth_only"
        vad_silences = []
        protected_intervals = []
    else:
        needs_interval_resolution = any(
            point.pause_interval_sec is None for point in normalized_points
        )
        resolution_mode = "explicit_only"
        alignments = []
        vad_silences = []

        if needs_interval_resolution:
            resolution_mode = (
                "mixed"
                if any(
                    point.pause_interval_sec is not None
                    for point in normalized_points
                )
                else "auto_detect"
            )
            aligner_debug_dir = (
                os.path.join(debug_dir, "aligner") if debug and debug_dir else None
            )
            alignments = _call_aligner(
                audio,
                sample_rate,
                text,
                aligner_url,
                aligner_backend=aligner_backend,
                language=language,
                debug=debug,
                debug_dir=aligner_debug_dir,
            )

            vad_segments = _call_vad_detect(audio, sample_rate, vad_url)

            vad_silences = _compute_vad_silences(
                vad_segments, total_dur, shrink=0.05
            )

        plans = _plan_pause_adjustments(
            normalized_points,
            sample_rate=sample_rate,
            audio_num_samples=len(audio),
            total_dur=total_dur,
            alignments=alignments,
            vad_silences=vad_silences,
        )

        adjusted_audio = _build_pause_adjusted_audio(audio, plans)
        adjusted_duration = len(adjusted_audio) / sample_rate
        if debug and debug_dir:
            _debug_save_wav(
                debug_dir, "01_pause_adjusted.wav", adjusted_audio, sample_rate
            )

        (
            details,
            regen_spans,
            protected_intervals,
            points_processed,
        ) = _build_details_and_regen_spans(
            adjusted_audio,
            plans,
            sample_rate=sample_rate,
        )

    f5_input_audio = adjusted_audio
    if "plans" in locals():
        f5_input_audio = _build_f5_input_audio(adjusted_audio, plans)

    if regen_spans:
        regen_spans.sort()
        merged_spans = [list(regen_spans[0])]
        for s, e in regen_spans[1:]:
            if s <= merged_spans[-1][1]:
                merged_spans[-1][1] = max(merged_spans[-1][1], e)
            else:
                merged_spans.append([s, e])
    else:
        merged_spans = []

    if debug and debug_dir and merged_spans:
        _debug_save_wav(debug_dir, "02_f5_input.wav", f5_input_audio, sample_rate)
        mask_audio = _build_debug_mask_visualization(
            adjusted_audio,
            merged_spans=merged_spans,
            plans=plans if "plans" in locals() else [],
            sample_rate=sample_rate,
        )
        _debug_save_wav(debug_dir, "03_mask_visualization.wav", mask_audio, sample_rate)

    post_f5_audio = f5_input_audio
    raw_f5_output_audio = None
    raw_f5_output_sample_rate = F5_SAMPLE_RATE
    if merged_spans:
        regeneration = _regenerate_regions(
            f5_input_audio,
            sample_rate,
            text,
            merged_spans,
            f5tts_url=f5tts_url,
            steps=f5_steps,
            cfg_strength=f5_cfg_strength,
            sway_sampling_coef=f5_sway_sampling_coef,
            seed=f5_seed,
        )
        post_f5_audio = regeneration.spliced_audio
        raw_f5_output_audio = regeneration.f5_output_audio
        raw_f5_output_sample_rate = regeneration.f5_output_sample_rate
    final_audio = post_f5_audio
    if "plans" in locals():
        final_audio = _apply_audio_inserts(post_f5_audio, plans)

    if debug and debug_dir:
        if raw_f5_output_audio is not None:
            _debug_save_wav(
                debug_dir,
                "04_f5_output.wav",
                raw_f5_output_audio,
                raw_f5_output_sample_rate,
            )
        _debug_save_wav(debug_dir, "05_smoothed.wav", post_f5_audio, sample_rate)
        if "plans" in locals() and any(plan.audio_insert is not None for plan in plans):
            _debug_save_wav(
                debug_dir,
                "06_audio_inserted.wav",
                final_audio,
                sample_rate,
            )
        debug_info = {
            "interval_resolution_mode": resolution_mode,
            "text": text,
            "sample_rate": sample_rate,
            "original_duration": _round_sec(total_dur),
            "pause_adjusted_duration": _round_sec(adjusted_duration),
            "final_duration": _round_sec(len(final_audio) / sample_rate),
            "num_points": len(points),
            "points_processed": points_processed,
            "f5_steps": f5_steps,
            "f5_cfg_strength": f5_cfg_strength,
            "f5_sway_sampling_coef": f5_sway_sampling_coef,
            "f5_seed": f5_seed,
            "regen_spans_merged": [
                {"start": _round_sec(s), "end": _round_sec(e), "duration": _round_sec(e - s)}
                for s, e in merged_spans
            ],
            "protected_intervals": [
                {"start": _round_sec(s), "end": _round_sec(e), "duration": _round_sec(e - s)}
                for s, e in protected_intervals
            ],
            "vad_silences": vad_silences,
            "points_config": [
                {
                    "position_sec": _round_sec(p.position_sec),
                    "mode": p.mode,
                    "pause_interval_sec": None
                    if p.pause_interval_sec is None
                    else [_round_sec(p.pause_interval_sec[0]), _round_sec(p.pause_interval_sec[1])],
                    "max_pause_sec": None if p.max_pause_sec is None else _round_sec(p.max_pause_sec),
                    "min_pause_sec": None if p.min_pause_sec is None else _round_sec(p.min_pause_sec),
                    "regen_margin_sec": _round_sec(p.regen_margin_sec),
                    "audio_insert": p.audio_insert is not None,
                    "audio_insert_intrusion_ratio": _round_sec(
                        p.audio_insert_intrusion_ratio
                    ),
                }
                for p in normalized_points
            ],
            "details": details,
        }
        with open(os.path.join(debug_dir, "debug_info.json"), "w") as f:
            json_mod.dump(debug_info, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Debug artifacts saved to {debug_dir}")

    return SmoothingResult(
        audio=final_audio,
        sample_rate=sample_rate,
        points_processed=points_processed,
        details=details,
        debug_dir=debug_dir,
    )


# ---- VAD Silence Detection ----


def _compute_vad_silences(
    vad_segments: List[Dict],
    total_dur: float,
    shrink: float = 0.05,
) -> List[Dict]:
    """Convert VAD speech segments to silence gaps, shrunk by `shrink` seconds.

    Returns list of {"start": ..., "end": ..., "duration": ...} for each silence gap.
    """
    if not vad_segments:
        return []

    segs = sorted(vad_segments, key=lambda s: s["start"])
    silences = []

    # Gap before first speech
    if segs[0]["start"] > shrink * 2:
        s = 0.0 + shrink
        e = segs[0]["start"] - shrink
        if e > s:
            silences.append(
                {"start": round(s, 4), "end": round(e, 4), "duration": round(e - s, 4)}
            )

    # Gaps between speech segments
    for i in range(len(segs) - 1):
        gap_start = segs[i]["end"] + shrink
        gap_end = segs[i + 1]["start"] - shrink
        if gap_end > gap_start:
            silences.append(
                {
                    "start": round(gap_start, 4),
                    "end": round(gap_end, 4),
                    "duration": round(gap_end - gap_start, 4),
                }
            )

    # Gap after last speech
    if total_dur - segs[-1]["end"] > shrink * 2:
        s = segs[-1]["end"] + shrink
        e = total_dur - shrink
        if e > s:
            silences.append(
                {"start": round(s, 4), "end": round(e, 4), "duration": round(e - s, 4)}
            )

    return silences


def _refine_pause_with_vad(
    pause: Dict,
    vad_silences: List[Dict],
) -> Optional[Dict]:
    """Refine alignment-detected pause boundaries using VAD silence gaps.

    Finds the VAD silence that best overlaps with the alignment-detected pause,
    and uses it as the refined pause boundary.

    Returns refined pause dict or None if no matching VAD silence found.
    """
    pause_start = pause["pause_start"]
    pause_end = pause["pause_end"]

    best_silence = None
    best_overlap = 0.0

    for sil in vad_silences:
        # Compute overlap between alignment pause and VAD silence
        overlap_start = max(pause_start, sil["start"])
        overlap_end = min(pause_end, sil["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_silence = sil

    if best_silence is None or best_overlap < 0.01:
        return None

    # Use the intersection of alignment pause and VAD silence
    refined_start = max(pause_start, best_silence["start"])
    refined_end = min(pause_end, best_silence["end"])
    refined_dur = max(0.0, refined_end - refined_start)

    if refined_dur < 0.01:
        return None

    return {
        "pause_start": round(refined_start, 4),
        "pause_end": round(refined_end, 4),
        "pause_duration": round(refined_dur, 4),
        "method": "alignment+vad",
    }


# ---- Pause Detection from Alignment ----


def _find_pause_from_alignment(
    alignments: List[Dict],
    position_sec: float,
    total_dur: float,
) -> Dict:
    """Find the pause around a boundary position using word alignment data."""
    last_before_end = 0.0
    first_after_start = total_dur

    for a in alignments:
        char_end = a["end"]
        char_start = a["start"]
        char_val = a.get("char", "")

        # Skip spaces, punctuation entries, and punctuation chars
        if char_val.strip() == "" or a.get("is_punctuation", False):
            continue
        if char_val in "。，！？.,!?;；：:、":
            continue

        if char_end <= position_sec + 0.05:
            last_before_end = max(last_before_end, char_end)
        if char_start >= position_sec - 0.05:
            first_after_start = min(first_after_start, char_start)

    pause_start = last_before_end
    pause_end = first_after_start
    pause_duration = max(0.0, pause_end - pause_start)

    return {
        "pause_start": round(pause_start, 4),
        "pause_end": round(pause_end, 4),
        "pause_duration": round(pause_duration, 4),
        "method": "alignment",
    }


# ---- F5-TTS Regeneration (multi-span) ----


def _regenerate_regions(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    regen_spans: List[List[float]],
    *,
    f5tts_url: str,
    steps: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    seed: Optional[int] = None,
    crossfade_sec: float = 0.02,
) -> _RegenerationResult:
    """
    Regenerate multiple time regions using F5-TTS mask prediction in a single call.

    Sends the full audio + text + all edit_spans_sec to F5-TTS, which
    regenerates all masked regions simultaneously and returns a waveform-
    preserved splice with crossfade.

    Handles sample rate conversion: resamples to 24kHz for F5-TTS,
    while passing the original-rate waveform as preserve material.
    """
    if not regen_spans:
        if sample_rate != F5_SAMPLE_RATE:
            audio_f5 = _resample(audio, sample_rate, F5_SAMPLE_RATE)
        else:
            audio_f5 = audio.copy()
        return _RegenerationResult(
            spliced_audio=audio.copy(),
            f5_output_audio=audio_f5,
        )

    # Resample to F5-TTS sample rate if needed
    if sample_rate != F5_SAMPLE_RATE:
        audio_f5 = _resample(audio, sample_rate, F5_SAMPLE_RATE)
    else:
        audio_f5 = audio.copy()
    num_frames = (len(audio_f5) // F5_HOP_LENGTH) + 1
    # The old non-preserve F5 path auto-extended editing duration by one frame
    # before local splicing. Preserve mode forbids implicit extension, so the
    # smoother requests that legacy model duration explicitly while F5 still
    # returns an exact-length preserve splice.
    model_duration_frames = num_frames + 1
    requested_duration_sec = model_duration_frames * F5_HOP_LENGTH / F5_SAMPLE_RATE
    if requested_duration_sec > MAX_F5_EDIT_SECONDS:
        raise ValueError(
            "Boundary smoother F5 regeneration input exceeds the single-call "
            f"F5 edit limit: requested_duration={requested_duration_sec:.2f}s > "
            f"{MAX_F5_EDIT_SECONDS:.2f}s. Reduce input length or disable smoothing."
        )

    # Encode audio as tensor
    audio_2d = audio_f5.reshape(1, -1).astype(np.float32)
    cond_encoded = {
        "data": base64.b64encode(audio_2d.tobytes()).decode("utf-8"),
        "shape": list(audio_2d.shape),
        "dtype": "float32",
    }
    preserve_2d = audio.reshape(1, -1).astype(np.float32)
    preserve_encoded = {
        "data": base64.b64encode(preserve_2d.tobytes()).decode("utf-8"),
        "shape": list(preserve_2d.shape),
        "dtype": "float32",
    }

    payload = {
        "cond": cond_encoded,
        "text": [text],
        "duration": model_duration_frames,
        "edit_spans_sec": regen_spans,
        "steps": steps,
        "cfg_strength": cfg_strength,
        "sway_sampling_coef": sway_sampling_coef,
        "output_format": "numpy",
        "input_sample_rate": F5_SAMPLE_RATE,
        "preserve_unmasked_waveform": True,
        "preserve_waveform": preserve_encoded,
        "preserve_waveform_sample_rate": sample_rate,
        "preserve_spans_sec": regen_spans,
        "source_crossfade_sec": 0.0,
        "regen_crossfade_sec": crossfade_sec,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(f"{f5tts_url}/sample", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"F5-TTS sample failed ({resp.status_code}): {resp.text[:300]}"
        )

    result = resp.json()
    returned_sample_rate = result.get("sample_rate")
    if returned_sample_rate is not None and returned_sample_rate != sample_rate:
        raise RuntimeError(
            f"F5-TTS preserve splice returned sample_rate={returned_sample_rate}, expected {sample_rate}"
        )
    tensor_data = result["tensor"]
    spliced_audio = np.frombuffer(
        base64.b64decode(tensor_data["data"]),
        dtype=np.dtype(tensor_data["dtype"]),
    ).reshape(tensor_data["shape"])
    spliced_audio = np.asarray(spliced_audio, dtype=np.float32).squeeze()

    if len(spliced_audio) != len(audio):
        raise RuntimeError(
            f"F5-TTS preserve splice returned {len(spliced_audio)} samples, expected {len(audio)}"
        )
    return _RegenerationResult(
        spliced_audio=spliced_audio,
        f5_output_audio=spliced_audio,
        f5_output_sample_rate=sample_rate,
    )


# ---- Service Clients ----


def _call_aligner(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    aligner_url: str,
    *,
    aligner_backend: str = "auto",
    language: Optional[str] = None,
    debug: bool = False,
    debug_dir: Optional[str] = None,
) -> List[Dict]:
    """Call the aligner gateway service and return alignments."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    backend = str(aligner_backend or "auto").strip().lower()
    payload = {
        "backend": backend,
        "audio": {"data": audio_b64, "format": "wav"},
        "text": text,
        "word_level": True,
    }
    if backend not in {"auto", "qwen3"}:
        raise ValueError("aligner_backend must be 'auto' or 'qwen3'")
    if backend == "qwen3" and not language:
        raise ValueError("language is required when aligner_backend='qwen3'")
    if language:
        payload["language"] = language

    resp = requests.post(f"{aligner_url}/align", json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Aligner failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    return data.get("alignments", [])


def _call_vad_detect(
    audio: np.ndarray,
    sample_rate: int,
    vad_url: str,
) -> List[Dict]:
    """Call VAD service /detect and return speech segments."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "audio": {"data": audio_b64, "format": "wav"},
        "min_silence_len": 100,
        "silence_thresh": -40,
    }
    resp = requests.post(f"{vad_url}/detect", json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"VAD detect failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    return data.get("segments", [])


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using soxr (high quality)."""
    if orig_sr == target_sr:
        return audio

    import soxr

    return soxr.resample(audio, orig_sr, target_sr).astype(np.float32)
