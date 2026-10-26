"""Speech synthesis for pause editing pipeline.

The waveform edit is explicit: insertion adds silence and reduction removes a
centered span from the aligned gap between adjacent anchor words.  F5-TTS only
regenerates small transition regions around the resulting seams.

This keeps pause timing under explicit control while still letting F5 repair the
two seam regions.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from speech_edit.clients.services import (
    AlignmentClient,
    BoundaryPauseSmootherClient,
    ServiceError,
)
from speech_edit.data.language import classify_text_language

from .tag_utils import (
    ALL_PAUSE_PUNCTUATION,
    PauseEditGroup,
    PauseEditProfile,
    PauseTag,
    compute_generated_text,
    tags_to_aligner_queries,
    tags_to_boundary_queries,
)

logger = logging.getLogger(__name__)

DEFAULT_DUAL_BOUNDARY_MAX_REGEN_MARGIN_SEC = 0.06
MIN_REGEN_MARGIN_SEC = 0.05
DEFAULT_CENTER_CONTEXT_SEC = 0.03
DEFAULT_CENTER_MAX_REGEN_MARGIN_SEC = 0.12
MAX_SMOOTHER_REGEN_MARGIN_SEC = 1.0


@dataclass
class SynthesizedPair:
    """An original -> generated audio pair from pause editing."""

    profile: PauseEditProfile
    normed_text: str
    generated_text: str
    original_audio_bytes: bytes
    generated_audio_bytes: bytes
    original_duration_sec: float
    generated_duration_sec: float
    original_word_alignments: Optional[List[Dict[str, Any]]] = None
    pause_edit_plan: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


def _aligner_language_code(text: str) -> Optional[str]:
    language = classify_text_language(text)
    return {
        "en": "eng",
        "zh": "zho",
    }.get(language)


def _read_audio_duration(audio_bytes: bytes) -> float:
    with io.BytesIO(audio_bytes) as buf:
        return sf.info(buf).duration


def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as buf:
        audio, sample_rate = sf.read(buf, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=-1)
    return audio.astype(np.float32), int(sample_rate)


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _extract_gap_from_answers(answers: List[Dict]) -> Optional[float]:
    if len(answers) < 2:
        return None
    left_end = answers[0].get("end")
    right_start = answers[1].get("start")
    if left_end is None or right_start is None:
        return None
    return max(0.0, float(right_start) - float(left_end))


def _extract_boundary_point(answer: Dict) -> Optional[float]:
    start = answer.get("start")
    end = answer.get("end")
    if start is None or end is None:
        return None
    return (float(start) + float(end)) / 2.0


def _resolve_boundary_position_sec(
    normed_text: str,
    tag: PauseTag,
    original_audio_bytes: bytes,
    alignment_client: AlignmentClient,
    *,
    aligner_backend: str,
    aligner_language: Optional[str],
) -> float:
    boundary_query = tags_to_boundary_queries(normed_text, [tag])[0]["boundary_query_full"]
    response = alignment_client.align(
        audio_bytes=original_audio_bytes,
        text=normed_text,
        language=aligner_language,
        backend=aligner_backend,
        queries=[boundary_query],
        return_response=True,
    )
    answers = response.get("answers") or []
    query_ok = response.get("query_tokenization_matches_default", True)
    if query_ok and answers:
        point = _extract_boundary_point(answers[0])
        if point is not None:
            return point

    raise ValueError(f"Could not resolve insertion boundary for char_idx={tag.char_idx}")


def _resolve_boundary_positions_sec(
    normed_text: str,
    tags: List[PauseTag],
    original_audio_bytes: bytes,
    alignment_client: AlignmentClient,
    *,
    aligner_backend: str,
    aligner_language: Optional[str],
) -> List[float]:
    return [
        _resolve_boundary_position_sec(
            normed_text,
            tag,
            original_audio_bytes,
            alignment_client,
            aligner_backend=aligner_backend,
            aligner_language=aligner_language,
        )
        for tag in tags
    ]


def _resolve_reduction_gap_sec(
    normed_text: str,
    tag: PauseTag,
    original_audio_bytes: bytes,
    alignment_client: AlignmentClient,
    *,
    aligner_backend: str,
    aligner_language: Optional[str],
) -> Tuple[float, float]:
    anchor_spec = tags_to_aligner_queries(normed_text, [tag])[0]
    response = alignment_client.align(
        audio_bytes=original_audio_bytes,
        text=normed_text,
        language=aligner_language,
        backend=aligner_backend,
        queries=[anchor_spec["left_query_full"], anchor_spec["right_query_full"]],
        return_response=True,
    )
    answers = response.get("answers") or []
    if len(answers) < 2:
        raise ValueError(
            f"Could not resolve reduction gap for char_idx={tag.char_idx}: "
            "missing aligned anchor answers"
        )
    left_end = answers[0].get("end")
    right_start = answers[1].get("start")
    if left_end is None or right_start is None:
        raise ValueError(
            f"Could not resolve reduction gap for char_idx={tag.char_idx}: "
            "missing left end or right start"
        )
    gap_start = float(left_end)
    gap_end = float(right_start)
    if gap_end < gap_start:
        raise ValueError(
            f"Could not resolve reduction gap for char_idx={tag.char_idx}: "
            f"right anchor starts before left anchor ends ({gap_start:.6f}, {gap_end:.6f})"
        )
    return gap_start, gap_end


def _align_original_words(
    normed_text: str,
    original_audio_bytes: bytes,
    alignment_client: AlignmentClient,
    *,
    aligner_backend: str,
    aligner_language: Optional[str],
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    response = alignment_client.align(
        audio_bytes=original_audio_bytes,
        text=normed_text,
        language=aligner_language,
        backend=aligner_backend,
        word_level=True,
        return_response=True,
    )

    alignments = response.get("alignments") if isinstance(response, dict) else None
    if not isinstance(alignments, list):
        raise ValueError("Missing alignments in aligner response")
    return alignments, {
        "backend": response.get("backend", aligner_backend),
        "language": aligner_language,
        "word_alignment_count": len(alignments),
    }


def _apply_pause_waveform_edits(
    original_audio: np.ndarray,
    sample_rate: int,
    *,
    operations: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Apply original-timeline operations in one pass and return edited positions."""
    if not operations:
        return original_audio, []
    prepared: List[Dict[str, Any]] = []
    for operation in operations:
        item = dict(operation)
        if item["action"] == "ins":
            position = min(
                max(int(round(float(item["position_sec"]) * sample_rate)), 0),
                len(original_audio),
            )
            item.update(edit_start=position, edit_end=position)
        else:
            gap_start = min(
                max(int(round(float(item["gap_start_sec"]) * sample_rate)), 0),
                len(original_audio),
            )
            gap_end = min(
                max(int(round(float(item["gap_end_sec"]) * sample_rate)), gap_start),
                len(original_audio),
            )
            requested = max(0, int(round(float(item["duration_sec"]) * sample_rate)))
            removed = min(requested, gap_end - gap_start)
            remove_start = gap_start + ((gap_end - gap_start - removed) // 2)
            item.update(
                gap_start_sample=gap_start,
                gap_end_sample=gap_end,
                edit_start=remove_start,
                edit_end=remove_start + removed,
            )
        prepared.append(item)
    prepared.sort(
        key=lambda item: (
            int(item["edit_start"]),
            int(item["edit_end"]),
            int(item["tag_index"]),
        )
    )
    parts: List[np.ndarray] = []
    cursor = 0
    output_samples = 0
    plans: List[Optional[Dict[str, Any]]] = [None] * len(operations)
    for item in prepared:
        edit_start = int(item["edit_start"])
        edit_end = int(item["edit_end"])
        if edit_start < cursor:
            raise ValueError("Aligned pause edit spans overlap on the original waveform")
        untouched = original_audio[cursor:edit_start]
        parts.append(untouched)
        output_samples += len(untouched)
        tag_index = int(item["tag_index"])
        if item["action"] == "ins":
            inserted_samples = max(
                0, int(round(float(item["duration_sec"]) * sample_rate))
            )
            pause_start = output_samples
            parts.append(np.zeros(inserted_samples, dtype=np.float32))
            output_samples += inserted_samples
            plans[tag_index] = {
                "action": "ins",
                "tag_index": tag_index,
                "requested_duration_sec": inserted_samples / sample_rate,
                "actual_duration_sec": inserted_samples / sample_rate,
                "source_position_sec": edit_start / sample_rate,
                "edited_pause_interval_sec": [
                    pause_start / sample_rate,
                    output_samples / sample_rate,
                ],
            }
        else:
            removed_samples = edit_end - edit_start
            plans[tag_index] = {
                "action": "red",
                "tag_index": tag_index,
                "requested_duration_sec": float(item["duration_sec"]),
                "actual_duration_sec": removed_samples / sample_rate,
                "source_gap_sec": [
                    int(item["gap_start_sample"]) / sample_rate,
                    int(item["gap_end_sample"]) / sample_rate,
                ],
                "source_remove_interval_sec": [
                    edit_start / sample_rate,
                    edit_end / sample_rate,
                ],
                "edited_stitch_position_sec": output_samples / sample_rate,
            }
        cursor = edit_end
    tail = original_audio[cursor:]
    parts.append(tail)
    return np.concatenate(parts, axis=0), [
        plan for plan in plans if plan is not None
    ]


def _build_f5_text_for_pauses(normed_text: str, tags: List[PauseTag]) -> str:
    """Build the internal F5 text, conditionally injecting comma markers.

    The human-facing generated_text stays equal to the original transcript. This
    helper only prepares the text prompt passed to F5. For each insertion point,
    if the position is already after a pause-like punctuation mark, no extra
    comma is inserted.
    """
    result = normed_text
    for tag in sorted(
        (item for item in tags if item.act == "ins"),
        key=lambda item: item.char_idx,
        reverse=True,
    ):
        if tag.char_idx > 0 and normed_text[tag.char_idx - 1] in ALL_PAUSE_PUNCTUATION:
            continue
        result = result[: tag.char_idx] + "," + result[tag.char_idx :]
    return result


def _effective_regen_margin_sec(
    silence_sec: float,
    max_regen_margin_sec: float,
) -> float:
    # Keep some untouched center silence so left/right smoothing spans do not
    # collapse into one merged region for the shortest pause level.
    upper = max(MIN_REGEN_MARGIN_SEC, (silence_sec / 2.0) - 0.005)
    return min(max_regen_margin_sec, upper)


def _build_smoothing_points(
    pause_start_sec: float,
    pause_end_sec: float,
    *,
    max_regen_margin_sec: float,
) -> Tuple[List[Dict], float]:
    silence_sec = pause_end_sec - pause_start_sec
    regen_margin_sec = _effective_regen_margin_sec(
        silence_sec, max_regen_margin_sec
    )
    return (
        [
            {
                "mode": "smooth_only",
                "position_sec": pause_start_sec,
                "regen_margin_sec": regen_margin_sec,
            },
            {
                "mode": "smooth_only",
                "position_sec": pause_end_sec,
                "regen_margin_sec": regen_margin_sec,
            },
        ],
        regen_margin_sec,
    )


def _build_exact_stage1_points(
    pause_intervals_sec: List[Tuple[float, float]],
    *,
    context_margin_sec: float,
) -> Tuple[List[Dict], float]:
    """Build two local smooth-only points using only shared-tool-supported fields."""
    margin = max(MIN_REGEN_MARGIN_SEC, context_margin_sec)
    points: List[Dict] = []
    for pause_start_sec, pause_end_sec in pause_intervals_sec:
        points.extend(
            [
                {
                    "mode": "smooth_only",
                    "position_sec": pause_start_sec,
                    "regen_margin_sec": margin,
                },
                {
                    "mode": "smooth_only",
                    "position_sec": pause_end_sec,
                    "regen_margin_sec": margin,
                },
            ]
        )
    return (points, margin)


def _build_single_region_smoothing_points(
    pause_intervals_sec: List[Tuple[float, float]],
    *,
    context_margin_sec: float,
) -> Tuple[List[Dict], float]:
    """Build one center point per pause interval using shared-tool-supported fields."""
    points: List[Dict] = []
    regen_margin_sec = 0.0
    for pause_start_sec, pause_end_sec in pause_intervals_sec:
        silence_sec = pause_end_sec - pause_start_sec
        full_start = max(0.0, pause_start_sec - context_margin_sec)
        full_end = pause_end_sec + context_margin_sec
        regen_margin_sec = min(
            MAX_SMOOTHER_REGEN_MARGIN_SEC,
            max(MIN_REGEN_MARGIN_SEC, (full_end - full_start) / 2.0),
        )
        points.append(
            {
                "mode": "smooth_only",
                "position_sec": (full_start + full_end) / 2.0,
                "regen_margin_sec": regen_margin_sec,
            }
        )
    return (points, max(regen_margin_sec, MIN_REGEN_MARGIN_SEC))


def _build_reduction_smoothing_points(
    stitch_positions_sec: List[float],
    *,
    context_margin_sec: float,
) -> Tuple[List[Dict], float]:
    margin = min(
        MAX_SMOOTHER_REGEN_MARGIN_SEC,
        max(MIN_REGEN_MARGIN_SEC, context_margin_sec),
    )
    return (
        [
            {
                "mode": "smooth_only",
                "position_sec": position,
                "regen_margin_sec": margin,
            }
            for position in stitch_positions_sec
        ],
        margin,
    )


def _build_center_regen_points(
    pause_intervals_sec: List[Tuple[float, float]],
    *,
    center_regen_margin_sec: Optional[float],
) -> Tuple[List[Dict], float]:
    points: List[Dict] = []
    effective_margin = MIN_REGEN_MARGIN_SEC
    for pause_start_sec, pause_end_sec in pause_intervals_sec:
        silence_sec = pause_end_sec - pause_start_sec
        margin = (
            float(center_regen_margin_sec)
            if center_regen_margin_sec is not None
            else min(
                DEFAULT_CENTER_MAX_REGEN_MARGIN_SEC,
                max(MIN_REGEN_MARGIN_SEC, (silence_sec / 2.0) + DEFAULT_CENTER_CONTEXT_SEC),
            )
        )
        margin = min(
            margin,
            max(MIN_REGEN_MARGIN_SEC, (silence_sec / 2.0) - 0.005),
        )
        effective_margin = max(effective_margin, margin)
        points.append(
            {
                "mode": "smooth_only",
                "position_sec": (pause_start_sec + pause_end_sec) / 2.0,
                "regen_margin_sec": margin,
            }
        )
    return points, effective_margin


def _merged_smoothing_span(points: List[Dict]) -> Optional[List[float]]:
    """Compute the actual merged regen span implied by smoothing points."""
    if not points:
        return None
    starts = []
    ends = []
    for point in points:
        starts.append(float(point["position_sec"]) - float(point["regen_margin_sec"]))
        ends.append(float(point["position_sec"]) + float(point["regen_margin_sec"]))
    return [round(min(starts), 6), round(max(ends), 6)]


def synthesize_pause_edit(
    profile: PauseEditProfile,
    group: PauseEditGroup,
    smoother_client: BoundaryPauseSmootherClient,
    *,
    alignment_client: AlignmentClient,
    output_dir: Optional[str] = None,
    expand_radius: float,
    nfe_step: Optional[int] = None,
    cfg_strength: Optional[float] = None,
    sway_sampling_coef: Optional[float] = None,
    timeout: float = 600,
    aligner_backend: str,
    synth_method: str,
    center_regen_second_pass: bool,
    center_regen_margin_sec: Optional[float],
) -> SynthesizedPair:
    """Synthesize a pause-edited audio from a profile.

    `dual_boundary`:
        Insert silence, smooth left/right boundaries separately, optionally run
        a second center regen.

    `single_region`:
        Insert silence, then mask one continuous region spanning the full pause
        plus a bit of left/right context, and regenerate once.

    `insert_only`:
        Insert silence and skip all smoother/F5 regeneration passes. Useful for
        isolating the effect of explicit silence insertion itself.
    """

    if alignment_client is None:
        raise ValueError(
            "Pause edits require an alignment client to resolve edit positions."
        )

    normed_text = group.normed_text
    tags = profile.tags
    if not tags:
        raise ValueError("Pause editing requires at least one tag per profile")

    generated_text = compute_generated_text(normed_text, tags)
    f5_text = _build_f5_text_for_pauses(normed_text, tags)
    aligner_language = _aligner_language_code(normed_text)
    original_word_alignments, original_alignment_info = _align_original_words(
        normed_text,
        group.original_audio,
        alignment_client,
        aligner_backend=aligner_backend,
        aligner_language=aligner_language,
    )

    original_audio, sample_rate = _decode_audio(group.original_audio)
    waveform_operations: List[Dict[str, Any]] = []
    for tag_index, tag in enumerate(tags):
        duration_sec = tag.delta_dur_ms / 1000.0
        if tag.act == "ins":
            position_sec = _resolve_boundary_position_sec(
                normed_text,
                tag,
                group.original_audio,
                alignment_client,
                aligner_backend=aligner_backend,
                aligner_language=aligner_language,
            )
            waveform_operations.append(
                {
                    "action": "ins",
                    "tag_index": tag_index,
                    "position_sec": position_sec,
                    "duration_sec": duration_sec,
                }
            )
        elif tag.act == "red":
            gap_start_sec, gap_end_sec = _resolve_reduction_gap_sec(
                normed_text,
                tag,
                group.original_audio,
                alignment_client,
                aligner_backend=aligner_backend,
                aligner_language=aligner_language,
            )
            waveform_operations.append(
                {
                    "action": "red",
                    "tag_index": tag_index,
                    "gap_start_sec": gap_start_sec,
                    "gap_end_sec": gap_end_sec,
                    "duration_sec": duration_sec,
                }
            )
        else:
            raise ValueError(f"Unsupported pause edit action: {tag.act}")
    edited_audio, operation_plans = _apply_pause_waveform_edits(
        original_audio,
        sample_rate,
        operations=waveform_operations,
    )
    edited_audio_bytes = _encode_wav(edited_audio, sample_rate)
    pause_intervals_sec = [
        tuple(plan["edited_pause_interval_sec"])
        for plan in operation_plans
        if plan["action"] == "ins"
    ]
    reduction_stitches_sec = [
        float(plan["edited_stitch_position_sec"])
        for plan in operation_plans
        if plan["action"] == "red" and float(plan["actual_duration_sec"]) > 0.0
    ]
    insert_positions_sec = [
        float(plan["source_position_sec"])
        for plan in operation_plans
        if plan["action"] == "ins"
    ]
    silence_secs = [
        float(plan["actual_duration_sec"])
        for plan in operation_plans
        if plan["action"] == "ins"
    ]

    if synth_method == "insert_only":
        smoothing_points = []
        effective_margin_sec = 0.0
    elif synth_method == "single_region":
        insertion_points, insertion_margin = _build_single_region_smoothing_points(
            pause_intervals_sec,
            context_margin_sec=expand_radius,
        )
        reduction_points, reduction_margin = _build_reduction_smoothing_points(
            reduction_stitches_sec,
            context_margin_sec=expand_radius,
        )
        smoothing_points = sorted(
            insertion_points + reduction_points,
            key=lambda point: float(point["position_sec"]),
        )
        effective_margin_sec = max(insertion_margin, reduction_margin)
    elif synth_method == "dual_boundary":
        dual_margin = min(
            expand_radius, DEFAULT_DUAL_BOUNDARY_MAX_REGEN_MARGIN_SEC
        )
        insertion_points, insertion_margin = _build_exact_stage1_points(
            pause_intervals_sec,
            context_margin_sec=dual_margin,
        )
        reduction_points, reduction_margin = _build_reduction_smoothing_points(
            reduction_stitches_sec,
            context_margin_sec=dual_margin,
        )
        smoothing_points = sorted(
            insertion_points + reduction_points,
            key=lambda point: float(point["position_sec"]),
        )
        effective_margin_sec = max(insertion_margin, reduction_margin)
    else:
        raise ValueError(f"Unknown synth_method: {synth_method}")

    if synth_method == "insert_only" or not smoothing_points:
        generated_audio_bytes = edited_audio_bytes
    else:
        generated_audio_bytes = smoother_client.smooth(
            audio_bytes=edited_audio_bytes,
            text=f5_text,
            points=smoothing_points,
            aligner_backend=aligner_backend,
            language=aligner_language,
            f5_steps=nfe_step or 32,
            f5_cfg_strength=cfg_strength or 2.0,
            f5_sway_sampling_coef=sway_sampling_coef or -1.0,
            output_format="wav",
            debug=False,
            timeout=timeout,
        )

    center_regen_info = None
    if (
        center_regen_second_pass
        and synth_method == "dual_boundary"
        and pause_intervals_sec
    ):
        center_points, effective_center_margin_sec = _build_center_regen_points(
            pause_intervals_sec,
            center_regen_margin_sec=center_regen_margin_sec,
        )
        generated_audio_bytes = smoother_client.smooth(
            audio_bytes=generated_audio_bytes,
            text=f5_text,
            points=center_points,
            aligner_backend=aligner_backend,
            language=aligner_language,
            f5_steps=nfe_step or 32,
            f5_cfg_strength=cfg_strength or 2.0,
            f5_sway_sampling_coef=sway_sampling_coef or -1.0,
            output_format="wav",
            debug=False,
            timeout=timeout,
        )
        center_regen_info = {
            "enabled": True,
            "points": center_points,
            "effective_regen_margin_sec": effective_center_margin_sec,
        }

    original_duration = _read_audio_duration(group.original_audio)
    generated_duration = _read_audio_duration(generated_audio_bytes)

    has_reduction = bool(reduction_stitches_sec)
    pause_edit_plan = {
        "mode": (
            "splice_only"
            if synth_method == "insert_only" and has_reduction
            else "insert_only_explicit_silence"
            if synth_method == "insert_only"
            else "single_region_pause_edit"
            if synth_method == "single_region" and has_reduction
            else "single_region_smooth_only"
            if synth_method == "single_region"
            else "dual_boundary_pause_edit"
            if has_reduction
            else "dual_boundary_smooth_only"
        ),
        "aligner_backend": aligner_backend,
        "aligner_language": aligner_language,
        "original_alignment": original_alignment_info,
        "f5_text": f5_text,
        "insert_position_sec": [round(pos, 6) for pos in insert_positions_sec],
        "pause_duration_sec": [round(sec, 6) for sec in silence_secs],
        "pause_interval_sec": [
            [round(start, 6), round(end, 6)] for start, end in pause_intervals_sec
        ],
        "smoothing_points": smoothing_points,
        "operations": [
            {
                **plan,
                "level": tags[int(plan["tag_index"])].level,
            }
            for plan in operation_plans
        ],
    }
    if synth_method == "single_region":
        pause_edit_plan["merged_regen_expected_span"] = _merged_smoothing_span(
            smoothing_points
        )
    if center_regen_info is not None:
        pause_edit_plan["second_pass_center_regen"] = center_regen_info

    return SynthesizedPair(
        profile=profile,
        normed_text=normed_text,
        generated_text=generated_text,
        original_audio_bytes=group.original_audio,
        generated_audio_bytes=generated_audio_bytes,
        original_duration_sec=original_duration,
        generated_duration_sec=generated_duration,
        original_word_alignments=original_word_alignments,
        pause_edit_plan=pause_edit_plan,
        metadata={
            "max_regen_margin_sec": expand_radius,
            "effective_regen_margin_sec": effective_margin_sec,
            "inserted_audio_duration_sec": [round(sec, 6) for sec in silence_secs],
            "removed_audio_duration_sec": [
                round(float(plan["actual_duration_sec"]), 6)
                for plan in operation_plans
                if plan["action"] == "red"
            ],
            "aligner_backend": aligner_backend,
            "synth_method": synth_method,
            "center_regen_second_pass": center_regen_second_pass,
            "center_regen_margin_sec": center_regen_info["effective_regen_margin_sec"]
            if center_regen_info is not None
            else None,
        },
    )
