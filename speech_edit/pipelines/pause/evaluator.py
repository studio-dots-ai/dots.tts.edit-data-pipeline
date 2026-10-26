"""
Evaluation for pause editing pipeline.

Evaluates synthesized pause edit pairs on:
  1. Content accuracy: ASR transcription → WER/CER against reference text.
  2. Pause accuracy: Alignment-gap measurement at edit positions.
"""

from __future__ import annotations

from speech_edit.evaluation.audio import compute_wer, evaluate_quality, evaluate_speaker_similarity

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import soundfile as sf

from speech_edit.clients.services import (
    AlignmentClient,
    ASRClient,
    SpkSimClient,
    ServiceError,
    WERClient,
    WDTWClient,
)
from speech_edit.data.language import alignment_language_code, classify_text_language

from .synthesizer import SynthesizedPair
from .tag_utils import (
    PauseTag,
    tags_to_aligner_queries,
)

logger = logging.getLogger(__name__)

DEFAULT_EVAL_ALIGNER_BACKEND = "qwen3"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

WER_THRESHOLD = 0.05  # 5% — F5 editing may slightly affect adjacent words


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PauseEvalResult:
    """Evaluation result for a single pause edit pair."""

    profile_id: int

    # Content evaluation
    original_wer: float
    generated_wer: float
    content_passed: bool

    # Pause accuracy evaluation (per operation)
    pause_measurements: List[Dict]  # Per-tag measurements

    metadata: Dict = field(default_factory=dict)
    original_utmos: Optional[Dict[str, Any]] = None
    generated_utmos: Optional[Dict[str, Any]] = None
    spk_sim: Optional[Dict[str, Any]] = None
    wdtw_result: Optional[Dict[str, Any]] = None


def _content_payload(
    *,
    transcript: str,
    wer: Dict[str, Any],
    utmos: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "transcript": transcript,
        "wer": wer,
    }
    if isinstance(utmos, dict) and utmos:
        payload["utmos"] = dict(utmos)
    return payload


def _pause_operations_for_direction(
    measurements: List[Dict],
    *,
    direction: str,
) -> List[Dict[str, Any]]:
    sign = -1.0 if direction == "generated_to_original" else 1.0
    operations: List[Dict[str, Any]] = []
    for measurement in measurements:
        operation: Dict[str, Any] = {}
        if measurement.get('measurement_error'):
            raise ValueError("Cannot measure requested pause: missing anchors or aligned gaps")

        expected = _safe_float(measurement.get("target_added_pause_ms"))
        observed = _safe_float(measurement.get("actual_added_pause_ms"))
        if expected is not None:
            operation["expected_tgt_minus_src_pause_gap_ms"] = round(sign * expected, 1)
        if observed is not None:
            operation["measured_tgt_minus_src_pause_gap_ms"] = round(sign * observed, 1)
        if expected is not None and observed is not None:
            operation["error_tgt_minus_src_pause_gap_ms"] = round(
                sign * observed - sign * expected,
                1,
            )
        operations.append(operation)
    return operations


def serialize_eval_results(
    eval_result: Optional[PauseEvalResult],
    *,
    direction: str = "original_to_generated",
) -> Dict[str, object]:
    if eval_result is None:
        return {}
    metadata = dict(eval_result.metadata or {})
    original_payload = _content_payload(
        transcript=str(metadata.pop("original_transcription", "")),
        wer=metadata.pop("original_wer_result"),
        utmos=eval_result.original_utmos,
    )
    generated_payload = _content_payload(
        transcript=str(metadata.pop("generated_transcription", "")),
        wer=metadata.pop("generated_wer_result"),
        utmos=eval_result.generated_utmos,
    )
    if direction == "generated_to_original":
        src_payload = generated_payload
        tgt_payload = original_payload
    else:
        src_payload = original_payload
        tgt_payload = generated_payload

    payload: Dict[str, object] = {
        "src": src_payload,
        "tgt": tgt_payload,
        "pause": {
            "operations": _pause_operations_for_direction(
                eval_result.pause_measurements,
                direction=direction,
            )
        },
    }
    if eval_result.spk_sim is not None:
        payload["spk_sim"] = dict(eval_result.spk_sim)
    if eval_result.wdtw_result is not None:
        payload["wdtw_result"] = dict(eval_result.wdtw_result)
        if isinstance(eval_result.wdtw_result, dict) and "wdtw" in eval_result.wdtw_result:
            payload["wdtw"] = eval_result.wdtw_result.get("wdtw")
    if metadata:
        payload["evaluation_metadata"] = metadata
    return payload


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# WER evaluation
# ---------------------------------------------------------------------------


def evaluate_wer(normed_text: str, audio_bytes: bytes, asr_client: ASRClient, wer_client: WERClient, *, language: str = "auto") -> Dict:
    """Collect content accuracy; service failures propagate."""
    if asr_client is None or wer_client is None:
        raise ValueError("ASR and WER clients are required")
    transcription = asr_client.transcribe(audio_bytes)["text"]
    result = compute_wer(normed_text, transcription, wer_client=wer_client, language=language)
    return {"transcription": transcription, "wer": result["error_rate"], "wer_result": result, "passed": result["error_rate"] <= WER_THRESHOLD}








def evaluate_wdtw(
    synth_pair: SynthesizedPair,
    *,
    wdtw_client: Optional[WDTWClient],
    sample_language: Optional[str],
    wdtw_mode: str = "preserved",
) -> Optional[Dict[str, Any]]:
    if wdtw_client is None:
        return None

    original_alignments = synth_pair.original_word_alignments or None

    return wdtw_client.compute(
        source_audio=synth_pair.original_audio_bytes,
        target_audio=synth_pair.generated_audio_bytes,
        source_transcript=synth_pair.normed_text,
        target_transcript=synth_pair.generated_text,
        edit_st_inst=synth_pair.profile.operation,
        preservation_st_inst=synth_pair.profile.operation,
        source_alignments=original_alignments,
        target_alignments=None,
        language=alignment_language_code(
            sample_language,
            text=synth_pair.generated_text,
        ),
        mode=wdtw_mode,
        include_segments=False,
    )


# ---------------------------------------------------------------------------
# Pause accuracy evaluation
# ---------------------------------------------------------------------------


def _find_word_in_text(text: str, word: str, near_pos: int = 0) -> int:
    """Find the best occurrence of *word* in *text* near *near_pos*.

    Returns the start index of the word, or 0 if not found.
    """
    if not word:
        return 0
    # Search near near_pos first
    start = max(0, near_pos - 30)
    end = min(len(text), near_pos + 30)
    idx = text.find(word, start, end)
    if idx >= 0:
        return idx
    # Fall back to global search
    idx = text.find(word)
    if idx >= 0:
        return idx
    return 0


def _build_query_for_text(text: str, word: str, near_pos: int = 0) -> str:
    """Build an aligner query by wrapping *word* with <q> tags inside *text*.

    Searches near *near_pos* first to handle cases where the word appears
    multiple times (e.g. after pause-marker insertion shifts text).
    """
    if not word:
        return text
    start = max(0, near_pos - 20)
    end = min(len(text), near_pos + 20)
    idx = text.find(word, start, end)
    if idx < 0:
        idx = text.find(word)
    if idx < 0:
        return text
    return text[:idx] + f"<q>{word}</q>" + text[idx + len(word):]


def _aligner_language_code(text: str) -> Optional[str]:
    language = classify_text_language(text)
    return {
        "en": "eng",
        "zh": "zho",
    }.get(language)


def _target_pause_delta_ms(tag: PauseTag) -> int:
    """Return the signed target gap change for one pause instruction."""
    if tag.act == "ins":
        return tag.delta_dur_ms
    if tag.act == "red":
        return -tag.delta_dur_ms
    raise ValueError(f"unsupported pause edit action: {tag.act!r}")


def evaluate_pause_accuracy(
    synth_pair: SynthesizedPair,
    alignment_client: AlignmentClient,
    *,
    aligner_backend: str = DEFAULT_EVAL_ALIGNER_BACKEND,
) -> List[Dict]:
    """Evaluate pause accuracy at each edit position.

    Uses aligner queries to measure the gap between left and right
    anchor words in both original and generated audio.

    Args:
        synth_pair: The synthesized original/generated audio pair.
        alignment_client: Aligner service client.

    Returns:
        List of per-operation measurement dicts.
    """
    measurements = []
    normed_text = synth_pair.normed_text
    generated_text = synth_pair.generated_text
    tags = synth_pair.profile.tags
    aligner_language = _aligner_language_code(normed_text)

    # Build aligner queries for each tag position (based on normed_text)
    query_specs = tags_to_aligner_queries(normed_text, tags)

    for i, spec in enumerate(query_specs):
        tag = tags[i]
        target_added_pause_ms = _target_pause_delta_ms(tag)
        left_word = spec["left_query"]
        right_word = spec["right_query"]
        char_idx = spec["char_idx"]

        if not left_word or not right_word:
            raise ValueError("Cannot measure requested pause: missing anchors or aligned gaps")

        # Original queries (based on normed_text)
        original_queries = [spec["left_query_full"], spec["right_query_full"]]

        # Generated queries are rebuilt on generated_text.
        generated_left_idx = _find_word_in_text(generated_text, left_word, char_idx)
        generated_right_idx = _find_word_in_text(generated_text, right_word, char_idx)
        generated_left = _build_query_for_text(generated_text, left_word, generated_left_idx)
        generated_right = _build_query_for_text(generated_text, right_word, generated_right_idx)
        generated_queries = [generated_left, generated_right]

        # Measure original audio
        original_resp = alignment_client.align(
            audio_bytes=synth_pair.original_audio_bytes,
            text=normed_text,
            language=aligner_language,
            backend=aligner_backend,
            queries=original_queries,
            return_response=True,
        )
        original_answers = original_resp.get("answers", [])

        # Measure generated audio
        generated_resp = alignment_client.align(
            audio_bytes=synth_pair.generated_audio_bytes,
            text=generated_text,
            language=aligner_language,
            backend=aligner_backend,
            queries=generated_queries,
            return_response=True,
        )
        generated_answers = generated_resp.get("answers", [])

        # Extract gap measurements
        original_gap = _extract_gap(original_answers)
        generated_gap = _extract_gap(generated_answers)

        if original_gap is None or generated_gap is None:
            raise ValueError("Cannot measure requested pause: missing anchors or aligned gaps")

        # Compute pause metrics
        original_gap_ms = original_gap * 1000
        generated_gap_ms = generated_gap * 1000
        actual_added_pause_ms = generated_gap_ms - original_gap_ms

        measurement = {
            "tag_index": i,
            "edit_action": tag.act,
            "target_added_pause_ms": target_added_pause_ms,
            "original_gap_ms": round(original_gap_ms, 1),
            "generated_gap_ms": round(generated_gap_ms, 1),
            "actual_added_pause_ms": round(actual_added_pause_ms, 1),
            "direction_correct": target_added_pause_ms * actual_added_pause_ms > 0.0,
        }

        target_vs_actual_gap_error_ms = abs(actual_added_pause_ms - target_added_pause_ms)
        measurement["target_vs_actual_gap_error_ms"] = round(target_vs_actual_gap_error_ms, 1)

        measurements.append(measurement)

    return measurements


def _extract_gap(answers: List[Dict]) -> Optional[float]:
    """Extract the gap between two anchor word answers.

    The gap is: right_word_start - left_word_end.

    Args:
        answers: List of aligner answer dicts (expected 2 elements).

    Returns:
        Gap in seconds, or None if extraction fails.
    """
    if len(answers) < 2:
        return None

    left_answer = answers[0]
    right_answer = answers[1]

    left_end = left_answer.get("end")
    right_start = right_answer.get("start")

    if left_end is None or right_start is None:
        return None

    gap = right_start - left_end
    return max(0.0, gap)  # Gap should be non-negative


def _ms(value: Optional[float]) -> Optional[float]:
    """Convert seconds to ms, or None."""
    return round(value * 1000, 1) if value is not None else None


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_pause_edit(
    synth_pair: SynthesizedPair,
    asr_client: ASRClient,
    alignment_client: AlignmentClient,
    *,
    wer_client: Optional[WERClient] = None,
    aligner_backend: str = DEFAULT_EVAL_ALIGNER_BACKEND,
    utmos_client: Any = None,
    spk_sim_client: Optional[SpkSimClient] = None,
    wdtw_client: Optional[WDTWClient] = None,
    sample_language: Optional[str] = None,
    wdtw_mode: str = "preserved",
) -> PauseEvalResult:
    """Evaluate a synthesized pause edit pair.

    Args:
        synth_pair: The synthesized original/generated audio pair.
        asr_client: ASR service client.
        alignment_client: Aligner service client.
        wer_client: WER/CER service client.

    Returns:
        PauseEvalResult with all evaluation metrics.
    """
    profile = synth_pair.profile
    normed_text = synth_pair.normed_text

    # Step 1: WER evaluation on both original and generated audio
    original_wer_result = evaluate_wer(
        normed_text,
        synth_pair.original_audio_bytes,
        asr_client,
        wer_client,
        language="auto",
    )
    generated_ref_text = synth_pair.generated_text
    generated_wer_result = evaluate_wer(
        generated_ref_text,
        synth_pair.generated_audio_bytes,
        asr_client,
        wer_client,
        language="auto",
    )

    content_passed = (
        original_wer_result["passed"] and generated_wer_result["passed"]
    )

    # Step 2: Pause accuracy evaluation
    pause_measurements = evaluate_pause_accuracy(
        synth_pair,
        alignment_client,
        aligner_backend=aligner_backend,
    )

    original_utmos = (
        evaluate_quality(synth_pair.original_audio_bytes, utmos_client)
        if utmos_client is not None
        else None
    )
    generated_utmos = (
        evaluate_quality(synth_pair.generated_audio_bytes, utmos_client)
        if utmos_client is not None
        else None
    )
    spk_sim = evaluate_speaker_similarity(
        synth_pair.original_audio_bytes,
        synth_pair.generated_audio_bytes,
        spk_sim_client,
    )
    wdtw_result = evaluate_wdtw(
        synth_pair,
        wdtw_client=wdtw_client,
        sample_language=sample_language,
        wdtw_mode=wdtw_mode,
    )

    return PauseEvalResult(
        profile_id=profile.profile_id,
        original_wer=original_wer_result["wer"],
        generated_wer=generated_wer_result["wer"],
        content_passed=content_passed,
        pause_measurements=pause_measurements,
        metadata={
            "original_transcription": original_wer_result.get("transcription", ""),
            "generated_transcription": generated_wer_result.get("transcription", ""),
            "original_wer_result": original_wer_result.get("wer_result"),
            "generated_wer_result": generated_wer_result.get("wer_result"),
            "aligner_backend": aligner_backend,
        },
        original_utmos=original_utmos,
        generated_utmos=generated_utmos,
        spk_sim=spk_sim,
        wdtw_result=wdtw_result,
    )
