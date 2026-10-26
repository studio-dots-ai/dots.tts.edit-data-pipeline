"""
Audio editor for pitch/rate editing pipeline.

Workflow
--------
1.  Build exact `<q>...</q>` queries for each tagged edit region.
2.  Ask the aligner gateway for `queries -> answers` using the `qwen3` backend.
3.  Slice source audio at the returned timestamps.
4.  Apply pitch_shift / stretch to tagged word segments via HTTP services:
    - Pitch: ``psola+qwen3_codec`` or ``psola+indextts2_cco`` via
      Pitch Conversion Service (port 8027)
    - Rate:  ``wsola+indextts2`` via Speech Rate Conversion Service (port 8004)
5.  Reassemble with short crossfade at boundaries to avoid clicks.
6.  (Optional) Smooth splice boundaries via F5-TTS mask-based inpainting.

The module uses HTTP service clients for pitch and rate conversion (which
internally call Qwen3-TTS and IndexTTS2 for codec refinement) and the
**AlignmentClient** HTTP gateway for forced alignment. When an ``F5TTSClient``
is provided, step 6 regenerates a small margin around each splice boundary
using F5-TTS, producing natural-sounding transitions.
"""

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .alignment import align_queries
from .tag_utils import (
    PITCH_TAG,
    RATE_TAG,
    TaggedSegment,
    parse_tags,
    strip_tags,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24000  # All processing at 24 kHz
PITCH_METHOD_CHOICES = ("psola+qwen3_codec", "psola+indextts2_cco")
RATE_METHOD_CHOICES = ("wsola", "indextts2", "qwen3_tts", "wsola+indextts2")
SMOOTH_POLICY_CHOICES = ("auto", "always", "never")
INDEXTTS_SR = 22050
QWEN3_SR = 24000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EditResult:
    """Result of applying edits to source audio."""

    edited_audio: np.ndarray  # float32 mono array at TARGET_SR
    sample_rate: int
    edits_applied: List[Dict]  # metadata about each edit
    edit_time_sec: float  # wall-clock time for editing
    boundary_times: List[float] = field(
        default_factory=list
    )  # splice boundary positions (sec)
    smoothing_applied: bool = False  # whether F5-TTS boundary smoothing was used
    alignment_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformResult:
    """Audio transform output plus sample-rate metadata."""

    audio: np.ndarray
    sample_rate: int
    metadata: Dict[str, Any]


@dataclass
class EditRegion:
    """One contiguous source-audio region that should be edited."""

    text: str
    start_sec: float
    end_sec: float
    segment: TaggedSegment


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def wav_to_numpy(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Convert WAV bytes to float32 mono numpy array."""
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo to mono
    return audio, sr


def numpy_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to WAV bytes (PCM 16-bit)."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Alignment query helpers
# ---------------------------------------------------------------------------


def _collect_tagged_queries(tagged_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Build exact-span queries for each tagged segment in plain-text order."""

    plain_text = strip_tags(tagged_text)
    queries: List[Dict[str, Any]] = []
    pos = 0

    for seg in parse_tags(tagged_text):
        start_char = pos
        end_char = pos + len(seg.text)
        pos = end_char
        if not seg.is_tagged:
            continue

        leading_ws = len(seg.text) - len(seg.text.lstrip())
        trailing_ws = len(seg.text) - len(seg.text.rstrip())
        query_start = start_char + leading_ws
        query_end = end_char - trailing_ws
        if query_start >= query_end:
            continue

        queries.append(
            {
                "segment": seg,
                "text": plain_text[query_start:query_end],
                "start_char": query_start,
                "end_char": query_end,
                "query": (
                    f"{plain_text[:query_start]}<q>"
                    f"{plain_text[query_start:query_end]}</q>"
                    f"{plain_text[query_end:]}"
                ),
            }
        )

    return plain_text, queries


def _build_regions_from_alignment_response(
    aligner_backend: str,
    tagged_queries: List[Dict[str, Any]],
    response: Dict[str, Any],
) -> Tuple[List[EditRegion], Dict[str, Any]]:
    answers = response.get("answers")
    if not isinstance(answers, list) or len(answers) != len(tagged_queries):
        raise ValueError(
            "Aligner did not return one answer per tagged region. "
            f"queries={len(tagged_queries)}, answers={answers!r}"
        )

    tokenization_flag = response.get("query_tokenization_matches_default")
    if tokenization_flag is False:
        logger.warning(
            "Alignment backend %s reported query_tokenization_matches_default=false; "
            "using returned query answers as-is.",
            aligner_backend,
        )

    regions: List[EditRegion] = []
    metadata_regions: List[Dict[str, Any]] = []
    for idx, (item, answer) in enumerate(zip(tagged_queries, answers)):
        if not isinstance(answer, dict) or "start" not in answer or "end" not in answer:
            raise ValueError(
                f"Invalid answer for tagged region {idx}: expected start/end, got {answer!r}"
            )

        start_sec = float(answer["start"])
        end_sec = float(answer["end"])
        if end_sec < start_sec:
            raise ValueError(
                f"Invalid answer for tagged region {idx}: start={start_sec}, end={end_sec}"
            )

        segment = item["segment"]
        regions.append(
            EditRegion(
                text=item["text"],
                start_sec=start_sec,
                end_sec=end_sec,
                segment=segment,
            )
        )
        metadata_regions.append(
            {
                "query_index": idx,
                "query": item["query"],
                "text": item["text"],
                "start_char": item["start_char"],
                "end_char": item["end_char"],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "tag_type": segment.tag_type,
                "semitones": segment.semitones,
                "factor": segment.factor,
            }
        )

    return regions, {
        "backend": str(response.get("backend") or aligner_backend),
        "language": response.get("language"),
        "level": response.get("level"),
        "query_tokenization_matches_default": tokenization_flag,
        "alignments": response.get("alignments"),
        "edit_regions": metadata_regions,
    }


# ---------------------------------------------------------------------------
# Sample-rate policy helpers
# ---------------------------------------------------------------------------


def resolve_working_sample_rate(
    tagged_text: str,
    *,
    source_sample_rate: int,
    pitch_method: str,
    rate_method: str,
) -> int:
    """Resolve the utterance-level sample rate for pitch/rate editing.

    Prosody editing runs on a single 24 kHz source timeline. Service-specific
    sample-rate requirements are handled only for replacement segments after
    the service returns.
    """
    segments = parse_tags(tagged_text)
    has_pitch = any(seg.tag_type == PITCH_TAG for seg in segments)
    has_rate = any(seg.tag_type == RATE_TAG for seg in segments)

    if has_pitch:
        if pitch_method not in {"psola+qwen3_codec", "psola+indextts2_cco", "psola"}:
            raise ValueError(f"Unsupported pitch_method: {pitch_method!r}")

    if has_rate:
        if rate_method not in {"wsola+indextts2", "indextts2", "qwen3_tts", "wsola"}:
            raise ValueError(f"Unsupported rate_method: {rate_method!r}")

    return TARGET_SR


def _resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return _resample(audio, orig_sr, target_sr)


def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or pad an audio segment to exactly target_len samples."""
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        return audio[:target_len].copy()
    return np.pad(audio, (0, target_len - len(audio)), mode="constant").astype(
        np.float32,
        copy=False,
    )


def parse_psola_indextts2_cco_pitch_boost(raw: str) -> Tuple[str, float]:
    """Parse the empirical IndexTTS2 CCO pitch boost policy."""
    value = (raw or "none").strip().lower()
    if value == "none":
        return "none", 0.0
    if ":" not in value:
        raise ValueError(
            "psola_indextts2_cco_pitch_boost must be 'none', 'add:<value>', or 'mul:<value>'"
        )
    mode, amount_text = value.split(":", 1)
    if mode not in {"add", "mul"}:
        raise ValueError(
            "psola_indextts2_cco_pitch_boost mode must be 'add' or 'mul'"
        )
    try:
        amount = float(amount_text)
    except ValueError as exc:
        raise ValueError(
            "psola_indextts2_cco_pitch_boost value must be numeric"
        ) from exc
    if mode == "add" and amount < 0:
        raise ValueError("additive pitch boost must be non-negative")
    if mode == "mul" and amount <= 0:
        raise ValueError("multiplicative pitch boost must be positive")
    return mode, amount


def resolve_psola_indextts2_cco_effective_semitones(
    semitones: float,
    *,
    pitch_method: str,
    psola_indextts2_cco_pitch_boost: str,
) -> Tuple[float, Dict[str, Any]]:
    """Return service semitones after optional IndexTTS2 CCO pitch boost."""
    requested = float(semitones)
    mode, amount = parse_psola_indextts2_cco_pitch_boost(
        psola_indextts2_cco_pitch_boost
    )
    effective = requested
    boost_applied = False
    if pitch_method == "psola+indextts2_cco" and mode != "none":
        boost_applied = True
        if mode == "add":
            if requested > 0:
                effective = requested + amount
            elif requested < 0:
                effective = requested - amount
            else:
                effective = 0.0
        elif mode == "mul":
            effective = requested * amount
    clamped = max(-24.0, min(24.0, effective))
    return clamped, {
        "requested_semitones": requested,
        "effective_semitones": clamped,
        "psola_indextts2_cco_pitch_boost": psola_indextts2_cco_pitch_boost,
        "boost_mode": mode,
        "boost_value": amount,
        "boost_applied": boost_applied,
        "boost_clamped": clamped != effective,
    }


# ---------------------------------------------------------------------------
# Segment-level audio transforms (via HTTP services)
# ---------------------------------------------------------------------------


def _run_pitch_shift(
    audio: np.ndarray,
    sr: int,
    semitones: float,
    *,
    pitch_client=None,
    text: Optional[str] = None,
    pitch_method: str,
    psola_indextts2_cco_pitch_boost: str,
) -> TransformResult:
    """Apply the configured transform; service failures propagate to the caller."""
    if pitch_client is None:
        raise ValueError("pitch_client is required")
    effective_semitones, boost_metadata = resolve_psola_indextts2_cco_effective_semitones(semitones, pitch_method=pitch_method, psola_indextts2_cco_pitch_boost=psola_indextts2_cco_pitch_boost)
    segment_wav = numpy_to_wav(audio, sr)
    result_wav = pitch_client.shift(
        semitones=effective_semitones,
        audio_bytes=segment_wav,
        method=pitch_method,
        text=text,
        output_format="wav",
    )
    result_audio, result_sr = wav_to_numpy(result_wav)
    resampled_to_target_sr = result_sr != sr
    result_audio = _resample_if_needed(result_audio, result_sr, sr)
    return TransformResult(result_audio, sr, {
        "name": pitch_method,
        "service_output_sample_rate": result_sr,
        "working_sample_rate": sr,
        "target_sample_rate": sr,
        "resampled_to_working_sample_rate": resampled_to_target_sr,
        "resampled_to_target_sample_rate": resampled_to_target_sr,
        **boost_metadata,
    })


def _run_rate_stretch(
    audio: np.ndarray,
    sr: int,
    factor: float,
    *,
    rate_client=None,
    text: Optional[str] = None,
    spk_audio_bytes: Optional[bytes] = None,
    rate_method: str,
) -> TransformResult:
    """Apply the configured transform; service failures propagate to the caller."""
    if rate_client is None:
        raise ValueError("rate_client is required")
    if text is None or spk_audio_bytes is None:
        raise ValueError("text and spk_audio_bytes are required")
    segment_wav = numpy_to_wav(audio, sr)
    result_wav = rate_client.stretch(
        rate=factor,
        method=rate_method,
        audio_bytes=segment_wav,
        text=text,
        spk_audio_bytes=spk_audio_bytes,
        output_format="wav",
    )
    result_audio, result_sr = wav_to_numpy(result_wav)
    resampled_to_target_sr = result_sr != sr
    result_audio = _resample_if_needed(result_audio, result_sr, sr)
    return TransformResult(result_audio, sr, {
        "name": rate_method,
        "service_output_sample_rate": result_sr,
        "working_sample_rate": sr,
        "target_sample_rate": sr,
        "resampled_to_working_sample_rate": resampled_to_target_sr,
        "resampled_to_target_sample_rate": resampled_to_target_sr,
    })


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using librosa."""
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


# ---------------------------------------------------------------------------
# Splice-in-place reassembly
# ---------------------------------------------------------------------------


def _blend_replacement_edges(
    original: np.ndarray,
    start: int,
    end: int,
    replacement: np.ndarray,
    crossfade_samples: int,
    *,
    has_left_boundary: bool,
    has_right_boundary: bool,
) -> np.ndarray:
    """Blend replacement edges against the original samples they replace.

    The blend is duration-preserving: it modifies samples inside
    ``replacement`` only, so no samples are removed from either the preserved
    original context or the replacement.  At the left boundary, the replacement
    head is blended against ``original[start:start+L]``.  At the right boundary,
    the replacement tail is blended against ``original[end-L:end]``.
    """
    if crossfade_samples <= 0 or len(replacement) == 0:
        return replacement

    window_len = max(0, end - start)
    if window_len <= 0:
        return replacement

    left_len = (
        min(crossfade_samples, len(replacement), window_len)
        if has_left_boundary
        else 0
    )
    right_len = (
        min(crossfade_samples, len(replacement), window_len)
        if has_right_boundary
        else 0
    )

    if left_len + right_len > len(replacement):
        # Keep the two boundary blends from overlapping in very short regions.
        half = len(replacement) // 2
        left_len = min(left_len, half)
        right_len = min(right_len, len(replacement) - left_len)

    blended = replacement.copy()

    if left_len > 0:
        fade_original = np.linspace(1.0, 0.0, left_len, dtype=np.float32)
        fade_replacement = np.linspace(0.0, 1.0, left_len, dtype=np.float32)
        original_head = original[start : start + left_len]
        blended[:left_len] = (
            original_head * fade_original + blended[:left_len] * fade_replacement
        )

    if right_len > 0:
        fade_replacement = np.linspace(1.0, 0.0, right_len, dtype=np.float32)
        fade_original = np.linspace(0.0, 1.0, right_len, dtype=np.float32)
        original_tail = original[end - right_len : end]
        blended[-right_len:] = (
            blended[-right_len:] * fade_replacement + original_tail * fade_original
        )

    return blended


def _splice_replacements(
    original: np.ndarray,
    replacements: List[Tuple[int, int, np.ndarray]],
    sample_rate: int,
    crossfade_ms: float,
) -> Tuple[np.ndarray, List[float]]:
    """Splice edited segments into the original audio at their positions.

    Preserves all original audio outside the replacement windows.
    Applies a duration-preserving boundary blend at each splice boundary.

    Args:
        original: The full original audio array.
        replacements: Sorted list of (start_sample, end_sample, new_audio).
            Must be non-overlapping and sorted by start_sample.
        sample_rate: Sample rate for converting positions to seconds.
        crossfade_ms: Number of milliseconds inside the replacement edge to
            blend against the corresponding original audio.  The output length
            is not shortened.

    Returns:
        Tuple of:
        - New audio array with replacements spliced in.
        - List of boundary times (seconds) where edited regions meet
          unedited regions.  Each replacement can produce up to 2
          boundaries (left: original→edited, right: edited→original).
          Boundaries at the very start (0.0s) or very end of the audio
          are excluded since there's no transition to smooth.
    """
    cf = max(0, int(crossfade_ms / 1000.0 * sample_rate))
    chunks: List[np.ndarray] = []
    boundary_samples: List[int] = []
    pos = 0

    def _current_output_len() -> int:
        """Total samples accumulated so far across all chunks."""
        return sum(len(c) for c in chunks)

    previous_end: Optional[int] = None
    sorted_replacements = sorted(replacements, key=lambda r: r[0])
    for start, end, new_seg in sorted_replacements:
        if start < pos:
            raise ValueError(
                "Replacement windows must be sorted and non-overlapping: "
                f"got start={start}, current_pos={pos}"
            )

        # 1. Copy original audio from current position up to the edit start.
        if start > pos:
            chunks.append(original[pos:start].copy())

        # 2. Blend only inside the replacement window, then splice it in.
        blend_left = start > 0
        record_left_boundary = start > 0 and previous_end != start
        right_boundary = end < len(original)

        if record_left_boundary:
            boundary_samples.append(_current_output_len())

        blended_seg = _blend_replacement_edges(
            original,
            start,
            end,
            new_seg,
            cf,
            has_left_boundary=blend_left,
            has_right_boundary=right_boundary,
        )
        chunks.append(blended_seg)

        if right_boundary:
            boundary_samples.append(_current_output_len())

        pos = end
        previous_end = end

    # 3. Copy remaining original audio after the last edit.
    if pos < len(original):
        chunks.append(original[pos:].copy())

    if not chunks:
        return original.copy(), []

    result = np.concatenate(chunks)

    # Convert boundary sample positions to seconds and filter out
    # boundaries that sit at the very start or end (nothing to smooth).
    total_samples = len(result)
    boundary_times: List[float] = []
    for bs in boundary_samples:
        if bs <= 0 or bs >= total_samples:
            continue
        boundary_times.append(bs / sample_rate)

    return result, boundary_times


# ---------------------------------------------------------------------------
# F5-TTS boundary smoothing
# ---------------------------------------------------------------------------


def smooth_splice_boundaries(
    spliced_audio: np.ndarray,
    sr: int,
    plain_text: str,
    boundary_times: List[float],
    f5_client,  # F5TTSClient instance
    *,
    margin_sec: float,
    source_crossfade_sec: float,
    regen_crossfade_sec: float,
    f5_steps: int,
    seed: Optional[int],
) -> Tuple[np.ndarray, int]:
    """Smooth splice boundaries using F5-TTS mask-based inpainting.

    After splice-in-place reassembly, the junctions between edited and
    unedited audio can sound artificial (especially for prominent pitch
    or rate changes).  This function uses F5-TTS to regenerate a small
    margin around each boundary, producing natural-sounding transitions.

    Args:
        spliced_audio: The spliced audio array (float32 mono, working sample rate).
        sr: Sample rate.
        plain_text: The full spoken text (no tags).
        boundary_times: Boundary positions in seconds (from
            ``_splice_replacements``).
        f5_client: An ``F5TTSClient`` instance connected to the F5-TTS
            service.
        margin_sec: Seconds of audio to regenerate on each side of a boundary.
        source_crossfade_sec: F5 preserve splice crossfade outside smoothing spans.
        regen_crossfade_sec: F5 preserve splice crossfade inside smoothing spans.
        f5_steps: Number of F5-TTS denoising steps.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of smoothed audio and the original sample rate.
    """
    if not boundary_times:
        logger.info("No boundaries to smooth — skipping F5-TTS smoothing")
        return spliced_audio, sr
    if sr != TARGET_SR:
        raise ValueError(
            f"F5-TTS smoothing expects {TARGET_SR} Hz input, got {sr} Hz"
        )

    logger.info(
        "Smoothing %d boundaries via F5-TTS (margin=%.2fs, steps=%d)",
        len(boundary_times),
        margin_sec,
        f5_steps,
    )

    t0 = time.time()
    smoothed_wav_bytes = f5_client.smooth_boundaries(
        waveform=spliced_audio,
        sample_rate=sr,
        text=plain_text,
        boundary_times=boundary_times,
        margin_sec=margin_sec,
        source_crossfade_sec=source_crossfade_sec,
        regen_crossfade_sec=regen_crossfade_sec,
        steps=f5_steps,
        seed=seed,
    )

    smoothed_audio, smoothed_sr = wav_to_numpy(smoothed_wav_bytes)
    if smoothed_sr != sr:
        raise ValueError(
            f"F5-TTS returned sr={smoothed_sr}, expected input sr={sr}"
        )

    elapsed = time.time() - t0
    logger.info(
        "F5-TTS smoothing done in %.2fs (%.2fs → %.2fs)",
        elapsed,
        len(spliced_audio) / sr,
        len(smoothed_audio) / sr,
    )

    return smoothed_audio, sr


def _apply_regions(
    audio_bytes: bytes,
    plain_text: str,
    regions: List[EditRegion],
    sample_rate: int = TARGET_SR,
    *,
    alignment_metadata: Optional[Dict[str, Any]] = None,
    pitch_client=None,
    rate_client=None,
    pitch_method: str,
    rate_method: str,
    psola_indextts2_cco_pitch_boost: str,
    crossfade_ms: float,
    f5_client=None,
    smooth_margin_sec: float,
    f5_steps: int,
    f5_seed: Optional[int],
    f5_source_crossfade_sec: float,
    f5_regen_crossfade_sec: float,
    source_sample_rate: Optional[int] = None,
) -> EditResult:
    t0 = time.time()

    audio, sr = wav_to_numpy(audio_bytes)
    if sr != sample_rate:
        raise ValueError(
            f"Audio sample rate {sr} != expected {sample_rate}. "
            "The prosody pipeline expects audio_bytes to already be at the "
            "normalized target sample rate."
        )
    if sr != TARGET_SR:
        raise ValueError(
            f"Prosody edit regions must be applied on {TARGET_SR} Hz audio, got {sr}"
        )

    edits_applied: List[Dict] = []
    replacements: List[Tuple[int, int, np.ndarray]] = []

    for region in regions:
        tag_seg = region.segment
        start_sec = max(0.0, float(region.start_sec))
        end_sec = max(start_sec, float(region.end_sec))
        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(audio), int(end_sec * sr))
        if end_sample <= start_sample:
            raise ValueError(
                f"Empty edit region for {region.text!r}: "
                f"{start_sec:.3f}s -> {end_sec:.3f}s"
            )

        segment_audio = audio[start_sample:end_sample].copy()
        word_text = region.text

        if tag_seg.tag_type == PITCH_TAG and tag_seg.semitones is not None:
            logger.info(
                "Pitch shift '%s': %+.1f semitones",
                word_text,
                tag_seg.semitones,
            )
            pitch_result = _run_pitch_shift(
                segment_audio,
                sr,
                tag_seg.semitones,
                pitch_client=pitch_client,
                text=word_text,
                pitch_method=pitch_method,
                psola_indextts2_cco_pitch_boost=psola_indextts2_cco_pitch_boost,
            )
            pitch_method_metadata = dict(pitch_result.metadata)
            length_delta_samples = len(pitch_result.audio) - len(segment_audio)
            edited_segment = _match_length(pitch_result.audio, len(segment_audio))
            pitch_method_metadata.update(
                {
                    "source_window_samples": len(segment_audio),
                    "replacement_samples_before_length_normalization": len(pitch_result.audio),
                    "length_delta_samples_before_normalization": int(length_delta_samples),
                    "length_normalized_to_source_window": length_delta_samples != 0,
                }
            )
            edits_applied.append(
                {
                    "words": word_text,
                    "edit_type": "pitch",
                    "semitones": tag_seg.semitones,
                    "method": pitch_method_metadata,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_span": [start_sec, end_sec],
                    "edited_duration_sec": len(edited_segment) / sr,
                }
            )
            replacements.append((start_sample, end_sample, edited_segment))

        elif tag_seg.tag_type == RATE_TAG and tag_seg.factor is not None:
            logger.info(
                "Rate stretch '%s': %.2fx",
                word_text,
                tag_seg.factor,
            )
            rate_result = _run_rate_stretch(
                segment_audio,
                sr,
                tag_seg.factor,
                rate_client=rate_client,
                text=word_text,
                spk_audio_bytes=audio_bytes,
                rate_method=rate_method,
            )
            edited_segment = rate_result.audio
            edits_applied.append(
                {
                    "words": word_text,
                    "edit_type": "rate",
                    "factor": tag_seg.factor,
                    "method": rate_result.metadata,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_span": [start_sec, end_sec],
                    "edited_duration_sec": len(edited_segment) / sr,
                }
            )
            replacements.append((start_sample, end_sample, edited_segment))

    boundary_times: List[float] = []
    smoothing_requested = f5_client is not None
    pre_f5_crossfade_ms = 0.0 if smoothing_requested else crossfade_ms
    if not replacements:
        edited_audio = audio.copy()
    else:
        replacements.sort(key=lambda r: r[0])
        edited_audio, boundary_times = _splice_replacements(
            audio,
            replacements,
            sample_rate=sr,
            crossfade_ms=pre_f5_crossfade_ms,
        )

    # Record the edited-region location in the reassembled target timeline.
    # This is needed by parquet rows so consumers do not need to reverse-engineer
    # rate edits from source spans.
    offset_samples = 0
    for edit, (source_start, source_end, replacement) in zip(
        sorted(edits_applied, key=lambda item: float(item["start_sec"])), replacements
    ):
        target_start = source_start + offset_samples
        target_end = target_start + len(replacement)
        edit["target_span"] = [target_start / sr, target_end / sr]
        offset_samples += len(replacement) - (source_end - source_start)

    smoothing_applied = False
    final_sr = sr
    if f5_client is not None and boundary_times:
        edited_audio, final_sr = smooth_splice_boundaries(
            spliced_audio=edited_audio,
            sr=sr,
            plain_text=plain_text,
            boundary_times=boundary_times,
            f5_client=f5_client,
            margin_sec=smooth_margin_sec,
            source_crossfade_sec=f5_source_crossfade_sec,
            regen_crossfade_sec=f5_regen_crossfade_sec,
            f5_steps=f5_steps,
            seed=f5_seed,
        )
        smoothing_applied = True

    edit_time = time.time() - t0
    logger.info(
        "Editing complete: %d edits applied in %.2fs, duration %.2fs -> %.2fs%s",
        len(edits_applied),
        edit_time,
        len(audio) / sr,
        len(edited_audio) / final_sr,
        " (smoothed)" if smoothing_applied else "",
    )
    final_alignment_metadata = dict(alignment_metadata or {})
    final_alignment_metadata.update(
        {
            "source_sample_rate": int(source_sample_rate or sr),
            "normalized_sample_rate": int(TARGET_SR),
            "source_resampled_to_target": bool((source_sample_rate or sr) != TARGET_SR),
            "working_sample_rate": int(sr),
            "final_sample_rate": int(final_sr),
            "pitch_method": pitch_method,
            "rate_method": rate_method,
            "psola_indextts2_cco_pitch_boost": psola_indextts2_cco_pitch_boost,
            "pre_f5_crossfade_ms": float(pre_f5_crossfade_ms),
            "smooth_margin_sec": float(smooth_margin_sec) if smoothing_requested else None,
            "f5_source_crossfade_sec": (
                float(f5_source_crossfade_sec) if smoothing_requested else None
            ),
            "f5_regen_crossfade_sec": (
                float(f5_regen_crossfade_sec) if smoothing_requested else None
            ),
        }
    )

    return EditResult(
        edited_audio=edited_audio,
        sample_rate=final_sr,
        edits_applied=edits_applied,
        edit_time_sec=edit_time,
        boundary_times=boundary_times,
        smoothing_applied=smoothing_applied,
        alignment_metadata=final_alignment_metadata,
    )


# ---------------------------------------------------------------------------
# Main editing entry point
# ---------------------------------------------------------------------------


def apply_edits_from_service(
    audio_bytes: bytes,
    tagged_text: str,
    aligner_client,
    sample_rate: int = TARGET_SR,
    *,
    aligner_backend: str = "qwen3",
    language: str = "eng",
    pitch_client=None,
    rate_client=None,
    pitch_method: str,
    rate_method: str,
    psola_indextts2_cco_pitch_boost: str,
    crossfade_ms: float,
    f5_client=None,
    smooth_margin_sec: float,
    f5_steps: int,
    f5_seed: Optional[int],
    f5_source_crossfade_sec: float,
    f5_regen_crossfade_sec: float,
) -> EditResult:
    """Apply pitch/rate edits using the aligner gateway query-answer flow.

    Args:
        audio_bytes: Source audio WAV bytes.
        tagged_text: Text with <pitch> and <rate> tags.
        aligner_client: An AlignmentClient instance.
        language: Aligner language code (`eng` or `zho`).
        sample_rate: Expected sample rate.
        pitch_client: Optional ``PitchConversionClient`` for pitch shifting.
        pitch_method: Pitch Conversion method to use for pitch edits.
        rate_method: Speech Rate Conversion method to use for rate edits.
        psola_indextts2_cco_pitch_boost: Optional empirical boost for
            psola+indextts2_cco pitch semitones.
        rate_client: Optional ``SpeechRateClient`` for wsola+indextts2
            rate stretching.
        f5_client: Optional ``F5TTSClient`` for boundary smoothing.
        crossfade_ms: Duration-preserving boundary blend duration in milliseconds.
        smooth_margin_sec: Margin around boundaries to regenerate.
        f5_steps: F5-TTS denoising steps.
        f5_seed: Random seed for F5-TTS.

    Returns:
        EditResult with edited audio and metadata.
    """
    source_audio, source_sr = wav_to_numpy(audio_bytes)
    if sample_rate != source_sr:
        logger.warning(
            "Provided sample_rate=%d differs from WAV header sr=%d; using WAV header",
            sample_rate,
            source_sr,
        )
    working_sr = resolve_working_sample_rate(
        tagged_text,
        source_sample_rate=source_sr,
        pitch_method=pitch_method,
        rate_method=rate_method,
    )
    normalized_audio = source_audio
    if source_sr != working_sr:
        logger.info(
            "Normalizing source utterance for prosody editing: %d Hz -> %d Hz "
            "(pitch_method=%s, rate_method=%s). Unedited regions are preserved "
            "from this normalized timeline.",
            source_sr,
            working_sr,
            pitch_method,
            rate_method,
        )
        normalized_audio = _resample(source_audio, source_sr, working_sr)
    audio_bytes = numpy_to_wav(normalized_audio, working_sr)
    sample_rate = working_sr

    plain_text, tagged_queries = _collect_tagged_queries(tagged_text)
    if not tagged_queries:
        return _apply_regions(
            audio_bytes,
            plain_text,
            [],
            sample_rate,
            alignment_metadata={
                "backend": aligner_backend,
                "language": language,
                "level": None,
                "query_tokenization_matches_default": None,
                "alignments": [],
                "edit_regions": [],
            },
            pitch_client=pitch_client,
            rate_client=rate_client,
            pitch_method=pitch_method,
            rate_method=rate_method,
            psola_indextts2_cco_pitch_boost=psola_indextts2_cco_pitch_boost,
            crossfade_ms=crossfade_ms,
            f5_client=f5_client,
            smooth_margin_sec=smooth_margin_sec,
            f5_steps=f5_steps,
            f5_seed=f5_seed,
            f5_source_crossfade_sec=f5_source_crossfade_sec,
            f5_regen_crossfade_sec=f5_regen_crossfade_sec,
            source_sample_rate=source_sr,
        )

    logger.info(
        "Requesting %s query alignment for: %s (%d tagged regions)",
        aligner_backend,
        plain_text[:80],
        len(tagged_queries),
    )
    response, query_metadata = align_queries(
        aligner_client,
        audio_bytes=audio_bytes,
        plain_text=plain_text,
        backend=aligner_backend,
        language=language,
        queries=[item["query"] for item in tagged_queries],
        query_texts=[item["text"] for item in tagged_queries],
    )
    regions, alignment_metadata = _build_regions_from_alignment_response(
        aligner_backend,
        tagged_queries,
        response,
    )
    alignment_metadata.update(query_metadata)

    return _apply_regions(
        audio_bytes,
        plain_text,
        regions,
        sample_rate,
        alignment_metadata=alignment_metadata,
        pitch_client=pitch_client,
        rate_client=rate_client,
        pitch_method=pitch_method,
        rate_method=rate_method,
        psola_indextts2_cco_pitch_boost=psola_indextts2_cco_pitch_boost,
        crossfade_ms=crossfade_ms,
        f5_client=f5_client,
        smooth_margin_sec=smooth_margin_sec,
        f5_steps=f5_steps,
        f5_seed=f5_seed,
        f5_source_crossfade_sec=f5_source_crossfade_sec,
        f5_regen_crossfade_sec=f5_regen_crossfade_sec,
        source_sample_rate=source_sr,
    )
