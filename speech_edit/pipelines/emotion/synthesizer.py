"""
Stage II: Speech Synthesis for emotion/style editing pipeline.

Routes tagged segments to IndexTTS2 or TED-TTS,
synthesizes per-segment audio with voice cloning, concatenates and smooths
boundaries, and assembles editing pairs where unchanged segments are preserved.

Three synthesis modes:
  - single:  Full text → one selected backend.
  - multi_normal:  Per-segment independent synthesis → concat → Boundary Pause Smoother.
  - multi_frequent: TED-TTS whole tagged text path.
"""

import io
import hashlib
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from speech_edit.clients.services import (
    AlignmentClient,
    AudioAssemblyClient,
    BoundaryPauseSmootherClient,
    IndexTTSClient,
    ServiceError,
    TedTTSClient,
)

from .tag_utils import (
    EMOTION_CONTROL_DESCRIPTION,
    NEUTRAL_TAG,
    TaggedSegment,
    TaggedTextGroup,
    build_segment_queries,
    build_segment_timestamps_from_answers,
    ensure_fully_tagged,
    describe_tag_diff,
    get_default_alpha,
    parse_tags,
    render_emo_tag,
    strip_tags,
)
from .synthesis_profile import (
    SYNTHESIS_BACKEND_INDEX,
    SYNTHESIS_BACKEND_TED,
    SYNTHESIS_MODE_MULTI_FREQUENT,
    SYNTHESIS_MODE_MULTI_NORMAL,
    SYNTHESIS_MODE_SINGLE,
    TED_TAGGED_DURATION_MAX_MEL_TOKENS,
    TED_TAGGED_DURATION_MAX_SEC,
    normalize_synthesis_backend,
    normalize_synthesis_profile,
    ted_max_mel_tokens_from_duration_sec,
    validate_tagged_texts_for_profile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24000  # Normalize everything to 24kHz for downstream processing
TED_MEL_TOKEN_SECONDS = 0.02
TED_DURATION_MODE = "both"
TED_EOS_CONTROL_POLICY = "attention_gated_delayed"
TED_EOS_GRACE_SECONDS = 1.0

# Emotion vectors: 8-dim one-hot [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMO_PRESETS = {
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "angry": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "afraid": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "disgusted": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "melancholic": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "calm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

SILENCE_GAP_MS = 50  # Silence gap between segments for independent mode


def _aligner_language_code(language: Optional[str]) -> Optional[str]:
    """Map pipeline language labels to aligner gateway language codes."""
    if language is None:
        return None
    normalized = str(language).strip().lower()
    if not normalized:
        return None
    return {
        "en": "eng",
        "zh": "zho",
        "eng": "eng",
        "zho": "zho",
    }.get(normalized, normalized)


def _debug_subdir(debug_root_dir: Optional[str], *parts: str) -> Optional[str]:
    if not debug_root_dir:
        return None
    path = os.path.join(debug_root_dir, *parts)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data classes for synthesis results
# ---------------------------------------------------------------------------


@dataclass
class SynthesizedAudio:
    """Audio for a single tagged text variant."""

    text_id: str
    tagged_text: str
    audio_bytes: bytes  # WAV at TARGET_SR
    sample_rate: int
    duration_sec: float
    segment_timestamps: List[Dict]  # [{text, tag, alpha, start_sec, end_sec}, ...]
    synthesis_mode: str  # "single", "multi_normal", "multi_frequent"
    duration_control: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledEditPair:
    """A source→target editing pair with assembled audio."""

    source_id: str
    target_id: str
    nl_instruction: str
    source_audio: bytes  # WAV bytes
    target_audio: bytes  # WAV bytes (natural TTS output — ground truth)
    source_duration: float
    target_duration: float
    changed_segments: List[Dict]  # From describe_tag_diff
    assembly_method: str  # "direct" or "splice_and_smooth"
    st_instruction: str = ""
    source_alignment: Dict[str, Any] = field(default_factory=dict)
    target_alignment: Dict[str, Any] = field(default_factory=dict)
    source_segment_timestamps: List[Dict[str, Any]] = field(default_factory=list)
    target_segment_timestamps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Full result from Stage II synthesis."""

    synthesized_audios: Dict[str, SynthesizedAudio]  # text_id → audio
    edit_pairs: List[AssembledEditPair]
    timing: Dict[str, float]
    alignment_cache: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def _numpy_to_wav(audio: np.ndarray, sr: int) -> bytes:
    """Convert float32 numpy array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _wav_to_numpy(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Convert WAV bytes to float32 numpy array and sample rate."""
    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio, sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using librosa."""
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def _resample_wav(wav_bytes: bytes, target_sr: int) -> bytes:
    """Resample WAV bytes to target sample rate."""
    audio, sr = _wav_to_numpy(wav_bytes)
    if sr == target_sr:
        return wav_bytes
    audio = _resample(audio, sr, target_sr)
    return _numpy_to_wav(audio, target_sr)


def _get_wav_duration(wav_bytes: bytes) -> float:
    """Get duration of WAV bytes in seconds."""
    audio, sr = _wav_to_numpy(wav_bytes)
    return len(audio) / sr


def _hash_audio_bytes(wav_bytes: bytes) -> str:
    return hashlib.sha256(wav_bytes).hexdigest()


def _concat_audio_arrays(arrays: List[np.ndarray], gap_samples: int = 0) -> np.ndarray:
    """Concatenate audio arrays with optional silence gaps."""
    if not arrays:
        return np.array([], dtype=np.float32)
    if gap_samples <= 0:
        return np.concatenate(arrays)

    parts = []
    gap = np.zeros(gap_samples, dtype=np.float32)
    for i, arr in enumerate(arrays):
        parts.append(arr)
        if i < len(arrays) - 1:
            parts.append(gap)
    return np.concatenate(parts)


def _slice_audio(wav_bytes: bytes, start_sec: float, end_sec: float) -> bytes:
    """Extract a time slice from WAV bytes."""
    audio, sr = _wav_to_numpy(wav_bytes)
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    start_idx = max(0, start_idx)
    end_idx = min(len(audio), end_idx)
    return _numpy_to_wav(audio[start_idx:end_idx], sr)


def _duration_target_seconds(tagged_text: str) -> Optional[List[float]]:
    tagged_segments = [segment for segment in parse_tags(tagged_text) if segment.tag_name is not None]
    if not tagged_segments:
        return None
    dur_flags = [segment.dur is not None for segment in tagged_segments]
    if not any(dur_flags):
        return None
    if not all(dur_flags):
        raise ValueError("dur must be present on every tagged segment when used")
    target_seconds = [float(segment.dur) for segment in tagged_segments if segment.dur is not None]
    total_seconds = sum(target_seconds)
    if total_seconds > TED_TAGGED_DURATION_MAX_SEC:
        raise ValueError(
            "TED-TTS tagged duration total exceeds the 850-token limit "
            f"({TED_TAGGED_DURATION_MAX_SEC:.1f}s): {total_seconds:.2f}s"
        )
    return target_seconds


def _segment_timestamp_dict(
    segment: TaggedSegment,
    *,
    start_sec: float,
    end_sec: float,
) -> Dict[str, Any]:
    return {
        "text": segment.text.strip(),
        "tag": segment.evaluation_tag,
        "tag_type": segment.tag_type,
        "alpha": segment.effective_alpha,
        "emotion_control_mode": segment.emotion_control_mode,
        "emotion_desc": segment.emotion_desc,
        "emotion_type_ref": segment.emotion_type_ref,
        "dur": segment.dur,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


# ---------------------------------------------------------------------------
# Segment-level synthesis routing
# ---------------------------------------------------------------------------


def _synthesize_segment_indextts(
    text: str,
    speaker_prompt_bytes: bytes,
    indextts_client: IndexTTSClient,
    *,
    emo_name: Optional[str] = None,
    emo_alpha: Optional[float] = None,
    emotion_desc: Optional[str] = None,
    timeout: float = 120,
) -> bytes:
    """Synthesize one segment using IndexTTS2 with emotion control.

    Returns WAV bytes at IndexTTS2's native sample rate (22050).
    """
    kwargs = {}
    if emotion_desc:
        kwargs["emo_text"] = emotion_desc
        kwargs["emo_alpha"] = 1.0
    elif emo_name and emo_name in EMO_PRESETS:
        kwargs["emo_vector"] = EMO_PRESETS[emo_name]
        kwargs["emo_alpha"] = (
            emo_alpha if emo_alpha is not None else get_default_alpha(emo_name)
        )
    elif emo_name == "calm" or emo_name == NEUTRAL_TAG:
        # Use calm emotion vector with low alpha for neutral
        kwargs["emo_vector"] = EMO_PRESETS["calm"]
        kwargs["emo_alpha"] = 0.3

    wav_bytes = indextts_client.synthesize(
        text=text,
        spk_audio_bytes=speaker_prompt_bytes,
        output_format="wav",
        timeout=timeout,
        **kwargs,
    )
    return wav_bytes


def _synthesize_segment_ted(
    segment: TaggedSegment,
    speaker_prompt_bytes: bytes,
    ted_tts_client: TedTTSClient,
    *,
    timeout: float = 240,
    ted_max_mel_tokens: Optional[int] = None,
) -> bytes:
    """Synthesize one description-mode segment using TED-TTS tagged synthesis."""
    if segment.tag_type not in {"emotion", "none"}:
        raise ValueError("TED backend only supports emotion description tags")
    if segment.emotion_control_mode != EMOTION_CONTROL_DESCRIPTION:
        raise ValueError("TED backend requires <emo desc=...> tags")
    tagged_text = render_emo_tag(
        text=segment.text.strip(),
        emotion_desc=segment.emotion_desc,
        emotion_type_ref=segment.emotion_type_ref,
    )
    result = ted_tts_client.synthesize_tagged(
        tagged_text=tagged_text,
        spk_audio_bytes=speaker_prompt_bytes,
        output_format="wav",
        timeout=timeout,
        method="hmm",
        max_mel_tokens=ted_max_mel_tokens,
    )
    assert isinstance(result, bytes)
    return result


def _synthesize_segment(
    segment: TaggedSegment,
    speaker_prompt_bytes: bytes,
    speaker_prompt_text: str,
    indextts_client: IndexTTSClient,
    *,
    backend: str = SYNTHESIS_BACKEND_INDEX,
    ted_tts_client: Optional[TedTTSClient] = None,
    ted_max_mel_tokens: Optional[int] = None,
) -> bytes:
    """Route a single segment to the appropriate synthesis tool.

    Returns WAV bytes resampled to TARGET_SR (24kHz).
    """
    backend = normalize_synthesis_backend(backend)
    tag_type = segment.tag_type

    if tag_type == "untagged":
        raise ValueError(
            "Untagged spans are not allowed in synthesis input. "
            'Wrap neutral content with <emo type="none">...</emo>.'
        )

    if backend == SYNTHESIS_BACKEND_TED:
        if ted_tts_client is None:
            raise ValueError("ted_tts_client is required for backend=ted")
        raw_wav = _synthesize_segment_ted(
            segment,
            speaker_prompt_bytes,
            ted_tts_client,
            ted_max_mel_tokens=ted_max_mel_tokens,
        )
    elif tag_type in ("emotion", "none"):
        emo_name = segment.emotion_type or segment.tag_name if tag_type == "emotion" else None
        emo_alpha = segment.effective_alpha
        raw_wav = _synthesize_segment_indextts(
            text=segment.text.strip(),
            speaker_prompt_bytes=speaker_prompt_bytes,
            indextts_client=indextts_client,
            emo_name=emo_name,
            emo_alpha=emo_alpha,
            emotion_desc=segment.emotion_desc
            if segment.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION
            else None,
        )
    else:
        raise ValueError(f"Unknown tag type: {tag_type}")

    # Resample to TARGET_SR
    return _resample_wav(raw_wav, TARGET_SR)


# ---------------------------------------------------------------------------
# Synthesis modes
# ---------------------------------------------------------------------------


def _synthesize_single_tag(
    tagged_text: str,
    text_id: str,
    speaker_prompt_bytes: bytes,
    speaker_prompt_text: str,
    indextts_client: IndexTTSClient,
    ted_tts_client: Optional[TedTTSClient] = None,
    backend: str = SYNTHESIS_BACKEND_INDEX,
    max_candidates: int = 3,
    ted_max_mel_tokens: Optional[int] = None,
) -> SynthesizedAudio:
    """Synthesize a text where one dominant tag covers the entire text.

    The full plain text is synthesized as a single utterance with the dominant
    tag's emotion/style applied.  This is appropriate when the tagged text has
    a single effective tag (e.g. ``<emo type="happy">full text</emo>`` or
    ``<emo type="happy">part A</emo><emo type="none">part B</emo>`` where
    happy is the dominant emotion).

    For texts with multiple *distinct non-none* tags, use _synthesize_multi_normal.
    """
    backend = normalize_synthesis_backend(backend)
    segments = parse_tags(tagged_text)
    tagged_segs = [s for s in segments if s.tag_name is not None]
    if not tagged_segs:
        raise ValueError(f"No tagged segments found in: {tagged_text}")

    # Find the dominant (non-none, non-calm) tag — this is the "active" tag
    active_segs = [
        s for s in tagged_segs if s.tag_name not in (NEUTRAL_TAG, "calm", "none")
    ]
    dominant_seg = active_segs[0] if active_segs else tagged_segs[0]

    # Use the FULL plain text (all segments combined) for synthesis
    plain_text = strip_tags(tagged_text)

    # Create a synthetic TaggedSegment covering the full text with the dominant tag
    full_seg = TaggedSegment(
        text=plain_text,
        tag_name=dominant_seg.tag_name,
        tag_type=dominant_seg.tag_type,
        alpha=dominant_seg.alpha,
        emotion_control_mode=dominant_seg.emotion_control_mode,
        emotion_type=dominant_seg.emotion_type,
        emotion_level=dominant_seg.emotion_level,
        emotion_desc=dominant_seg.emotion_desc,
        emotion_type_ref=dominant_seg.emotion_type_ref,
        dur=None if backend == SYNTHESIS_BACKEND_TED else dominant_seg.dur,
    )

    best_wav = None
    best_dur = None

    for attempt in range(max_candidates):
        try:
            wav = _synthesize_segment(
                full_seg,
                speaker_prompt_bytes,
                speaker_prompt_text,
                indextts_client,
                backend=backend,
                ted_tts_client=ted_tts_client,
                ted_max_mel_tokens=ted_max_mel_tokens,
            )
            dur = _get_wav_duration(wav)

            if best_wav is None:
                best_wav = wav
                best_dur = dur
                break
        except ServiceError as e:
            if backend == SYNTHESIS_BACKEND_TED:
                raise
            logger.warning(f"Single-tag synthesis attempt {attempt + 1} service error: {e}")
            continue
        except Exception as e:
            logger.warning(f"Single-tag synthesis attempt {attempt + 1} failed: {e}")
            continue

    if best_wav is None:
        raise RuntimeError(
            f"All {max_candidates} synthesis attempts failed for text_id={text_id}"
        )

    return SynthesizedAudio(
        text_id=text_id,
        tagged_text=tagged_text,
        audio_bytes=best_wav,
        sample_rate=TARGET_SR,
        duration_sec=best_dur,
        segment_timestamps=[
            {
                "text": plain_text,
                "tag": full_seg.evaluation_tag,
                "tag_type": full_seg.tag_type,
                "alpha": full_seg.effective_alpha,
                "emotion_control_mode": full_seg.emotion_control_mode,
                "emotion_desc": full_seg.emotion_desc,
                "emotion_type_ref": full_seg.emotion_type_ref,
                "dur": full_seg.dur,
                "start_sec": 0.0,
                "end_sec": best_dur,
            }
        ],
        synthesis_mode="single" if backend == SYNTHESIS_BACKEND_INDEX else "single_ted",
        duration_control=(
            {
                "enabled": False,
                "backend": "ted",
                "max_mel_tokens": ted_max_mel_tokens,
            }
            if backend == SYNTHESIS_BACKEND_TED
            else {"enabled": False}
        ),
    )


def _synthesize_multi_normal(
    tagged_text: str,
    text_id: str,
    speaker_prompt_bytes: bytes,
    speaker_prompt_text: str,
    boundaries: List,
    indextts_client: IndexTTSClient,
    smoother_client: BoundaryPauseSmootherClient,
    ted_tts_client: Optional[TedTTSClient] = None,
    backend: str = SYNTHESIS_BACKEND_INDEX,
    aligner_language: Optional[str] = None,
    aligner_backend: str = "qwen3",
    max_candidates: int = 2,
    debug_synthesize: bool = False,
    debug_root_dir: Optional[str] = None,
    ted_max_mel_tokens: Optional[int] = None,
) -> SynthesizedAudio:
    """Synthesize multi-tag text with normal switching frequency.

    Per-segment independent synthesis → concatenate with silence → smooth.
    """
    backend = normalize_synthesis_backend(backend)
    segments = parse_tags(tagged_text)
    tagged_segs = [s for s in segments if s.tag_name is not None]

    if not tagged_segs:
        raise ValueError(f"No tagged segments found in: {tagged_text}")

    # Synthesize each segment independently, collect candidates
    seg_audios = []  # List of np.ndarray at TARGET_SR
    seg_durations = []

    for seg_idx, seg in enumerate(tagged_segs):
        best_audio = None
        for attempt in range(max_candidates):
            try:
                wav = _synthesize_segment(
                    seg,
                    speaker_prompt_bytes,
                    speaker_prompt_text,
                    indextts_client,
                    backend=backend,
                    ted_tts_client=ted_tts_client,
                    ted_max_mel_tokens=ted_max_mel_tokens,
                )
                audio_arr, sr = _wav_to_numpy(wav)
                assert sr == TARGET_SR, f"Expected {TARGET_SR}, got {sr}"
                best_audio = audio_arr
                break
            except ServiceError as e:
                if backend == SYNTHESIS_BACKEND_TED:
                    raise
                logger.warning(
                    f"Multi-normal seg {seg_idx} attempt {attempt + 1} service error: {e}"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"Multi-normal seg {seg_idx} attempt {attempt + 1} failed: {e}"
                )
                continue

        if best_audio is None:
            raise RuntimeError(
                f"All synthesis attempts failed for segment {seg_idx} "
                f"(tag={seg.tag_name}, text='{seg.text[:50]}...')"
            )
        seg_audios.append(best_audio)
        seg_durations.append(len(best_audio) / TARGET_SR)

    # Concatenate with silence gaps
    gap_samples = int(SILENCE_GAP_MS / 1000.0 * TARGET_SR)
    concat_audio = _concat_audio_arrays(seg_audios, gap_samples=gap_samples)
    concat_wav = _numpy_to_wav(concat_audio, TARGET_SR)

    # Compute boundary positions in the concatenated audio
    full_text = strip_tags(tagged_text)
    boundary_positions = []
    cumulative_sec = 0.0

    for i, dur in enumerate(seg_durations):
        cumulative_sec += dur
        if i < len(seg_durations) - 1:
            # Boundary is at the center of the silence gap
            gap_sec = SILENCE_GAP_MS / 1000.0
            boundary_sec = cumulative_sec + gap_sec / 2.0

            # Determine if boundary has punctuation
            seg_text = tagged_segs[i].text.strip()
            has_punct = False
            if seg_text:
                from .tag_utils import ALL_PUNCTUATION

                has_punct = seg_text[-1] in ALL_PUNCTUATION

            boundary_positions.append(
                {
                    "mode": "adjust",
                    "position_sec": boundary_sec,
                    "pause_interval_sec": [
                        boundary_sec - gap_sec / 2.0,
                        boundary_sec + gap_sec / 2.0,
                    ],
                    "max_pause_sec": 0.4 if has_punct else 0.0,
                    "min_pause_sec": 0.0,
                    "regen_margin_sec": 0.2,
                }
            )
            cumulative_sec += gap_sec

    # Apply boundary pause smoothing at boundaries (mandatory — failures propagate)
    if boundary_positions:
        final_wav = smoother_client.smooth(
            audio_bytes=concat_wav,
            text=full_text,
            points=boundary_positions,
            aligner_backend=aligner_backend,
            language=aligner_language,
            debug=debug_synthesize,
            debug_dir=_debug_subdir(debug_root_dir, "boundary_smoother", text_id)
            if debug_synthesize
            else None,
            timeout=180,
        )
    else:
        final_wav = concat_wav

    # Compute segment timestamps in the final audio
    # After smoothing, timestamps may have shifted. Use CTC aligner if available,
    # but for now estimate from original durations.
    seg_timestamps = []
    t = 0.0
    total_dur = _get_wav_duration(final_wav)

    # Scale factor in case smoothing changed total duration
    orig_total = sum(seg_durations) + (len(seg_durations) - 1) * SILENCE_GAP_MS / 1000.0
    scale = total_dur / orig_total if orig_total > 0 else 1.0

    for i, (seg, dur) in enumerate(zip(tagged_segs, seg_durations)):
        scaled_dur = dur * scale
        seg_end = t + scaled_dur
        # Last segment: extend to total duration to avoid trailing truncation
        if i == len(seg_durations) - 1:
            seg_end = total_dur
        seg_timestamps.append(
            {
                "text": seg.text.strip(),
                "tag": seg.evaluation_tag,
                "tag_type": seg.tag_type,
                "alpha": seg.effective_alpha,
                "emotion_control_mode": seg.emotion_control_mode,
                "emotion_desc": seg.emotion_desc,
                "emotion_type_ref": seg.emotion_type_ref,
                "dur": seg.dur,
                "start_sec": t,
                "end_sec": seg_end,
            }
        )
        t += scaled_dur
        if i < len(seg_durations) - 1:
            t += (SILENCE_GAP_MS / 1000.0) * scale

    return SynthesizedAudio(
        text_id=text_id,
        tagged_text=tagged_text,
        audio_bytes=final_wav,
        sample_rate=TARGET_SR,
        duration_sec=total_dur,
        segment_timestamps=seg_timestamps,
        synthesis_mode=(
            "multi_normal" if backend == SYNTHESIS_BACKEND_INDEX else "multi_normal_ted"
        ),
        duration_control=(
            {
                "enabled": False,
                "backend": "ted",
                "max_mel_tokens": ted_max_mel_tokens,
            }
            if backend == SYNTHESIS_BACKEND_TED
            else {"enabled": False}
        ),
    )


def _synthesize_multi_frequent_ted(
    tagged_text: str,
    text_id: str,
    speaker_prompt_bytes: bytes,
    ted_tts_client: TedTTSClient,
    *,
    aligner_url: Optional[str] = None,
    smoother_url: Optional[str] = None,
    aligner_language: Optional[str] = None,
    aligner_backend: str = "qwen3",
    smooth_radius_sec: Optional[float] = 0.45,
    max_candidates: int = 3,
    debug_synthesize: bool = False,
    debug_root_dir: Optional[str] = None,
    ted_max_mel_tokens: Optional[int] = None,
) -> SynthesizedAudio:
    """Synthesize frequent emotion switches via TED-TTS tagged endpoint."""
    plain_text = strip_tags(tagged_text)
    tagged_segs = [s for s in parse_tags(tagged_text) if s.tag_name is not None]
    target_seconds = _duration_target_seconds(tagged_text)
    uses_duration = target_seconds is not None

    best_wav = None
    best_dur = None

    for attempt in range(max_candidates):
        try:
            try:
                health = ted_tts_client.health()
                if health.get("model_loaded") is False:
                    raise ServiceError(f"TED-TTS service not loaded: {health}")
            except Exception as ce:
                raise ServiceError(f"TED-TTS service unreachable: {ce}") from ce

            result = ted_tts_client.synthesize_tagged(
                tagged_text=tagged_text,
                spk_audio_bytes=speaker_prompt_bytes,
                output_format="wav",
                timeout=240,
                method="hmm",
                **(
                    {
                        "duration_mode": TED_DURATION_MODE,
                        "eos_control_policy": TED_EOS_CONTROL_POLICY,
                        "eos_grace_seconds": TED_EOS_GRACE_SECONDS,
                    }
                    if uses_duration
                    else {}
                ),
                max_mel_tokens=TED_TAGGED_DURATION_MAX_MEL_TOKENS
                if uses_duration
                else ted_max_mel_tokens,
            )
            assert isinstance(result, bytes)
            wav = _resample_wav(result, TARGET_SR)
            dur = _get_wav_duration(wav)
            if best_wav is None or dur > 0:
                best_wav = wav
                best_dur = dur
                break
        except ServiceError as se:
            logger.error(
                f"TED multi-frequent synthesis attempt {attempt + 1} service error: {se}"
            )
            raise
        except Exception as e:
            logger.warning(
                f"TED multi-frequent synthesis attempt {attempt + 1} failed: {e}",
                exc_info=True,
            )
            continue

    if best_wav is None:
        raise RuntimeError(
            f"All {max_candidates} TED tagged synthesis attempts failed "
            f"for text_id={text_id}"
        )

    total_dur = _get_wav_duration(best_wav)
    duration_control: Dict[str, Any] = {}
    if target_seconds is not None:
        duration_control = {
            "enabled": True,
            "backend": "ted",
            "duration_mode": TED_DURATION_MODE,
            "eos_control_policy": TED_EOS_CONTROL_POLICY,
            "eos_grace_seconds": TED_EOS_GRACE_SECONDS,
            "target_seconds": target_seconds,
            "max_mel_tokens": TED_TAGGED_DURATION_MAX_MEL_TOKENS,
            "max_seconds": TED_TAGGED_DURATION_MAX_SEC,
        }
        seg_timestamps = []
        cursor = 0.0
        scale = total_dur / sum(target_seconds) if sum(target_seconds) > 0 else 1.0
        for index, (segment, target_sec) in enumerate(zip(tagged_segs, target_seconds)):
            end_sec = cursor + target_sec * scale
            if index == len(tagged_segs) - 1:
                end_sec = total_dur
            seg_timestamps.append(
                _segment_timestamp_dict(
                    segment,
                    start_sec=cursor,
                    end_sec=end_sec,
                )
            )
            cursor = end_sec
    else:
        duration_control = {
            "enabled": False,
            "backend": "ted",
            "max_mel_tokens": ted_max_mel_tokens,
        }
        seg_timestamps = [
            {
                "text": plain_text,
                "tag": tagged_segs[0].evaluation_tag if tagged_segs else None,
                "tag_type": tagged_segs[0].tag_type if tagged_segs else "none",
                "alpha": tagged_segs[0].effective_alpha if tagged_segs else None,
                "emotion_control_mode": tagged_segs[0].emotion_control_mode if tagged_segs else None,
                "emotion_desc": tagged_segs[0].emotion_desc if tagged_segs else None,
                "emotion_type_ref": tagged_segs[0].emotion_type_ref
                if tagged_segs
                else None,
                "dur": tagged_segs[0].dur if tagged_segs else None,
                "start_sec": 0.0,
                "end_sec": total_dur,
            }
        ]

    return SynthesizedAudio(
        text_id=text_id,
        tagged_text=tagged_text,
        audio_bytes=best_wav,
        sample_rate=TARGET_SR,
        duration_sec=best_dur,
        segment_timestamps=seg_timestamps,
        synthesis_mode="multi_frequent_ted",
        duration_control=duration_control,
    )


def _synthesize_multi_frequent(
    tagged_text: str,
    text_id: str,
    speaker_prompt_bytes: bytes,
    *,
    backend: str,
    ted_tts_client: Optional[TedTTSClient] = None,
    max_candidates: int = 3,
    ted_max_mel_tokens: Optional[int] = None,
) -> SynthesizedAudio:
    """Synthesize frequent-switching text through the supported backend."""
    backend = normalize_synthesis_backend(backend)
    if backend == SYNTHESIS_BACKEND_INDEX:
        raise ValueError("multi_emo_frequent + index is not supported")
    if ted_tts_client is None:
        raise ValueError("ted_tts_client is required for multi_emo_frequent + ted")
    return _synthesize_multi_frequent_ted(
        tagged_text=tagged_text,
        text_id=text_id,
        speaker_prompt_bytes=speaker_prompt_bytes,
        ted_tts_client=ted_tts_client,
        max_candidates=max_candidates,
        ted_max_mel_tokens=ted_max_mel_tokens,
    )


# ---------------------------------------------------------------------------
# Edit pair assembly
# ---------------------------------------------------------------------------


def _build_edit_tagged_text_and_diffs(
    source_text: str,
    target_text: str,
) -> Tuple[str, List[Dict]]:
    """Build deterministic assembly markup and diff metadata.

    The returned `edit_tagged_text` is the exact plain transcript with changed
    tagged segments wrapped as `<target>...</target>`. The returned `diffs`
    match the existing `describe_tag_diff()` schema.

    Raises:
        ValueError: if source/target tagged structures are incompatible for
            deterministic span replacement.
    """
    source_segs = [s for s in parse_tags(source_text) if s.tag_name is not None]
    target_segs = [s for s in parse_tags(target_text) if s.tag_name is not None]

    if len(source_segs) != len(target_segs):
        raise ValueError("Source and target tagged segment counts must match")

    diffs = []
    parts = []
    for idx, (source_seg, target_seg) in enumerate(zip(source_segs, target_segs)):
        if source_seg.text != target_seg.text:
            raise ValueError(
                f"Segment text mismatch at index {idx}: "
                f"source={source_seg.text!r} target={target_seg.text!r}"
            )

        changed = source_seg.control_signature != target_seg.control_signature

        if changed:
            diffs.append(
                {
                    "segment_idx": idx,
                    "segment_text": source_seg.text.strip(),
                    "source_tag": source_seg.evaluation_tag,
                    "target_tag": target_seg.evaluation_tag,
                    "source_alpha": source_seg.effective_alpha,
                    "target_alpha": target_seg.effective_alpha,
                    "source_emotion_control_mode": source_seg.emotion_control_mode,
                    "target_emotion_control_mode": target_seg.emotion_control_mode,
                    "source_emotion_desc": source_seg.emotion_desc,
                    "target_emotion_desc": target_seg.emotion_desc,
                    "source_emotion_type_ref": source_seg.emotion_type_ref,
                    "target_emotion_type_ref": target_seg.emotion_type_ref,
                    "source_dur": source_seg.dur,
                    "target_dur": target_seg.dur,
                }
            )
            parts.append(f"<target>{source_seg.text}</target>")
        else:
            parts.append(source_seg.text)

    return "".join(parts), diffs


def _assemble_edit_pair(
    source_synth: SynthesizedAudio,
    target_synth: SynthesizedAudio,
    nl_instruction: str,
    st_instruction: str = "",
    assembly_client: Optional[AudioAssemblyClient] = None,
    aligner_language: Optional[str] = None,
    aligner_backend: str = "qwen3",
    debug_synthesize: bool = False,
    debug_root_dir: Optional[str] = None,
    *,
    assembly_f5_steps: int,
) -> AssembledEditPair:
    """Assemble an editing pair by splicing segments from two TTS outputs.

    Builds a "source" (input) waveform for the editing model by taking:
      - Unchanged segments from TARGET audio (identical to ground truth —
        the model learns to preserve these)
      - Changed segments from SOURCE audio (old emotion — the model learns
        to replace these with the target version)

    The "target" (ground truth) is always the raw, natural TTS output for the
    target tagged text — no splicing artifacts.

    Delegates to the Audio Assembly service for alignment-based splicing and
    boundary regeneration.
    """
    source_text = source_synth.tagged_text
    target_text = target_synth.tagged_text

    edit_tagged_text, diffs = _build_edit_tagged_text_and_diffs(
        source_text, target_text
    )

    # If no diffs (shouldn't happen, but safety), return as-is
    if not diffs:
        return AssembledEditPair(
            source_id=source_synth.text_id,
            target_id=target_synth.text_id,
            nl_instruction=nl_instruction,
            st_instruction=st_instruction,
            source_audio=source_synth.audio_bytes,
            target_audio=target_synth.audio_bytes,
            source_duration=source_synth.duration_sec,
            target_duration=target_synth.duration_sec,
            changed_segments=diffs,
            assembly_method="direct",
        )

    # For single-tag texts, the entire audio changes — no splicing needed
    source_segs = [s for s in parse_tags(source_text) if s.tag_name is not None]
    if len(source_segs) <= 1:
        return AssembledEditPair(
            source_id=source_synth.text_id,
            target_id=target_synth.text_id,
            nl_instruction=nl_instruction,
            st_instruction=st_instruction,
            source_audio=source_synth.audio_bytes,
            target_audio=target_synth.audio_bytes,
            source_duration=source_synth.duration_sec,
            target_duration=target_synth.duration_sec,
            changed_segments=diffs,
            assembly_method="direct",
        )

    # Multi-segment: splice via Audio Assembly service
    target_segs = [s for s in parse_tags(target_text) if s.tag_name is not None]
    if len(source_segs) != len(target_segs):
        raise ValueError("Source and target tagged segment counts must match")

    if assembly_client is None:
        raise RuntimeError(
            "Audio Assembly client is required for multi-segment edit pair assembly"
        )

    spliced_wav = _assemble_from_base_and_donor(
        assembly_f5_steps=assembly_f5_steps,
        base_audio_bytes=target_synth.audio_bytes,  # unchanged segments
        donor_audio_bytes=source_synth.audio_bytes,  # changed segments
        edit_tagged_text=edit_tagged_text,
        assembly_client=assembly_client,
        aligner_language=aligner_language,
        aligner_backend=aligner_backend,
        debug=debug_synthesize,
        debug_dir=_debug_subdir(
            debug_root_dir,
            "assembly",
            f"{source_synth.text_id}_to_{target_synth.text_id}",
        )
        if debug_synthesize
        else None,
    )

    spliced_dur = _get_wav_duration(spliced_wav)

    return AssembledEditPair(
        source_id=source_synth.text_id,
        target_id=target_synth.text_id,
        nl_instruction=nl_instruction,
        st_instruction=st_instruction,
        source_audio=spliced_wav,
        target_audio=target_synth.audio_bytes,
        source_duration=spliced_dur,
        target_duration=target_synth.duration_sec,
        changed_segments=diffs,
        assembly_method="splice_and_smooth",
    )


def _assemble_from_base_and_donor(
    *,
    assembly_f5_steps: int,
    base_audio_bytes: bytes,
    donor_audio_bytes: bytes,
    edit_tagged_text: str,
    assembly_client: AudioAssemblyClient,
    aligner_language: Optional[str] = None,
    aligner_backend: str = "qwen3",
    debug: bool = False,
    debug_dir: Optional[str] = None,
    timeout: float = 300,
) -> bytes:
    """Assemble output from an acoustic base and donor audio.

    Audio Assembly's contract is:
      - untagged spans come from ``source_audio_bytes``
      - ``<target>`` spans come from ``target_audio_bytes``

    The dataset pipeline calls this with target-TTS audio as the base and
    source-TTS audio as the donor. The service calls it with the user's source
    audio as the base and the synthetic target audio as the donor.
    """
    assemble_kwargs = {
        "source_audio_bytes": base_audio_bytes,
        "target_audio_bytes": donor_audio_bytes,
        "edit_tagged_text": edit_tagged_text,
        "language": aligner_language,
        "aligner_backend": aligner_backend,
        "f5_steps": assembly_f5_steps,
        "timeout": timeout,
    }
    if debug:
        assemble_kwargs["debug"] = True
        assemble_kwargs["debug_dir"] = debug_dir
    return assembly_client.assemble(**assemble_kwargs)


def _segment_timestamps_from_alignment_response(
    tagged_text: str,
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    answers = response.get("answers")
    if not isinstance(answers, list):
        return []
    return build_segment_timestamps_from_answers(tagged_text, answers)


def _align_edit_pair_outputs(
    *,
    edit_pairs: List[AssembledEditPair],
    synthesized: Dict[str, SynthesizedAudio],
    aligner_client: AlignmentClient,
    aligner_backend: str,
    aligner_language: Optional[str],
) -> Dict[str, Any]:
    """Align final pair-side audio once in Synth so Eval can reuse it."""
    if not edit_pairs:
        return {
            "aligner_backend": aligner_backend,
            "aligner_language": aligner_language,
            "pairs": {},
        }

    batch_items: List[Dict[str, Any]] = []
    refs: List[Tuple[AssembledEditPair, str, str, bytes, str]] = []
    for pair in edit_pairs:
        src_synth = synthesized.get(pair.source_id)
        tgt_synth = synthesized.get(pair.target_id)
        if src_synth is None or tgt_synth is None:
            continue
        for side, audio_bytes, tagged_text in (
            ("source", pair.source_audio, src_synth.tagged_text),
            ("target", pair.target_audio, tgt_synth.tagged_text),
        ):
            plain_text, query_segments = build_segment_queries(tagged_text)
            item: Dict[str, Any] = {
                "audio": audio_bytes,
                "text": plain_text,
                "word_level": True,
            }
            if aligner_language is not None:
                item["language"] = aligner_language
            queries = [str(segment["query"]) for segment in query_segments]
            if queries:
                item["queries"] = queries
            batch_items.append(item)
            refs.append((pair, side, tagged_text, audio_bytes, plain_text))

    if not batch_items:
        return {
            "aligner_backend": aligner_backend,
            "aligner_language": aligner_language,
            "pairs": {},
        }

    responses = aligner_client.align_batch(batch_items, backend=aligner_backend)
    if len(responses) != len(refs):
        raise RuntimeError(
            "Aligner batch response count mismatch: "
            f"expected {len(refs)}, got {len(responses)}"
        )

    cache: Dict[str, Any] = {
        "aligner_backend": aligner_backend,
        "aligner_language": aligner_language,
        "pairs": {},
    }
    for (pair, side, tagged_text, audio_bytes, plain_text), response in zip(
        refs,
        responses,
    ):
        if not isinstance(response, dict):
            raise RuntimeError(
                "Invalid aligner response for "
                f"{pair.source_id}->{pair.target_id}: {response!r}"
            )
        segment_timestamps = _segment_timestamps_from_alignment_response(
            tagged_text,
            response,
        )
        if side == "source":
            pair.source_alignment = response
            pair.source_segment_timestamps = segment_timestamps
        else:
            pair.target_alignment = response
            pair.target_segment_timestamps = segment_timestamps

        pair_key = f"{pair.source_id}->{pair.target_id}"
        cache["pairs"].setdefault(pair_key, {})[side] = {
            "audio_hash": _hash_audio_bytes(audio_bytes),
            "plain_text": plain_text,
            "tagged_text": tagged_text,
            "alignment": response,
            "segment_timestamps": segment_timestamps,
        }
    return cache


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def synthesize_tagged_text_group(
    group: TaggedTextGroup,
    *,
    assembly_f5_steps: int,
    indextts_url: Optional[str] = None,
    ted_tts_url: Optional[str] = None,
    boundary_pause_smoother_url: str = "http://localhost:8018",
    audio_assembly_url: str = "http://localhost:8021",
    aligner_url: str = "http://localhost:8029",
    max_candidates: int = 1,
    aligner_backend: str = "qwen3",
    parallel: bool = True,
    debug_synthesize: bool = False,
    debug_root_dir: Optional[str] = None,
) -> SynthesisResult:
    """Synthesize speech for all tagged texts in a group and assemble editing pairs.

    Args:
        group: TaggedTextGroup from Stage I.
        indextts_url: IndexTTS2 service URL.
        ted_tts_url: TED-TTS service URL.
        boundary_pause_smoother_url: Boundary Pause Smoother service URL.
        audio_assembly_url: Audio Assembly service URL (for edit pair assembly).
        aligner_url: Aligner gateway service URL.
        max_candidates: Number of synthesis attempts per text (1 = no retries).
        ted_max_generated_duration_sec: TED hard generation cap in seconds.
        aligner_backend: Explicit aligner gateway backend for smoothing and
            edit-pair assembly ("qwen3").
        parallel: If True, synthesize texts concurrently via threads.
            Each request uses the selected synthesis backend.
        debug_synthesize: Save raw and service-level intermediate artifacts.
        debug_root_dir: Directory for debug artifacts.

    Returns:
        SynthesisResult with all synthesized audios and assembled edit pairs.
    """
    t0 = time.time()

    speaker_prompt = group.speaker_prompt_audio
    speaker_text = group.speaker_prompt_text
    synthesis_profile = normalize_synthesis_profile(group.metadata["synthesis_profile"])
    synthesis_backend = synthesis_profile["synthesis_backend"]
    synthesis_mode = synthesis_profile["synthesis_mode"]
    effective_ted_max_generated_duration_sec = float(synthesis_profile["ted_max_generated_duration_sec"])
    effective_ted_max_mel_tokens = ted_max_mel_tokens_from_duration_sec(
        effective_ted_max_generated_duration_sec
    )
    aligner_language = _aligner_language_code((group.metadata or {}).get("language"))

    for text_id, tagged_text in group.tagged_texts.items():
        try:
            ensure_fully_tagged(tagged_text)
        except ValueError as exc:
            raise ValueError(f"Invalid tagged text {text_id}: {exc}") from exc
    try:
        validate_tagged_texts_for_profile(group.tagged_texts, synthesis_profile)
    except ValueError as exc:
        raise ValueError(f"Invalid synthesis profile/tag combination: {exc}") from exc

    if synthesis_backend == SYNTHESIS_BACKEND_INDEX and not indextts_url:
        raise ValueError("IndexTTS synthesis requires the direct index_tts service dependency")
    if synthesis_backend == SYNTHESIS_BACKEND_TED and not ted_tts_url:
        raise ValueError("TED synthesis requires the direct ted_tts service dependency")
    indextts_client = IndexTTSClient(indextts_url) if indextts_url else None
    ted_tts_client = TedTTSClient(ted_tts_url) if ted_tts_url else None
    smoother_client = BoundaryPauseSmootherClient(boundary_pause_smoother_url)
    assembly_client = AudioAssemblyClient(audio_assembly_url)
    aligner_client = AlignmentClient(aligner_url)

    synthesized = {}
    timing = {}

    def _synthesize_one_text(text_id: str, tagged_text: str) -> SynthesizedAudio:
        """Synthesize a single tagged text. Raises on failure."""
        if synthesis_mode == SYNTHESIS_MODE_SINGLE:
            return _synthesize_single_tag(
                tagged_text,
                text_id,
                speaker_prompt,
                speaker_text,
                indextts_client,
                ted_tts_client=ted_tts_client,
                backend=synthesis_backend,
                max_candidates=max_candidates,
                ted_max_mel_tokens=effective_ted_max_mel_tokens,
            )
        if synthesis_mode == SYNTHESIS_MODE_MULTI_NORMAL:
            return _synthesize_multi_normal(
                tagged_text,
                text_id,
                speaker_prompt,
                speaker_text,
                group.boundaries,
                indextts_client,
                smoother_client,
                ted_tts_client=ted_tts_client,
                backend=synthesis_backend,
                aligner_language=aligner_language,
                aligner_backend=aligner_backend,
                max_candidates=max_candidates,
                debug_synthesize=debug_synthesize,
                debug_root_dir=debug_root_dir,
                ted_max_mel_tokens=effective_ted_max_mel_tokens,
            )
        if synthesis_mode == SYNTHESIS_MODE_MULTI_FREQUENT:
            return _synthesize_multi_frequent(
                tagged_text,
                text_id,
                speaker_prompt,
                backend=synthesis_backend,
                ted_tts_client=ted_tts_client,
                max_candidates=max_candidates,
                ted_max_mel_tokens=effective_ted_max_mel_tokens,
            )
        raise ValueError(f"Unknown synthesis_mode={synthesis_mode!r}")

    # ---- Step 1: Synthesize each unique tagged text ----
    t_synth_start = time.time()

    if parallel and len(group.tagged_texts) > 1:
        # Submit independent text requests concurrently.
        # calls across different texts. Services use internal inference locks,
        # so requests to the same service are serialized server-side.
        max_workers = min(len(group.tagged_texts), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_id = {
                pool.submit(_synthesize_one_text, tid, txt): tid
                for tid, txt in group.tagged_texts.items()
            }
            for future in as_completed(future_to_id):
                text_id = future_to_id[future]
                synth = future.result()
                synthesized[text_id] = synth
                logger.info(
                    f"Synthesized text_id={text_id} mode={synth.synthesis_mode} "
                    f"dur={synth.duration_sec:.2f}s"
                )
    else:
        # Sequential synthesis (single text or parallel=False)
        for text_id, tagged_text in group.tagged_texts.items():
            synth = _synthesize_one_text(text_id, tagged_text)
            synthesized[text_id] = synth
            logger.info(
                f"Synthesized text_id={text_id} mode={synth.synthesis_mode} "
                f"dur={synth.duration_sec:.2f}s"
            )

    timing["synthesis_sec"] = time.time() - t_synth_start

    if debug_synthesize and debug_root_dir and synthesized:
        raw_dir = _debug_subdir(debug_root_dir, "raw_texts")
        assert raw_dir is not None
        for text_id, synth in synthesized.items():
            with open(os.path.join(raw_dir, f"{text_id}.wav"), "wb") as f:
                f.write(synth.audio_bytes)
        logger.info("Saved %d raw synthesized wavs to %s", len(synthesized), raw_dir)

    # ---- Step 2: Assemble edit pairs ----
    t_assembly_start = time.time()

    edit_pairs = []

    for ep in group.edit_pairs:
        pair_desc = f"{ep.source_id}->{ep.target_id}"

        if ep.source_id not in synthesized:
            raise ValueError("Edit pair references an unknown synthesized text")
        if ep.target_id not in synthesized:
            raise ValueError("Edit pair references an unknown synthesized text")

        # Skip self-pairs (same source and target)
        if ep.source_id == ep.target_id:
            continue

        assembled = _assemble_edit_pair(
            assembly_f5_steps=assembly_f5_steps,
            source_synth=synthesized[ep.source_id],
            target_synth=synthesized[ep.target_id],
            nl_instruction=ep.nl_instruction,
            st_instruction=ep.st_instruction,
            assembly_client=assembly_client,
            aligner_language=aligner_language,
            aligner_backend=aligner_backend,
            debug_synthesize=debug_synthesize,
            debug_root_dir=debug_root_dir,
        )
        edit_pairs.append(assembled)
        logger.info(
            f"Assembled pair {pair_desc}: method={assembled.assembly_method}"
        )

    timing["assembly_sec"] = time.time() - t_assembly_start

    # ---- Step 3: Align final pair-side audio for Eval/WDTW reuse ----
    t_alignment_start = time.time()
    alignment_cache = _align_edit_pair_outputs(
        edit_pairs=edit_pairs,
        synthesized=synthesized,
        aligner_client=aligner_client,
        aligner_backend=aligner_backend,
        aligner_language=aligner_language,
    )
    timing["alignment_sec"] = time.time() - t_alignment_start
    timing["total_sec"] = time.time() - t0

    return SynthesisResult(
        synthesized_audios=synthesized,
        edit_pairs=edit_pairs,
        timing=timing,
        alignment_cache=alignment_cache,
    )
