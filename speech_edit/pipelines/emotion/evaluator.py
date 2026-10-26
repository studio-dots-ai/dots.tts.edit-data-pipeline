"""Stage III metric collection for the emotion/style editing pipeline."""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from speech_edit.data.language import asr_language_name, alignment_language_code
from speech_edit.evaluation.audio import compute_wer, evaluate_quality, evaluate_speaker_similarity

from speech_edit.clients.services import (
    ASRClient,
    SpkSimClient,
    UTMOSClient,
    WDTWClient,
    WERClient,
)

from .emotion_acoustic_effect import (
    METRIC_KEY as EMOTION_ACOUSTIC_EFFECT_KEY,
)
from .emotion_acoustic_effect import compute_emotion_acoustic_effect
from .synthesizer import (
    AssembledEditPair,
    SynthesisResult,
)
from .tag_utils import strip_tags

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluableAudio:
    """Final audio artifact plus its evaluation expectations."""

    audio_hash: str
    audio_bytes: bytes
    expected_text: str
    tagged_text: str


def _hash_audio_bytes(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes).hexdigest()


def _build_evaluable_audio(audio_bytes: bytes, tagged_text: str) -> EvaluableAudio:
    return EvaluableAudio(
        audio_hash=_hash_audio_bytes(audio_bytes),
        audio_bytes=audio_bytes,
        expected_text=strip_tags(tagged_text),
        tagged_text=tagged_text,
    )


def _normalize_changed_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def build_edit_time_spans_from_segment_timestamps(
    segment_timestamps: List[Dict],
    changed_segments: List[Dict],
) -> List[Dict[str, Any]]:
    """Filter aligned tagged segment timestamps down to edited regions.

    Output is intentionally compact because it is nested inside each parquet
    row's ``segments`` JSON:
        {"text": "...", "span": [start_sec, end_sec]}
    """
    if not segment_timestamps or not changed_segments:
        return []

    changed_indexes = set()
    for changed in changed_segments:
        if not isinstance(changed, dict) or changed.get("segment_idx") is None:
            continue
        try:
            changed_indexes.add(int(changed["segment_idx"]))
        except (TypeError, ValueError):
            continue
    changed_texts = {
        _normalize_changed_text(seg.get("segment_text"))
        for seg in changed_segments
        if isinstance(seg, dict)
    }
    changed_texts.discard("")

    spans: List[Dict[str, Any]] = []
    for index, segment in enumerate(segment_timestamps):
        segment_text = str(segment.get("text", ""))
        matches_index = index in changed_indexes
        matches_text = (
            not changed_indexes
            and _normalize_changed_text(segment_text) in changed_texts
        )
        if not matches_index and not matches_text:
            continue

        try:
            start_sec = float(segment["start_sec"])
            end_sec = float(segment["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        spans.append(
            {
                "text": segment_text.strip(),
                "span": [start_sec, end_sec],
            }
        )
    return spans


# ---------------------------------------------------------------------------
# WER evaluation
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# UTMOS quality evaluation
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Speaker similarity evaluation
# ---------------------------------------------------------------------------








def _alignment_list(response: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    alignments = response.get("alignments") if isinstance(response, dict) else None
    return alignments if isinstance(alignments, list) else None


def compute_pair_wdtw(
    *,
    pair: AssembledEditPair,
    source_tagged_text: str,
    target_tagged_text: str,
    wdtw_client: Optional[WDTWClient],
    aligner_language: Optional[str],
) -> Optional[Dict[str, Any]]:
    if wdtw_client is None:
        return None

    source_alignments = _alignment_list(pair.source_alignment)
    target_alignments = _alignment_list(pair.target_alignment)
    return wdtw_client.compute(
        source_audio=pair.source_audio,
        target_audio=pair.target_audio,
        source_transcript=strip_tags(source_tagged_text),
        target_transcript=strip_tags(target_tagged_text),
        edit_st_inst=pair.st_instruction,
        source_alignments=source_alignments,
        target_alignments=target_alignments,
        language=alignment_language_code(aligner_language),
        mode="preserved",
        include_segments=False,
    )


# ---------------------------------------------------------------------------
# Full evaluation of an edit pair
# ---------------------------------------------------------------------------


def evaluate_synthesis_result(
    synthesis_result: SynthesisResult,
    *,
    wer_url: str,
    asr_url: str = "http://localhost:8010",
    utmos_url: str = "http://localhost:8017",
    spk_sim_url: str = "http://localhost:8019",
    wdtw_url: Optional[str] = None,
    aligner_language: Optional[str] = None,
    skip_quality: bool = False,
    skip_spk_sim: bool = False,
    skip_wdtw: bool = False,
    speaker_prompt_bytes: Optional[bytes] = None,
    parallel: bool = True,
) -> Dict:
    """Collect metrics for all final audios and edit pairs in a synthesis result.

    Uses caching to avoid redundant evaluation for duplicate audio content.

    When *parallel* is True (default), the independent per-audio evaluation
    axes run concurrently via ThreadPoolExecutor.
    Set to False for deterministic sequential execution (useful for debugging).

    Returns:
        Dict with per-pair metric payloads and timing.
    """
    t0 = time.time()

    asr_client = ASRClient(asr_url)
    wer_client = WERClient(wer_url)
    utmos_client = UTMOSClient(utmos_url) if not skip_quality else None
    spk_sim_client = SpkSimClient(spk_sim_url) if not skip_spk_sim else None
    wdtw_client = WDTWClient(wdtw_url) if wdtw_url and not skip_wdtw else None

    def _pair_side_audio(pair: AssembledEditPair, side: str) -> EvaluableAudio:
        if side == "source":
            tagged_text = synthesis_result.synthesized_audios[
                pair.source_id
            ].tagged_text
            return _build_evaluable_audio(pair.source_audio, tagged_text)
        tagged_text = synthesis_result.synthesized_audios[pair.target_id].tagged_text
        return _build_evaluable_audio(pair.target_audio, tagged_text)

    unique_audios: Dict[str, EvaluableAudio] = {}
    for pair in synthesis_result.edit_pairs:
        for side in ("source", "target"):
            audio = _pair_side_audio(pair, side)
            unique_audios.setdefault(audio.audio_hash, audio)

    ordered_hashes = sorted(unique_audios)
    audio_eval_cache: Dict[str, Dict] = {}

    for pair in synthesis_result.edit_pairs:
        src_audio = _pair_side_audio(pair, "source")
        tgt_audio = _pair_side_audio(pair, "target")
        if pair.source_segment_timestamps:
            audio_eval_cache.setdefault(src_audio.audio_hash, {})[
                "segment_timestamps"
            ] = pair.source_segment_timestamps
        if pair.target_segment_timestamps:
            audio_eval_cache.setdefault(tgt_audio.audio_hash, {})[
                "segment_timestamps"
            ] = pair.target_segment_timestamps

    # --- Parallel evaluation of independent per-audio axes ---
    asr_time = qual_time = span_time = 0.0

    def _eval_asr():
        """Batch ASR → audio_eval_cache.content."""
        t = time.time()
        if asr_client is not None and ordered_hashes:
            audio_list = [
                unique_audios[audio_hash].audio_bytes for audio_hash in ordered_hashes
            ]
            languages = [
                asr_language_name(None, text=unique_audios[audio_hash].expected_text)
                for audio_hash in ordered_hashes
            ]
            batch_results = asr_client.transcribe_batch(
                audio_list,
                languages=languages,
                timeout=300,
            )
            if len(batch_results) != len(ordered_hashes):
                raise ValueError("ASR batch response count does not match request")
            for audio_hash, asr_res in zip(ordered_hashes, batch_results):
                transcript = asr_res["text"]
                expected_text = unique_audios[audio_hash].expected_text
                wer_result = compute_wer(expected_text, transcript, wer_client=wer_client)
                audio_eval_cache.setdefault(audio_hash, {})["content"] = {
                    "transcript": transcript,
                    "wer": wer_result,
                }
        return time.time() - t

    def _eval_edit_time_spans():
        """Ensure final-audio segment timestamps exist for edit span metadata."""
        t = time.time()
        if not any(pair.changed_segments for pair in synthesis_result.edit_pairs):
            return time.time() - t

        needed_hashes: set[str] = set()
        for pair in synthesis_result.edit_pairs:
            if not pair.changed_segments:
                continue
            for side in ("source", "target"):
                needed_hashes.add(_pair_side_audio(pair, side).audio_hash)

        for audio_hash in sorted(needed_hashes):
            cached = audio_eval_cache.setdefault(audio_hash, {})
            if "segment_timestamps" not in cached:
                raise ValueError("Missing cached segment timestamps from Synth alignment")
        return time.time() - t

    def _eval_quality():
        """Per-audio UTMOS → audio_eval_cache.quality."""
        t = time.time()
        if utmos_client is not None:
            for audio_hash in ordered_hashes:
                audio_eval_cache.setdefault(audio_hash, {})["quality"] = (
                    evaluate_quality(
                        unique_audios[audio_hash].audio_bytes,
                        utmos_client,
                    )
                )
        return time.time() - t

    # Run per-audio axes concurrently (or sequentially if parallel=False).
    if parallel:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="eval") as executor:
            futures = {
                executor.submit(_eval_asr): "asr",
                executor.submit(_eval_quality): "quality",
            }
            for future in as_completed(futures):
                axis = futures[future]
                elapsed = future.result()
                if axis == "asr":
                    asr_time = elapsed
                elif axis == "quality":
                    qual_time = elapsed
    else:
        asr_time = _eval_asr()
        qual_time = _eval_quality()

    span_time = _eval_edit_time_spans()

    n_unique = len(ordered_hashes)
    n_pairs = len(synthesis_result.edit_pairs)
    mode_label = "parallel" if parallel else "sequential"
    logger.info(
        f"Eval ({mode_label}): {n_unique} unique audios for {n_pairs} pairs | "
        f"ASR={asr_time:.1f}s, Quality={qual_time:.1f}s, "
        f"Spans={span_time:.1f}s"
    )

    # --- Assemble per-pair evaluations from cache ---
    pair_evals = []
    acoustic_time = 0.0
    for pair in synthesis_result.edit_pairs:
        src_synth = synthesis_result.synthesized_audios.get(pair.source_id)
        tgt_synth = synthesis_result.synthesized_audios.get(pair.target_id)
        if src_synth is None or tgt_synth is None:
            raise ValueError(f"Missing synthesized audio for {pair.source_id}->{pair.target_id}")

        src_audio = _pair_side_audio(pair, "source")
        tgt_audio = _pair_side_audio(pair, "target")
        src_eval = audio_eval_cache.get(src_audio.audio_hash, {})
        tgt_eval = audio_eval_cache.get(tgt_audio.audio_hash, {})

        src_payload = {}
        if src_eval.get("content") is not None:
            src_payload.update(src_eval["content"])
        if src_eval.get("quality") is not None:
            src_payload["utmos"] = src_eval["quality"]
        source_edit_spans = build_edit_time_spans_from_segment_timestamps(
            src_eval.get("segment_timestamps", []),
            pair.changed_segments,
        )
        src_payload["edit_time_spans"] = source_edit_spans

        tgt_payload = {}
        if tgt_eval.get("content") is not None:
            tgt_payload.update(tgt_eval["content"])
        if tgt_eval.get("quality") is not None:
            tgt_payload["utmos"] = tgt_eval["quality"]
        target_edit_spans = build_edit_time_spans_from_segment_timestamps(
            tgt_eval.get("segment_timestamps", []),
            pair.changed_segments,
        )
        tgt_payload["edit_time_spans"] = target_edit_spans

        eval_payload: Dict[str, Any] = {
            "version": "v3",
            "src": src_payload,
            "tgt": tgt_payload,
        }
        acoustic_started = time.time()
        eval_payload[EMOTION_ACOUSTIC_EFFECT_KEY] = (
            compute_emotion_acoustic_effect(
                source_audio_bytes=pair.source_audio,
                target_audio_bytes=pair.target_audio,
                source_alignment=pair.source_alignment,
                target_alignment=pair.target_alignment,
                source_edit_spans=source_edit_spans,
                target_edit_spans=target_edit_spans,
                changed_segments=pair.changed_segments,
            )
        )
        acoustic_time += time.time() - acoustic_started

        if spk_sim_client is not None:
            eval_payload["spk_sim"] = evaluate_speaker_similarity(
                pair.source_audio,
                pair.target_audio,
                spk_sim_client,
            )

        wdtw_result = compute_pair_wdtw(
            pair=pair,
            source_tagged_text=src_synth.tagged_text,
            target_tagged_text=tgt_synth.tagged_text,
            wdtw_client=wdtw_client,
            aligner_language=aligner_language,
        )
        if wdtw_result is not None:
            eval_payload["wdtw_result"] = wdtw_result
            if "wdtw" in wdtw_result and "error" not in wdtw_result:
                eval_payload["wdtw"] = wdtw_result["wdtw"]

        pair_evals.append(
            {
                "source_id": pair.source_id,
                "target_id": pair.target_id,
                "eval_results": eval_payload,
            }
        )

    elapsed = time.time() - t0
    for pair_eval in pair_evals:
        pair_eval["eval_results"]["eval_time_sec"] = elapsed

    logger.info("Pair acoustic-effect evaluation completed in %.1fs", acoustic_time)
    return {
        "pair_evaluations": pair_evals,
        "eval_time_sec": elapsed,
    }
