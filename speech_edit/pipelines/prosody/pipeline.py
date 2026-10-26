"""Prosody sample editing and evaluation shared by workers and the HTTP API."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

log_suffix = uuid.uuid4().hex[:6]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prosody_edit_pipeline")

from speech_edit.clients.services import AlignmentClient, ASRClient, F5TTSClient, F0EvalClient, PitchConversionClient, SpkSimClient, SpeechRateClient, UTMOSClient, WERClient, WDTWClient
from speech_edit.data.language import alignment_language_code  # noqa: E402

from .editor import SMOOTH_POLICY_CHOICES
from .editor import apply_edits_from_service, numpy_to_wav
from speech_edit.evaluation.audio import PairEvalResult, evaluate_pair  # noqa: E402
from .input_generator import PipelineInput, normalize_sample_language
from .sample_output import build_sample_metadata, pair_dir_name, serialize_edit_result, serialize_pair_eval
from .tag_utils import ORIGINAL_WAVE_ID  # noqa: E402
from .tag_utils import PITCH_TAG, parse_tags  # noqa: E402

PIPELINE_VERSION = "2.0.0"
PIPELINE_TYPE = "prosody_edit"


def _tagged_text_has_pitch(tagged_text: str) -> bool:
    return any(segment.tag_type == PITCH_TAG for segment in parse_tags(tagged_text))


def _f5_for_wave(
    *,
    tagged_text: str,
    f5_client: Optional[F5TTSClient],
    smooth_policy: str,
) -> Optional[F5TTSClient]:
    if f5_client is None or smooth_policy == "never":
        return None
    if smooth_policy == "always":
        return f5_client
    if smooth_policy == "auto":
        return f5_client if _tagged_text_has_pitch(tagged_text) else None
    raise ValueError(
        f"smooth_policy must be one of {SMOOTH_POLICY_CHOICES}, got {smooth_policy!r}"
    )


def _aligner_language(sample_language: Optional[str]) -> str:
    return alignment_language_code(sample_language)


def _pipeline_input_language(pipeline_input: PipelineInput) -> Optional[str]:
    return normalize_sample_language(pipeline_input.metadata.get("language"))


def _wave_alignment_for_pair_side(
    *,
    side_id: str,
    pair_wave_id: Optional[str],
    wave_results: Dict[str, Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    if side_id != ORIGINAL_WAVE_ID or not pair_wave_id:
        return None
    result = wave_results.get(str(pair_wave_id), {}).get("result")
    if not isinstance(result, dict):
        return None
    alignment = result.get("alignment")
    if not isinstance(alignment, dict):
        return None
    alignments = alignment.get("alignments")
    return alignments if isinstance(alignments, list) else None


def _edited_wave_id_for_pair(source_id: str, target_id: str) -> Optional[str]:
    if source_id == ORIGINAL_WAVE_ID and target_id != ORIGINAL_WAVE_ID:
        return target_id
    if target_id == ORIGINAL_WAVE_ID and source_id != ORIGINAL_WAVE_ID:
        return source_id
    return None


def compute_pair_wdtw_if_enabled(
    *,
    pair,
    source_audio_bytes: bytes,
    target_audio_bytes: bytes,
    source_text: str,
    target_text: str,
    sample_language: Optional[str],
    wave_results: Dict[str, Dict[str, Any]],
    wdtw_client: Optional[WDTWClient],
) -> Optional[Dict[str, Any]]:
    if wdtw_client is None:
        return None
    pair_wave_id = _edited_wave_id_for_pair(pair.source_id, pair.target_id)
    source_alignments = _wave_alignment_for_pair_side(
        side_id=pair.source_id,
        pair_wave_id=pair_wave_id,
        wave_results=wave_results,
    )
    target_alignments = _wave_alignment_for_pair_side(
        side_id=pair.target_id,
        pair_wave_id=pair_wave_id,
        wave_results=wave_results,
    )
    return wdtw_client.compute(
        source_audio=source_audio_bytes,
        target_audio=target_audio_bytes,
        source_transcript=source_text,
        target_transcript=target_text,
        edit_st_inst=pair.instructions.get("st"),
        source_alignments=source_alignments,
        target_alignments=target_alignments,
        language=_aligner_language(sample_language),
        mode="preserved",
        include_segments=False,
    )


def compute_pair_region_effects_if_enabled(
    *,
    pair,
    source_audio_bytes: bytes,
    target_audio_bytes: bytes,
    wave_results: Dict[str, Dict[str, Any]],
    f0_client: Optional[F0EvalClient],
) -> Optional[Dict[str, Any]]:
    if f0_client is None:
        return None
    from speech_edit.pipelines.prosody.effect_metrics import (
        compute_region_effect_metrics,
        spans_from_alignment_regions,
        spans_from_applied_edits,
    )

    pair_wave_id = _edited_wave_id_for_pair(pair.source_id, pair.target_id)
    if not pair_wave_id:
        return None
    wave_result = wave_results.get(str(pair_wave_id), {}).get("result")
    if not isinstance(wave_result, dict):
        raise ValueError(f"missing wave result for {pair_wave_id}")
    alignment = wave_result.get("alignment")
    if not isinstance(alignment, dict):
        alignment = {}
    original_spans = spans_from_alignment_regions(alignment.get("edit_regions"))
    generated_spans = spans_from_applied_edits(
        wave_result.get("edits_applied", []),
        "target_span",
    )
    if pair.source_id == ORIGINAL_WAVE_ID:
        source_spans = original_spans
        target_spans = generated_spans
    else:
        source_spans = generated_spans
        target_spans = original_spans
    tagged_text = str(pair.instructions.get("st") or "")
    return compute_region_effect_metrics(
        source_audio_bytes=source_audio_bytes,
        target_audio_bytes=target_audio_bytes,
        tagged_text=tagged_text,
        source_spans=source_spans,
        target_spans=target_spans,
        f0_client=f0_client,
    )


def edit_sample(
    pipeline_input: PipelineInput,
    *,
    aligner_client: AlignmentClient,
    aligner_backend: str,
    pitch_method: str,
    rate_method: str,
    psola_indextts2_cco_pitch_boost: str,
    pitch_client: Optional[PitchConversionClient],
    rate_client: Optional[SpeechRateClient],
    f5_client: Optional[F5TTSClient],
    crossfade_ms: float,
    smooth_margin_sec: float,
    smooth_policy: str,
    f5_steps: int,
    f5_seed: Optional[int],
    f5_source_crossfade_sec: float,
    f5_regen_crossfade_sec: float,
    t_start: float,
) -> Dict[str, Any]:
    """Run Stage II editing and build pair audio inputs for evaluation."""
    audio_bytes = pipeline_input.audio_bytes
    wave_group = pipeline_input.wave_group
    sample_id = pipeline_input.metadata.get("input_id") or f"sample_{uuid.uuid4().hex[:8]}"
    sample_language = _pipeline_input_language(pipeline_input)

    logger.info("--- Stage II: Multi-wave Editing ---")
    wave_results: Dict[str, Dict[str, Any]] = {}
    wave_audio_bytes: Dict[str, bytes] = {}
    for wave_id, tagged_text in wave_group.tagged_texts.items():
        edit_result = apply_edits_from_service(
            audio_bytes,
            tagged_text,
            aligner_client,
            sample_rate=pipeline_input.sample_rate,
            aligner_backend=aligner_backend,
            language=_aligner_language(sample_language),
            pitch_method=pitch_method,
            rate_method=rate_method,
            psola_indextts2_cco_pitch_boost=psola_indextts2_cco_pitch_boost,
            pitch_client=pitch_client,
            rate_client=rate_client,
            f5_client=_f5_for_wave(
                tagged_text=tagged_text,
                f5_client=f5_client,
                smooth_policy=smooth_policy,
            ),
            crossfade_ms=crossfade_ms,
            smooth_margin_sec=smooth_margin_sec,
            f5_steps=f5_steps,
            f5_seed=f5_seed,
            f5_source_crossfade_sec=f5_source_crossfade_sec,
            f5_regen_crossfade_sec=f5_regen_crossfade_sec,
        )

        edited_audio_bytes = numpy_to_wav(
            edit_result.edited_audio, edit_result.sample_rate
        )
        wave_audio_bytes[wave_id] = edited_audio_bytes
        wave_results[wave_id] = {
            "wave_id": wave_id,
            "tagged_text": tagged_text,
            "plain_text": wave_group.plain_text,
            "edited_duration": len(edit_result.edited_audio) / edit_result.sample_rate,
            "result": serialize_edit_result(edit_result),
        }
        logger.info(
            "Wave %s complete: %d edits, %.2fs -> %.2fs",
            wave_id,
            len(edit_result.edits_applied),
            pipeline_input.duration,
            wave_results[wave_id]["edited_duration"],
        )

    pair_inputs = []
    for pair in wave_group.edit_pairs:
        src_bytes = (
            audio_bytes
            if pair.source_id == ORIGINAL_WAVE_ID
            else wave_audio_bytes[pair.source_id]
        )
        tgt_bytes = (
            audio_bytes
            if pair.target_id == ORIGINAL_WAVE_ID
            else wave_audio_bytes[pair.target_id]
        )
        pair_inputs.append((pair, src_bytes, tgt_bytes))

    return {
        "pipeline_input": pipeline_input,
        "sample_id": sample_id,
        "t_start": t_start,
        "wave_results": wave_results,
        "wave_audio_bytes": wave_audio_bytes,
        "pair_inputs": pair_inputs,
    }


def evaluate_sample(
    stage2_state: Dict[str, Any],
    *,
    asr_client: Optional[ASRClient],
    wer_client: Optional[WERClient],
    utmos_client: Optional[UTMOSClient],
    spk_sim_client: Optional[SpkSimClient],
    do_evaluate: bool,
    wdtw_client: Optional[WDTWClient] = None,
    f0_client: Optional[F0EvalClient] = None,
) -> Dict[str, Any]:
    """Evaluate edited pairs and build sample metadata."""
    pipeline_input: PipelineInput = stage2_state["pipeline_input"]
    wave_group = pipeline_input.wave_group
    pair_inputs = stage2_state["pair_inputs"]
    sample_language = _pipeline_input_language(pipeline_input)

    logger.info("--- Stage III: Pair Evaluation (parallel) ---")
    t_eval_start = time.time()
    eval_results_map: Dict[int, Optional[PairEvalResult]] = {}
    if do_evaluate and pair_inputs:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _eval_one(idx, pair, src_bytes, tgt_bytes):
            return idx, evaluate_pair(
                source_audio_bytes=src_bytes,
                edited_audio_bytes=tgt_bytes,
                reference_text=wave_group.plain_text,
                asr_client=asr_client,
                wer_client=wer_client,
                utmos_client=utmos_client,
                spk_sim_client=spk_sim_client,
                parallel=True,
                sample_language=sample_language,
            )

        with ThreadPoolExecutor(
            max_workers=len(pair_inputs),
            thread_name_prefix="pair_eval",
        ) as pool:
            futures = {
                pool.submit(_eval_one, i, p, s, t): i
                for i, (p, s, t) in enumerate(pair_inputs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                pair = pair_inputs[idx][0]
                _, eval_result = future.result()
                eval_results_map[idx] = eval_result
                logger.info(
                    (
                        "Pair %s->%s eval: srcErr=%.2f%% tgtErr=%.2f%% "
                        "UTMOS=%.2f SpkSim=%.3f"
                    ),
                    pair.source_id,
                    pair.target_id,
                    eval_result.source_metrics.wer_pct,
                    eval_result.target_metrics.wer_pct,
                    eval_result.target_metrics.utmos_score,
                    eval_result.target_metrics.spk_similarity,
                )

    eval_wall_time = time.time() - t_eval_start
    logger.info(
        "Stage III complete: %d pairs evaluated in %.1fs (parallel)",
        len(pair_inputs),
        eval_wall_time,
    )

    pair_results: List[Dict[str, Any]] = []
    for i, (pair, src_bytes, tgt_bytes) in enumerate(pair_inputs):
        eval_result = eval_results_map.get(i)
        wdtw_result = (
            compute_pair_wdtw_if_enabled(
                pair=pair,
                source_audio_bytes=src_bytes,
                target_audio_bytes=tgt_bytes,
                source_text=pair.plain_text,
                target_text=pair.plain_text,
                sample_language=sample_language,
                wave_results=stage2_state.get("wave_results", {}),
                wdtw_client=wdtw_client,
            )
            if do_evaluate
            else None
        )
        serialized_eval = serialize_pair_eval(eval_result)
        if wdtw_result is not None:
            serialized_eval["wdtw_result"] = wdtw_result
            if "wdtw" in wdtw_result:
                serialized_eval["wdtw"] = wdtw_result.get("wdtw")
        effect_result = (
            compute_pair_region_effects_if_enabled(
                pair=pair,
                source_audio_bytes=src_bytes,
                target_audio_bytes=tgt_bytes,
                wave_results=stage2_state.get("wave_results", {}),
                f0_client=f0_client,
            )
            if do_evaluate
            else None
        )
        if effect_result is not None:
            serialized_eval.update(effect_result)
        pair_metadata = pair.to_metadata_dict()
        pair_results.append(
            {
                "pair_dir_name": pair_dir_name(pair),
                "source_id": pair.source_id,
                "target_id": pair.target_id,
                "source_audio_bytes": src_bytes,
                "target_audio_bytes": tgt_bytes,
                "instructions": pair_metadata["instructions"],
                "source_tagged_text": pair.source_tagged_text,
                "target_tagged_text": pair.target_tagged_text,
                "tag_diff": pair_metadata.get("tag_diff", []),
                "eval_results": serialized_eval,
            }
        )

    metadata = build_sample_metadata(
        stage2_state["sample_id"],
        pipeline_input,
        stage2_state["wave_results"],
        pair_results,
        pipeline_version=PIPELINE_VERSION,
        pipeline_type=PIPELINE_TYPE,
    )
    metadata["total_time_sec"] = time.time() - stage2_state["t_start"]
    return {
        **stage2_state,
        "pair_results": pair_results,
        "metadata": metadata,
    }


def make_input_seed(seed: int, attempt_index: int, retry_index: int = 0) -> int:
    base_seed = 0 if seed is None else int(seed)
    return (
        (base_seed * 6364136223846793005)
        + (attempt_index * 1442695040888963407)
        + (retry_index * 22695477)
        + 0x9E3779B97F4A7C15
    ) & ((1 << 63) - 1)
