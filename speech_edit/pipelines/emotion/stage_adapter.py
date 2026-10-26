"""Stage-runtime adapter for the emotion/style edit pipeline."""

from __future__ import annotations

import uuid
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from speech_edit.runtime.adapters import PipelineAdapter
from speech_edit.runtime.config import pipeline_config
from speech_edit.runtime.models import AdapterContext, OutputItem, StageResult
from speech_edit.data.parquet.reader import (
    Segment,
    read_segments_from_parquet_payload,
)

from .parquet_rows import EmotionStyleParquetRowProcessor
from .speaker_sources import initialize_language_speaker_samplers
from .synthesis_profile import (
    SYNTHESIS_BACKEND_INDEX,
    SYNTHESIS_BACKEND_TED,
    normalize_synthesis_backend,
    select_profile_and_segments,
)


SYNTH_BACKEND_OUTPUT_QUEUES = {
    SYNTHESIS_BACKEND_INDEX: "emotion.plan_to_synth.index",
    SYNTHESIS_BACKEND_TED: "emotion.plan_to_synth.ted",
}


def _segments_from_parquet_payload(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> List[Segment]:
    return read_segments_from_parquet_payload(payload, cfg)


def _segment_key(segment: Segment, fallback: str) -> str:
    meta = dict(segment.metadata or {})
    source_path = meta.get("source_parquet_path", "")
    row_idx = meta.get("source_row_idx", meta.get("source_row_index", meta.get("row_idx", "")))
    if source_path or row_idx != "":
        return f"{source_path}:{row_idx}:{segment.clip_id}"
    return f"{fallback}:{segment.clip_id}"


def _batch_key(segments: List[Segment], fallback: str) -> str:
    if not segments:
        return fallback
    joined = "|".join(
        _segment_key(segment, f"{fallback}:{idx}") for idx, segment in enumerate(segments)
    )
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{fallback}:{digest}"


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _validation_errors_from_exception(exc: BaseException) -> List[str]:
    raw = getattr(exc, "validation_errors", None)
    if raw:
        return [str(item) for item in raw]
    return [f"{type(exc).__name__}: {exc}"]


def _required_output_queue(ctx: AdapterContext) -> str:
    if not ctx.stage.output_queue:
        raise RuntimeError(f"stage {ctx.stage.name!r} must declare output_queue")
    return ctx.stage.output_queue


def _synth_output_queue_for_backend(backend: str) -> str:
    normalized = normalize_synthesis_backend(backend)
    try:
        return SYNTH_BACKEND_OUTPUT_QUEUES[normalized]
    except KeyError as exc:  # pragma: no cover - normalize guards this.
        raise ValueError(f"Unsupported synthesis backend for routing: {backend!r}") from exc


def _synthesis_backend_from_payload(payload: Mapping[str, Any]) -> str:
    group = payload.get("group")
    metadata = getattr(group, "metadata", {}) if group is not None else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    profile = metadata.get("synthesis_profile")
    if not isinstance(profile, Mapping):
        profile = {}
    backend = metadata.get("synthesis_backend") or profile.get("synthesis_backend")
    return normalize_synthesis_backend(str(backend or "index"))


def _synth_backend_filter(stage_name: str, cfg: Mapping[str, Any]) -> Optional[str]:
    configured = cfg.get("synthesis_backend_filter")
    if configured:
        return normalize_synthesis_backend(str(configured))
    if stage_name == "synth_index":
        return SYNTHESIS_BACKEND_INDEX
    if stage_name == "synth_ted":
        return SYNTHESIS_BACKEND_TED
    return None


def _select_utterance_batch_size(cfg: Mapping[str, Any], payload: Any) -> int:
    explicit = cfg.get("utterance_batch_size")
    if explicit is not None:
        return max(2, int(explicit))
    dist = cfg["utterance_batch_size_dist"]
    if not isinstance(dist, Mapping) or not dist:
        raise ValueError("utterance_batch_size_dist must be a non-empty mapping")
    sizes = [int(size) for size in dist.keys()]
    weights = [float(weight) for weight in dist.values()]
    key = str(payload.get("parquet_path", "")) if isinstance(payload, Mapping) else ""
    seed = _stable_seed(int(cfg['seed']), key or "utterance_batch_size")
    return max(2, random.Random(seed).choices(sizes, weights=weights, k=1)[0])


def _normalize_speaker_sources(raw_sources: Any) -> Optional[Tuple[str, ...]]:
    if raw_sources is None or raw_sources == "":
        return None
    if isinstance(raw_sources, (str, Path)):
        return (str(raw_sources),)
    if isinstance(raw_sources, (list, tuple)) or (
        hasattr(raw_sources, "__iter__") and not isinstance(raw_sources, Mapping)
    ):
        return tuple(str(source) for source in raw_sources if str(source))
    raise TypeError("pipeline_config.speaker_parquet_sources must be a string or list")


def _segment_language(segment: Segment, cfg: Mapping[str, Any]) -> Optional[str]:
    meta = dict(segment.metadata or {})
    language = meta.get("language") or meta.get("sample_language") or cfg['sample_language']
    return str(language) if language else None


def _sample_speaker_prompt(
    speaker_samplers: Mapping[str, Any],
    language: Optional[str],
) -> Segment:
    if not language:
        raise ValueError("speaker_parquet_sources requires language metadata or sample_language")
    normalized = str(language).lower()
    if normalized in {"zho", "zh-cn", "zh_cn", "chinese"}:
        normalized = "zh"
    elif normalized in {"eng", "english"}:
        normalized = "en"
    sampler = speaker_samplers.get(normalized)
    if sampler is None:
        available = ", ".join(sorted(speaker_samplers.keys()))
        raise ValueError(
            f"No speaker parquet sampler configured for language {language!r}; "
            f"available languages: {available}"
        )
    return sampler.sample()


class EmotionStyleStageAdapter(PipelineAdapter):
    name = "emotion_style_edit"

    def __init__(self) -> None:
        self._speaker_sampler_cache: Dict[Tuple[Tuple[str, ...], int], Mapping[str, Any]] = {}


    def process(self, stage_name: str, payload: Any, ctx: AdapterContext) -> StageResult:
        from speech_edit.clients.services import LLMClient
        from speech_edit.pipelines.emotion.evaluator import (
            evaluate_synthesis_result,
        )
        from speech_edit.pipelines.emotion.input_generator import (
            generate_tagged_text_group_attempt,
        )
        from speech_edit.pipelines.emotion.synthesizer import (
            synthesize_tagged_text_group,
        )

        cfg = ctx.pipeline_config
        services = ctx.services
        if stage_name == "sample":
            segments = _segments_from_parquet_payload(payload, cfg)
            batch_size = _select_utterance_batch_size(cfg, payload)
            speaker_sources = _normalize_speaker_sources(cfg['speaker_parquet_sources'])
            speaker_samplers = None
            if speaker_sources:
                cache_key = (speaker_sources, int(cfg['seed']))
                speaker_samplers = self._speaker_sampler_cache.get(cache_key)
                if speaker_samplers is None:
                    speaker_samplers = initialize_language_speaker_samplers(
                        list(speaker_sources),
                        seed=int(cfg['seed']),
                        log_prefix=f"[{ctx.worker_id}] ",
                    )
                    self._speaker_sampler_cache[cache_key] = speaker_samplers or {}
            outputs: List[OutputItem] = []
            source_step = max(batch_size - 1, 1) if speaker_samplers else batch_size
            for idx in range(0, len(segments), source_step):
                batch = segments[idx : idx + source_step]
                if speaker_samplers:
                    if not batch:
                        continue
                    language = _segment_language(batch[0], cfg)
                    speaker_prompt = _sample_speaker_prompt(speaker_samplers, language)
                    gen_segments = list(batch)
                    key = _batch_key(gen_segments, f"emotion-{idx}")
                else:
                    if len(batch) < 2:
                        continue
                    speaker_prompt = min(batch, key=lambda seg: seg.duration)
                    gen_segments = [seg for seg in batch if seg is not speaker_prompt]
                    language = _segment_language(speaker_prompt, cfg)
                    key = _segment_key(speaker_prompt, f"emotion-{idx}")
                language = (
                    language
                    or _segment_language(speaker_prompt, cfg)
                )
                profile_seed = _stable_seed(int(cfg['seed']), f"{key}:profile")
                synthesis_profile, selected_gen_segments = select_profile_and_segments(
                    gen_segments,
                    rng=random.Random(profile_seed),
                    emotion_condition_mode=cfg['emotion_condition_mode'],
                    synthesis_mode=cfg['synthesis_mode'],
                    synthesis_profile_weights=cfg['synthesis_profile_weights'],
                    language=language,
                    plan_text_unit_cap_en=cfg['plan_text_unit_cap_en'],
                    plan_text_unit_cap_zh=cfg['plan_text_unit_cap_zh'],
                    duration_estimation_en_word_sec=cfg['duration_estimation_en_word_sec'],
                    duration_estimation_cjk_char_sec=cfg['duration_estimation_cjk_char_sec'],
                    ted_estimated_duration_soft_limit_sec=cfg['ted_estimated_duration_soft_limit_sec'],
                    ted_max_generated_duration_sec=cfg['ted_max_generated_duration_sec'],
                )
                profile_key = (
                    f"{key}:"
                    f"{synthesis_profile['synthesis_mode']}:"
                    f"{synthesis_profile['emotion_condition_mode']}:"
                    f"{synthesis_profile.get('selected_text_window', {}).get('start_unit', 0)}:"
                    f"{synthesis_profile.get('selected_text_window', {}).get('end_unit', 0)}"
                )
                outputs.append(
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=profile_key,
                        payload={
                            "sample_id": f"sample_{uuid.uuid5(uuid.NAMESPACE_URL, profile_key).hex[:12]}",
                            "stage1_seed": _stable_seed(int(cfg['seed']), profile_key),
                            "speaker_prompt": speaker_prompt,
                            "gen_segments": selected_gen_segments,
                            "language": language,
                            "synthesis_profile": synthesis_profile,
                        },
                    )
                )
            return StageResult(outputs=outputs)
        if stage_name == "plan":
            llm_client = LLMClient(services.url("llm"))
            max_retries = max(1, int(cfg['llm_max_retries']))
            validation_errors: List[str] = []
            group = None
            for attempt in range(1, max_retries + 1):
                try:
                    group = generate_tagged_text_group_attempt(
                        speaker_prompt=payload["speaker_prompt"],
                        gen_segments=payload["gen_segments"],
                        language=payload.get("language"),
                        llm_client=llm_client,
                        stage1_seed=int(payload.get("stage1_seed", cfg['seed'])),
                        natural_language_instruction_mode=str(
                            cfg['natural_language_instruction_mode']
                        ),
                        rewrite_prob=float(cfg['rewrite_prob']),
                        emotion_edit_mode_probs=cfg['emotion_edit_mode_probs'],
                        emotion_tag_weights=cfg['emotion_tag_weights'],
                        strict_emotion_plan=bool(cfg['strict_emotion_plan']),
                        emotion_region_count_weights=cfg['emotion_region_count_weights'],
                        synthesis_profile=payload["synthesis_profile"],
                        max_stage1_text_words_en=cfg['plan_text_unit_cap_en'],
                        max_stage1_text_words_zh=cfg['plan_text_unit_cap_zh'],
                        attempt=attempt,
                        validation_errors=tuple(validation_errors),
                    )
                    break
                except Exception as exc:
                    validation_errors.extend(_validation_errors_from_exception(exc))
                    if attempt >= max_retries:
                        raise
            if group is None:
                raise RuntimeError("plan stage did not produce a tagged text group")
            sample_id = payload["sample_id"]
            metadata = getattr(group, "metadata", {}) or {}
            profile = metadata.get("synthesis_profile") if isinstance(metadata, Mapping) else {}
            backend = (
                metadata.get("synthesis_backend")
                if isinstance(metadata, Mapping)
                else None
            ) or (
                profile.get("synthesis_backend")
                if isinstance(profile, Mapping)
                else None
            )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_synth_output_queue_for_backend(str(backend or "index")),
                        item_key=str(sample_id),
                        payload={"sample_id": sample_id, "group": group},
                    )
                ]
            )
        if stage_name in {"synth", "synth_index", "synth_ted"}:
            backend_filter = _synth_backend_filter(stage_name, cfg)
            if backend_filter is not None:
                payload_backend = _synthesis_backend_from_payload(payload)
                if payload_backend != backend_filter:
                    raise ValueError(
                        f"stage {stage_name!r} only accepts backend={backend_filter}; "
                        f"got backend={payload_backend}"
                    )
            synth_result = synthesize_tagged_text_group(
                payload["group"],
                assembly_f5_steps=int(cfg["assembly_f5_steps"]),
                indextts_url=services.optional_url("index_tts"),
                ted_tts_url=services.optional_url("ted_tts"),
                boundary_pause_smoother_url=services.url("boundary_pause_smoother"),
                audio_assembly_url=services.url("audio_assembly"),
                aligner_url=services.url("aligner"),
                max_candidates=int(cfg['max_candidates']),
                aligner_backend=str(cfg['aligner_backend']),
                debug_synthesize=bool(cfg['debug_synthesize']),
                debug_root_dir=None,
            )
            sample_id = payload["sample_id"]
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=str(sample_id),
                        payload={
                            "sample_id": sample_id,
                            "group": payload["group"],
                            "synth_result": synth_result,
                        },
                    )
                ]
            )
        if stage_name == "eval":
            eval_result = evaluate_synthesis_result(
                payload["synth_result"],
                asr_url=services.url("asr"),
                wer_url=services.url("wer"),
                utmos_url=services.url("utmos"),
                spk_sim_url=services.url("spk_sim"),
                wdtw_url=services.url("wdtw"),
                aligner_language=(payload["group"].metadata or {}).get("language"),
                skip_quality=bool(cfg['skip_quality_eval']),
                skip_spk_sim=bool(cfg['skip_spk_sim_eval']),
                skip_wdtw=bool(cfg['skip_wdtw_eval']),
            )
            sample_id = str(payload["sample_id"])
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=sample_id,
                        payload={
                            "sample_id": sample_id,
                            "group": payload["group"],
                            "synth_result": payload["synth_result"],
                            "eval_result": eval_result,
                        },
                    )
                ]
            )
        if stage_name == "save":
            rows = EmotionStyleParquetRowProcessor().rows_from_payload(
                payload,
                dataset_name=ctx.stage.parquet_output.dataset_name,
            )
            return StageResult(parquet_rows=rows)
        raise ValueError(f"unknown stage for {self.name}: {stage_name}")
