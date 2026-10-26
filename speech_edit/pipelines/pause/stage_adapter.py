"""Distributed pipeline adapter for the pause edit pipeline."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from speech_edit.data.language import classify_text_language
from speech_edit.data.parquet.reader import (
    Segment,
    read_segments_from_parquet_payload,
)
from speech_edit.runtime.adapters import PipelineAdapter
from speech_edit.runtime.models import AdapterContext, OutputItem, StageResult

from .input_generator import (
    DURATION_POOL_SIZE,
    _normalize_plain_text,
    generate_profiles_with_retries,
    sample_pause_profile_targets,
    sample_duration_pool,
)
from .parquet_rows import PauseEditParquetRowProcessor
from .tag_utils import PauseEditGroup, PauseEditProfile


def _segments_from_parquet_payload(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> List[Segment]:
    return read_segments_from_parquet_payload(payload, cfg)


def _segment_key(segment: Segment, fallback: str) -> str:
    meta = dict(segment.metadata or {})
    source_path = meta.get("source_parquet_path", "")
    row_idx = meta.get("source_row_idx", meta.get("source_row_index", meta.get("row_idx", "")))
    if source_path or row_idx != "":
        return f"{source_path}:{row_idx}:{segment.clip_id}"
    return f"{fallback}:{segment.clip_id}"


def _stable_seed_index(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _required_output_queue(ctx: AdapterContext) -> str:
    if not ctx.stage.output_queue:
        raise RuntimeError(f"stage {ctx.stage.name!r} must declare output_queue")
    return ctx.stage.output_queue


@dataclass(frozen=True)
class PauseSampleWorkItem:
    sample_id: str
    segment: Segment
    attempt_index: int


@dataclass(frozen=True)
class PauseProfileWorkItem:
    sample_id: str
    normed_text: str
    profile: PauseEditProfile
    duration_pool: List[int]
    num_operations: int
    original_audio: bytes
    metadata: Dict[str, Any]

    def to_group(self) -> PauseEditGroup:
        return PauseEditGroup(
            normed_text=self.normed_text,
            profiles=[self.profile],
            duration_pool=list(self.duration_pool),
            num_operations=self.num_operations,
            original_audio=self.original_audio,
            metadata=dict(self.metadata or {}),
        )


@dataclass(frozen=True)
class PauseSynthesizedWorkItem:
    sample_id: str
    work_item: PauseProfileWorkItem
    synth_pair: Any


@dataclass(frozen=True)
class PauseEvaluatedWorkItem:
    sample_id: str
    work_item: PauseProfileWorkItem
    synth_pair: Any
    eval_result: Any | None


class PauseEditStageAdapter(PipelineAdapter):
    name = "pause_edit"


    def _generate_group(
        self,
        item: PauseSampleWorkItem,
        ctx: AdapterContext,
        *,
        llm_client: Any,
    ) -> PauseEditGroup:
        cfg = ctx.pipeline_config
        plain_text = _normalize_plain_text(item.segment.text)
        duration_pool = sample_duration_pool(DURATION_POOL_SIZE)
        num_profiles = int(cfg['num_profiles'])
        profile_targets = sample_pause_profile_targets(
            plain_text=plain_text,
            num_profiles=num_profiles,
            num_operation_weights=cfg['num_operation_weights'],
            boundary_type_weights=cfg['boundary_type_weights'],
            pause_level_weights=cfg['pause_level_weights'],
            seed=item.attempt_index,
        )
        max_operations = max(target.num_operations for target in profile_targets)
        max_retries = max(1, int(cfg['llm_max_retries']))

        profiles, normed_text = generate_profiles_with_retries(
            plain_text=plain_text,
            duration_pool=duration_pool,
            profile_targets=profile_targets,
            llm_client=llm_client,
            llm_backend=cfg.get("llm_backend"),
            llm_model=cfg.get("llm_model"),
            max_retries=max_retries,
        )

        segment_metadata = {
            key: value
            for key, value in dict(item.segment.metadata or {}).items()
            if key not in {"audio", "audios", "speaker_prompt"}
        }
        group_metadata = {
            "input_id": item.sample_id,
            "original_plain_text": plain_text,
            "gen_duration_sec": round(float(item.segment.duration), 2),
            "num_profiles_requested": num_profiles,
            "num_profiles_valid": len(profiles),
            "profile_targets": [
                target.to_prompt_dict() for target in profile_targets
            ],
            "duration_pool": duration_pool,
            "segment_clip_id": item.segment.clip_id,
            "clip_id": item.segment.clip_id,
            "sample_rate": int(item.segment.sample_rate),
            "language": cfg['sample_language'] or classify_text_language(normed_text),
            "source_dataset": segment_metadata.get("from_dataset")
            or segment_metadata.get("source_dataset"),
            "segment_metadata": segment_metadata,
        }
        return PauseEditGroup(
            normed_text=normed_text,
            profiles=profiles,
            duration_pool=duration_pool,
            num_operations=max_operations,
            original_audio=item.segment.audio_bytes,
            metadata=group_metadata,
        )

    def process(self, stage_name: str, payload: Any, ctx: AdapterContext) -> StageResult:
        from speech_edit.clients.services import (
            AlignmentClient,
            ASRClient,
            BoundaryPauseSmootherClient,
            LLMClient,
            SpkSimClient,
            UTMOSClient,
            WERClient,
            WDTWClient,
        )
        from speech_edit.pipelines.pause.evaluator import evaluate_pause_edit
        from speech_edit.pipelines.pause.synthesizer import synthesize_pause_edit

        cfg = ctx.pipeline_config
        services = ctx.services

        if stage_name == "sample":
            outputs: List[OutputItem] = []
            base_seed = int(cfg['seed'])
            sample_id_prefix = str(cfg['sample_id_prefix'])
            for idx, segment in enumerate(_segments_from_parquet_payload(payload, cfg), start=1):
                key = _segment_key(segment, f"pause-{idx}")
                outputs.append(
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=key,
                        payload=PauseSampleWorkItem(
                            sample_id=(
                                f"{sample_id_prefix}"
                                f"sample_{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:12]}"
                            ),
                            segment=segment,
                            attempt_index=_stable_seed_index(base_seed, key),
                        ),
                    )
                )
            return StageResult(outputs=outputs)

        if stage_name == "plan":
            item: PauseSampleWorkItem = payload
            group = self._generate_group(
                item,
                ctx,
                llm_client=LLMClient(services.url("llm")),
            )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=f"{item.sample_id}:profile_{profile.profile_id}",
                        payload=PauseProfileWorkItem(
                            sample_id=item.sample_id,
                            normed_text=group.normed_text,
                            profile=profile,
                            duration_pool=list(group.duration_pool),
                            num_operations=group.num_operations,
                            original_audio=group.original_audio,
                            metadata=dict(group.metadata or {}),
                        ),
                    )
                    for profile in group.profiles
                ]
            )

        if stage_name == "synth":
            item: PauseProfileWorkItem = payload
            center_regen_margin_sec = cfg['center_regen_margin_sec']
            synth_pair = synthesize_pause_edit(
                profile=item.profile,
                group=item.to_group(),
                smoother_client=BoundaryPauseSmootherClient(
                    services.url("boundary_pause_smoother")
                ),
                alignment_client=AlignmentClient(services.url("aligner")),
                expand_radius=float(cfg['expand_radius']),
                aligner_backend=str(cfg['aligner_backend']),
                synth_method=str(cfg['synth_method']),
                center_regen_second_pass=bool(cfg['center_regen_second_pass']),
                center_regen_margin_sec=(
                    None
                    if center_regen_margin_sec in (None, "")
                    else float(center_regen_margin_sec)
                ),
            )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=f"{item.sample_id}:profile_{item.profile.profile_id}",
                        payload=PauseSynthesizedWorkItem(
                            sample_id=item.sample_id,
                            work_item=item,
                            synth_pair=synth_pair,
                        ),
                    )
                ]
            )

        if stage_name == "eval":
            item: PauseSynthesizedWorkItem = payload
            eval_result = None
            if bool(cfg["evaluate"]):
                eval_result = evaluate_pause_edit(
                    synth_pair=item.synth_pair,
                    asr_client=ASRClient(services.url("asr")),
                    alignment_client=AlignmentClient(services.url("aligner")),
                    wer_client=WERClient(services.url("wer")),
                    aligner_backend=str(cfg['aligner_backend']),
                    utmos_client=(
                        None
                        if bool(cfg['skip_utmos_eval'])
                        else UTMOSClient(services.url("utmos"))
                    ),
                    spk_sim_client=(
                        None
                        if bool(cfg['skip_spk_sim_eval'])
                        else SpkSimClient(services.url("spk_sim"))
                    ),
                    wdtw_client=(
                        None
                        if bool(cfg['skip_wdtw_eval'])
                        else WDTWClient(services.url("wdtw"))
                    ),
                    sample_language=cfg['sample_language'],
                )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=f"{item.sample_id}:profile_{item.work_item.profile.profile_id}",
                        payload=PauseEvaluatedWorkItem(
                            sample_id=item.sample_id,
                            work_item=item.work_item,
                            synth_pair=item.synth_pair,
                            eval_result=eval_result,
                        ),
                    )
                ]
            )

        if stage_name == "save":
            rows = PauseEditParquetRowProcessor().rows_from_payload(
                payload,
                dataset_name=ctx.stage.parquet_output.dataset_name,
            )
            return StageResult(parquet_rows=rows)

        raise ValueError(f"unknown stage for {self.name}: {stage_name}")
