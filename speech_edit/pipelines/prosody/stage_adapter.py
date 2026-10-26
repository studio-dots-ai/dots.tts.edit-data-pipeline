"""Distributed pipeline adapter for the prosody edit pipeline."""

from __future__ import annotations

import time
import uuid
import hashlib
from typing import Any, List, Mapping


from speech_edit.runtime.adapters import PipelineAdapter
from speech_edit.runtime.models import AdapterContext, OutputItem, StageResult
from speech_edit.data.parquet.reader import Segment, read_segments_from_parquet_payload

from .parquet_rows import ProsodyParquetRowProcessor


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
    return int(digest[:8], 16)


class ProsodyStageAdapter(PipelineAdapter):
    name = "prosody_edit"


    def process(self, stage_name: str, payload: Any, ctx: AdapterContext) -> StageResult:
        from speech_edit.clients.services import (
            AlignmentClient,
            ASRClient,
            F5TTSClient,
            F0EvalClient,
            LLMClient,
            PitchConversionClient,
            SpeechRateClient,
            SpkSimClient,
            UTMOSClient,
            WERClient,
            WDTWClient,
        )
        from speech_edit.pipelines.prosody.input_generator import (
            Stage0Sample,
            generate_input_from_sampled_data,
        )

        cfg = ctx.pipeline_config
        services = ctx.services
        if stage_name == "sample":
            outputs: List[OutputItem] = []
            base_seed = int(cfg['seed'])
            for idx, segment in enumerate(_segments_from_parquet_payload(payload, cfg), start=1):
                key = _segment_key(segment, f"prosody-{idx}")
                sample = Stage0Sample(
                    clip_id=segment.clip_id,
                    transcript=segment.text,
                    audio_bytes=segment.audio_bytes,
                    sample_rate=segment.sample_rate,
                    duration=segment.duration,
                    metadata=segment.metadata,
                )
                outputs.append(
                    OutputItem(
                        queue_name=ctx.stage.output_queue or "prosody.sample_to_plan",
                        item_key=key,
                        payload={
                            "sample_id": f"sample_{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:12]}",
                            "attempt_index": _stable_seed_index(base_seed, key),
                            "sample": sample,
                        },
                    )
                )
            return StageResult(outputs=outputs)
        if stage_name == "plan":
            from speech_edit.pipelines.prosody.pipeline import make_input_seed

            llm_client = LLMClient(services.url("llm"))
            diagnostics: dict[str, Any] = {}
            pipeline_input = generate_input_from_sampled_data(
                payload["sample"],
                llm_client,
                edit_kind=cfg.get("edit_kind"),
                edit_kind_weights=cfg['edit_kind_weights'],
                span_level=cfg.get("span_level"),
                span_level_weights=cfg['span_level_weights'],
                num_waves=int(cfg['num_waves']),
                temperature=float(cfg["llm_temperature"]),
                max_retries=int(cfg["llm_max_retries"]),
                llm_backend=cfg.get("llm_backend"),
                llm_model=cfg.get("llm_model"),
                sample_language=cfg['sample_language'],
                parameter_policy=cfg['parameter_policy'],
                region_count_weights=cfg.get("region_count_weights"),
                forward_nl_mode=str(cfg['forward_nl_mode']),
                reverse_nl_mode=str(cfg['reverse_nl_mode']),
                stage1_seed=make_input_seed(int(cfg['seed']), int(payload["attempt_index"])),
                raise_on_llm_service_error=True,
                diagnostics=diagnostics,
            )
            if pipeline_input is None:
                import json

                raise RuntimeError(
                    "Prosody plan stage failed to generate pipeline input; "
                    f"diagnostics={json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)}"
                )
            pipeline_input.metadata.setdefault("input_id", payload["sample_id"])
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=ctx.stage.output_queue or "prosody.plan_to_synth",
                        item_key=str(payload["sample_id"]),
                        payload=pipeline_input,
                    )
                ]
            )
        if stage_name == "synth":
            pipeline_input = payload
            from speech_edit.pipelines.prosody.pipeline import edit_sample

            state = edit_sample(
                pipeline_input,
                aligner_client=AlignmentClient(services.url("aligner")),
                aligner_backend=str(cfg['aligner_backend']),
                pitch_method=str(cfg['pitch_method']),
                rate_method=str(cfg['rate_method']),
                psola_indextts2_cco_pitch_boost=str(
                    cfg['psola_indextts2_cco_pitch_boost']
                ),
                pitch_client=PitchConversionClient(services.url("pitch")),
                rate_client=SpeechRateClient(services.url("rate")),
                f5_client=F5TTSClient(services.url("f5")),
                crossfade_ms=float(cfg['crossfade_ms']),
                smooth_margin_sec=float(cfg['smooth_margin_sec']),
                smooth_policy=str(cfg['smooth_policy']),
                f5_steps=int(cfg["f5_steps"]),
                f5_seed=cfg["f5_seed"],
                f5_source_crossfade_sec=float(cfg["f5_source_crossfade_sec"]),
                f5_regen_crossfade_sec=float(cfg["f5_regen_crossfade_sec"]),
                t_start=time.time(),
            )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=ctx.stage.output_queue or "prosody.synth_to_eval",
                        item_key=str(state["sample_id"]),
                        payload=state,
                    )
                ]
            )
        if stage_name == "eval":
            from speech_edit.pipelines.prosody.pipeline import evaluate_sample

            state = evaluate_sample(
                payload,
                asr_client=ASRClient(services.url("asr")),
                wer_client=WERClient(services.url("wer")),
                utmos_client=UTMOSClient(services.url("utmos")),
                spk_sim_client=SpkSimClient(services.url("spk_sim")),
                wdtw_client=WDTWClient(services.url("wdtw"))
                if bool(cfg['evaluate'])
                else None,
                f0_client=F0EvalClient(services.url("f0"))
                if bool(cfg['evaluate'])
                else None,
                do_evaluate=bool(cfg['evaluate']),
            )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=ctx.stage.output_queue or "prosody.eval_to_save",
                        item_key=str(state["sample_id"]),
                        payload=state,
                    )
                ]
            )
        if stage_name == "save":
            rows = ProsodyParquetRowProcessor().rows_from_payload(
                payload,
                dataset_name=ctx.stage.parquet_output.dataset_name,
            )
            return StageResult(parquet_rows=rows)
        raise ValueError(f"unknown stage for {self.name}: {stage_name}")
