"""Stage-runtime adapter for the text edit pipeline."""

from __future__ import annotations

import random
import uuid
import hashlib
from dataclasses import fields
from typing import Any, List, Mapping, Optional, Sequence

from speech_edit.clients.services import (
    AlignmentClient, ASRClient, F5TTSClient, LLMClient,
    SpkSimClient, UTMOSClient, WDTWClient, WERClient,
)
from speech_edit.runtime.adapters import PipelineAdapter
from speech_edit.runtime.models import AdapterContext, OutputItem, StageResult
from speech_edit.data.parquet.reader import Segment, read_segments_from_parquet_payload

from .parquet_rows import TextEditParquetRowProcessor


def _segments_from_parquet_payload(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> List[Segment]:
    return read_segments_from_parquet_payload(payload, cfg)


def _segment_key(segment: Segment, fallback: str) -> str:
    meta = dict(segment.metadata or {})
    source_path = meta.get("source_parquet_path", "")
    row_idx = meta.get("source_row_idx", meta.get("source_row_index", meta.get("row_idx", "")))
    if source_path or row_idx != "":
        return f"{source_path}:{row_idx}:{segment.clip_id}"
    return f"{fallback}:{segment.clip_id}"


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


class TextEditStageAdapter(PipelineAdapter):
    name = "text_edit"


    def _config(self, ctx: AdapterContext, *, evaluate: Optional[bool] = None) -> Any:
        from speech_edit.pipelines.text.pipeline import PipelineConfig

        cfg = ctx.pipeline_config
        values = {field.name: cfg[field.name] for field in fields(PipelineConfig) if field.name != "output_dir"}
        values["output_dir"] = str(ctx.run_config["run_dir"])
        values["evaluate"] = bool(cfg['evaluate'] if evaluate is None else evaluate)
        return PipelineConfig(**values)

    def process(self, stage_name: str, payload: Any, ctx: AdapterContext) -> StageResult:
        from speech_edit.pipelines.text.generation_plan import (

            select_generation_specs,
        )
        from speech_edit.pipelines.text.pipeline import (
            PlannedWorkItem,
            SampleWorkItem,
            SynthesizedWorkItem,
            _evaluate_synthesized_result,
            execute_tagged_plan,
        )
        cfg = ctx.pipeline_config
        if stage_name == "sample":
            outputs: List[OutputItem] = []
            base_seed = int(cfg['seed'])
            for idx, segment in enumerate(_segments_from_parquet_payload(payload, cfg)):
                item_key = _segment_key(segment, f"text-{idx}")
                rng = random.Random(_stable_seed(base_seed, item_key))
                specs = tuple(
                    select_generation_specs(
                        num_generations=int(cfg['num_generations']),
                        span_level_weights=cfg['span_level_weights'],
                        op_count_weights=cfg['op_count_weights'],
                        require_del_ins_prob=float(
                            cfg['require_del_ins_prob']
                        ),
                        rng=rng,
                    )
                )
                sample_id = f"sample_{uuid.uuid5(uuid.NAMESPACE_URL, item_key).hex[:12]}"
                outputs.append(
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=item_key,
                        payload=SampleWorkItem(
                            sample_id=sample_id,
                            segment=segment,
                            attempt_index=idx,
                            specs=specs,
                        ),
                    )
                )
            return StageResult(outputs=outputs)
        if stage_name == "plan":
            plan_config = self._config(ctx)
            llm = LLMClient(ctx.services.url("llm"))
            max_retries = max(1, int(getattr(plan_config, "llm_max_retries", 3)))
            validation_errors: List[str] = []
            for attempt in range(1, max_retries + 1):
                try:
                    return self.process_plan_attempt(
                        payload,
                        ctx,
                        llm_client=llm,
                        config=plan_config,
                        attempt=attempt,
                        validation_errors=tuple(validation_errors),
                    )
                except Exception as exc:
                    validation_errors.extend(_validation_errors_from_exception(exc))
                    if attempt >= max_retries:
                        raise
        if stage_name == "synth":
            item: PlannedWorkItem = payload
            result = execute_tagged_plan(
                segment=item.segment,
                sample_id=item.sample_id,
                plan=item.plan,
                config=self._config(ctx, evaluate=False),
                aligner=AlignmentClient(ctx.services.url("aligner")),
                f5=F5TTSClient(ctx.services.url("f5")),
                persist=False,
                include_artifacts=True,
            )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=item.sample_id,
                        payload=SynthesizedWorkItem(
                            sample_id=item.sample_id,
                            segment=item.segment,
                            attempt_index=item.attempt_index,
                            result=result,
                        ),
                    )
                ]
            )
        if stage_name == "eval":
            item: SynthesizedWorkItem = payload
            evaluated = _evaluate_synthesized_result(
                item,
                config=self._config(ctx, evaluate=True),
                asr=ASRClient(ctx.services.url("asr")),
                wer=WERClient(ctx.services.url("wer")),
                utmos=UTMOSClient(ctx.services.url("utmos")),
                spk_sim=SpkSimClient(ctx.services.url("spk_sim")),
                wdtw=WDTWClient(ctx.services.url("wdtw")),
            )
            return StageResult(
                outputs=[
                    OutputItem(
                        queue_name=_required_output_queue(ctx),
                        item_key=evaluated.sample_id,
                        payload=evaluated,
                    )
                ]
            )
        if stage_name == "save":
            rows = TextEditParquetRowProcessor().rows_from_payload(
                payload,
                dataset_name=ctx.stage.parquet_output.dataset_name,
            )
            return StageResult(parquet_rows=rows)
        raise ValueError(f"unknown stage for {self.name}: {stage_name}")

    def process_plan_attempt(
        self,
        payload: Any,
        ctx: AdapterContext,
        *,
        llm_client: Any,
        config: Any,
        attempt: int,
        validation_errors: Sequence[str] = (),
    ) -> StageResult:
        from speech_edit.pipelines.text.pipeline import (
            PlannedWorkItem,
            SampleWorkItem,
            generate_tagged_edit_plan_with_specs_attempt,
        )

        item: SampleWorkItem = payload
        plan = generate_tagged_edit_plan_with_specs_attempt(
            item.segment.text,
            specs=item.specs,
            llm_client=llm_client,
            config=config,
            attempt=attempt,
            validation_errors=validation_errors,
        )
        return StageResult(
            outputs=[
                OutputItem(
                    queue_name=_required_output_queue(ctx),
                    item_key=item.sample_id,
                    payload=PlannedWorkItem(
                        sample_id=item.sample_id,
                        segment=item.segment,
                        attempt_index=item.attempt_index,
                        plan=plan,
                    ),
                )
            ]
        )
