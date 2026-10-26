"""Tagged-text based text edit pipeline.

Stages
------
1. LLM emits structured tagged text instructions.
2. Parser derives source/target text and source audio edit spans.
3. F5 sample regenerates target-timeline edit spans from tagged text only.
4. Generated target audio is aligned again for reverse-pair spans.
5. Pair metadata is evaluated and persisted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from speech_edit.clients.services import (  # noqa: E402
    AlignmentClient,
    ASRClient,
    decode_tensor_response,
    F5TTSClient,
    LLMClient,
    SpkSimClient,
    UTMOSClient,
    WERClient,
    WDTWClient,
)
from . import PIPELINE_TYPE, PIPELINE_VERSION
from .audio_timeline import (
    build_target_condition_waveform,
    mono_float_to_wav_bytes,
    nonsilent_speech_duration,
    wav_bytes_to_mono_float,
)
from .canonical_text import (
    canonicalize_text,
    project_tagged_instruction_to_surface,
    validate_canonical_sentence_edit,
    validate_generation_op_count,
)
from .generation_plan import GenerationSpec
from .prompts import build_forward_generation_prompt, build_reverse_instruction_prompt
from .sample_output import (
    build_sample_metadata,
    build_segments_metadata,
    pair_dir_name,
    save_bytes,
    save_json,
    save_sample_outputs,
)
from .span_planner import (
    SpanPlan,
    alignment_answers_to_spans,
    build_alignment_queries_from_st,
    plan_spans_from_answers,
)
from speech_edit.data.text_markup import (
    EQUAL_TAG,
    ORIGINAL_WAVE_ID,
    edit_segments,
    op_counts,
    parse_tagged_instruction,
    render_segment,
    render_source,
    render_target,
    reverse_tagged_instruction,
    validate_tagged_instruction,
)

logger = logging.getLogger("text_edit_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WER_CALC_ROOT = PROJECT_ROOT / "tools" / "wer_calc"
if str(WER_CALC_ROOT) not in sys.path:
    sys.path.insert(0, str(WER_CALC_ROOT))
from wer_calc import compute_error_rate as _compute_error_rate  # noqa: E402
from wer_calc import compute_local_wer as _compute_local_wer  # noqa: E402
from speech_edit.data.language import (  # noqa: E402
    alignment_language_code,
    wer_language_code,
)

NL_SIMPLE_COUNT = 4
NL_HARD_COUNT = 4

if TYPE_CHECKING:  # pragma: no cover
    from speech_edit.data.parquet.reader import Segment


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: str
    num_generations: int
    seed: int
    duration_mode: str
    expand_radius: float
    merge_gap: float
    span_level_weights: Optional[Dict[str, float]]
    op_count_weights: Optional[Dict[int, float]]
    require_del_ins_prob: float
    min_edit_distance_pct: Optional[float]
    min_preserved_source_ratio: Optional[float]
    llm_temperature: float
    llm_max_tokens: int
    llm_max_retries: int
    forward_nl_mode: str
    reverse_nl_mode: str
    aligner_backend: str
    f5_steps: int
    f5_cfg_strength: float
    f5_sway_sampling_coef: float
    source_crossfade_sec: float
    regen_crossfade_sec: float
    prepend_context_enabled: bool
    fail_if_cannot_prepend_context: bool
    min_unmasked_context_sec: float
    min_unmasked_context_ratio: float
    max_f5_edit_seconds: float
    synthesis_mode: str
    debug: bool
    debug_output_dir: Optional[str]
    evaluate: bool
    output_dataset_name: str


@dataclass(frozen=True)
class TaggedGeneration:
    generation_id: str
    st: str
    canonical_st: str
    spec: GenerationSpec
    forward_nl: Dict[str, List[str]]


@dataclass(frozen=True)
class TaggedEditPlan:
    """Stage 1 output consumed by Stage 2+ execution.

    The plan is intentionally audio-free. Bulk runs produce it through the
    distributed plan stage; the service can also build it from an externally
    supplied tagged instruction.
    """

    generations: List[TaggedGeneration]
    reverse_nl_by_generation: Dict[str, Dict[str, List[str]]]
    forward_llm_stats: Dict[str, Any]
    reverse_llm_stats: Dict[str, Any]


@dataclass(frozen=True)
class QueryAlignmentResult:
    answers: List[Dict[str, Any]]
    alignments: Optional[List[Dict[str, Any]]] = None
    backend: Optional[str] = None


class GenerationSkipped(RuntimeError):
    """Raised when a generation is intentionally skipped for policy reasons."""


class SampleSkipped(RuntimeError):
    """Raised when the current source sample should be skipped."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        forward_attempts: int = 0,
        reverse_attempts: int = 0,
        validation_errors: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.forward_attempts = forward_attempts
        self.reverse_attempts = reverse_attempts
        self.validation_errors = list(validation_errors or [])


@dataclass(frozen=True)
class SampleWorkItem:
    sample_id: str
    segment: Any
    attempt_index: int
    specs: Tuple[GenerationSpec, ...]


@dataclass(frozen=True)
class PlannedWorkItem:
    sample_id: str
    segment: Any
    attempt_index: int
    plan: TaggedEditPlan


@dataclass(frozen=True)
class SynthesizedWorkItem:
    sample_id: str
    segment: Any
    attempt_index: int
    result: Dict[str, Any]


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _loads_jsonish(text: str) -> Any:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(1))


_TEXT_EDIT_TAG_RE = re.compile(r"</?(?:del|ins|sub|q)(?:\s+[^>]*)?>")
def _repair_common_tag_typos(st: str) -> str:
    """Repair narrow, common LLM typos without changing the edit semantics."""
    return st.replace("</a>", "</sub>")


def _classify_validation_error(message: str) -> str:
    if "canonical source render" in message or "Canonical source render mismatch" in message:
        return "canonical_source_mismatch"
    if "Generation has" in message:
        return "op_count_mismatch"
    if "Sentence generation" in message:
        return "sentence_contract"
    if "Malformed" in message or "Missing </" in message:
        return "malformed_tags"
    if "nl" in message or "instructions" in message or "JSON" in message:
        return "schema_or_nl"
    return "other"


def _merge_counters(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _validation_error_distribution(errors: Sequence[str]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for error in errors:
        counter[_classify_validation_error(error)] += 1
    return dict(counter)


def _combined_validation_errors(*items: Any) -> Dict[str, int]:
    combined: Dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            _merge_counters(combined, item)
    return combined


def _validate_instruction_group(
    value: Any,
    *,
    key: str,
    expected_count: int,
) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"nl.{key} must be a list")
    if len(value) != expected_count:
        raise ValueError(f"nl.{key} must contain {expected_count} instructions")
    instructions: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"nl.{key}[{index}] must be a non-empty string")
        if _TEXT_EDIT_TAG_RE.search(item):
            raise ValueError(f"nl.{key}[{index}] must not contain edit tags")
        instructions.append(item.strip())
    return instructions


def _validate_nl_payload(
    payload: Any,
    *,
    simple_count: int,
    hard_count: int,
) -> Dict[str, List[str]]:
    if not isinstance(payload, dict):
        raise ValueError("nl must be an object")
    simple_per_language = simple_count // 2
    hard_per_language = hard_count // 2
    if simple_count != 4 or hard_count != 4:
        raise ValueError("Only 4 simple and 4 hard instructions are supported")
    simple_en = _validate_instruction_group(
        payload.get("simple_en"),
        key="simple_en",
        expected_count=simple_per_language,
    )
    simple_zh = _validate_instruction_group(
        payload.get("simple_zh"),
        key="simple_zh",
        expected_count=simple_per_language,
    )
    hard_en = _validate_instruction_group(
        payload.get("hard_en"),
        key="hard_en",
        expected_count=hard_per_language,
    )
    hard_zh = _validate_instruction_group(
        payload.get("hard_zh"),
        key="hard_zh",
        expected_count=hard_per_language,
    )
    return {
        "simple": simple_en + simple_zh,
        "hard": hard_en + hard_zh,
    }


def _validate_canonical_generation(
    canonical_st: str,
    *,
    canonical_source_text: str,
    spec: GenerationSpec,
) -> None:
    if canonicalize_text(render_source(canonical_st)).text != canonical_source_text:
        raise ValueError("canonical source render does not match canonical source text")
    validate_generation_op_count(canonical_st, expected_count=spec.op_count)
    counts = op_counts(canonical_st)
    if spec.require_del_ins and counts.get("del", 0) + counts.get("ins", 0) < 1:
        raise ValueError(
            f"generation {spec.generation_id} must contain at least one del or ins edit"
        )
    if spec.span_level == "sentence":
        validate_canonical_sentence_edit(canonical_st, canonical_source_text)


def _ratio(errors: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return errors / denominator


def _target_edit_distance_stats(
    source_text: str,
    target_text: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    metric_language = wer_language_code(
        language,
        reference_text=source_text,
        hypothesis_text=target_text,
        prefer_auto=True,
    )
    result = _compute_error_rate(source_text, target_text, language=metric_language)
    errors = int(result.substitutions) + int(result.deletions) + int(result.insertions)
    source_ratio = _ratio(errors, int(result.ref_length))
    target_ratio = _ratio(errors, int(result.hyp_length))
    return {
        "errors": errors,
        "source_ratio": source_ratio,
        "target_ratio": target_ratio,
        "max_ratio": max(source_ratio, target_ratio),
        "source_pct": 100.0 * source_ratio,
        "target_pct": 100.0 * target_ratio,
        "max_pct": 100.0 * max(source_ratio, target_ratio),
        "source_length": int(result.ref_length),
        "target_length": int(result.hyp_length),
        "metric": str(result.metric),
    }


def _visible_nonspace_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _preserved_source_ratio(st: str, source_text: str) -> float:
    total = _visible_nonspace_len(source_text)
    if total <= 0:
        return 0.0
    preserved = sum(
        _visible_nonspace_len(segment.source)
        for segment in parse_tagged_instruction(st)
        if segment.kind == EQUAL_TAG
    )
    return preserved / total


def _validate_target_edit_distance(
    source_text: str,
    target_text: str,
    *,
    min_ratio: Optional[float],
) -> None:
    if min_ratio is None or min_ratio <= 0:
        return
    stats = _target_edit_distance_stats(source_text, target_text)
    if stats["max_ratio"] < min_ratio:
        raise ValueError(
            "target edit distance is "
            f"{stats['source_pct']:.1f}% of source / {stats['target_pct']:.1f}% of target, "
            f"below required {min_ratio * 100.0:.1f}% on either side"
        )


def _validate_preserved_source_ratio(
    st: str,
    source_text: str,
    *,
    spec: GenerationSpec,
    min_ratio: Optional[float],
) -> None:
    if min_ratio is None or min_ratio <= 0 or spec.span_level == "sentence":
        return
    ratio = _preserved_source_ratio(st, source_text)
    if ratio < min_ratio:
        raise ValueError(
            "preserved source ratio is "
            f"{ratio * 100.0:.1f}%, below required {min_ratio * 100.0:.1f}% "
            "for non-sentence span levels"
        )


def _project_and_validate_generation(
    canonical_st: str,
    *,
    source_text: str,
    spec: GenerationSpec,
    min_edit_distance_pct: Optional[float] = None,
    min_preserved_source_ratio: Optional[float] = None,
) -> str:
    canonical_st = _repair_common_tag_typos(canonical_st)
    canonical_source_text = canonicalize_text(source_text).text
    _validate_canonical_generation(
        canonical_st,
        canonical_source_text=canonical_source_text,
        spec=spec,
    )
    st = project_tagged_instruction_to_surface(canonical_st, source_text)
    target_text = render_target(st)
    validate_tagged_instruction(st, source_text, target_text)
    _validate_target_edit_distance(
        source_text,
        target_text,
        min_ratio=min_edit_distance_pct,
    )
    _validate_preserved_source_ratio(
        st,
        source_text,
        spec=spec,
        min_ratio=min_preserved_source_ratio,
    )
    return st


def _call_llm_json(prompt: str, *, llm_client: LLMClient, config: PipelineConfig) -> str:
    return llm_client.generate(
        prompt,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        json_mode=True,
    )


def _prompt_with_validation_feedback(prompt: str, errors: Sequence[str]) -> str:
    if not errors:
        return prompt
    trimmed = []
    for error in errors[-12:]:
        one_line = " ".join(str(error).split())
        trimmed.append(one_line[:500])
    bullet_list = "\n".join(f"- {error}" for error in trimmed)
    return f"""{prompt}

## Retry Feedback
The previous attempt failed validation. Regenerate the entire JSON object from scratch and fix every issue below.
Do not copy invalid tags or invalid language-slot text from the failed attempt.

Validation errors:
{bullet_list}
""".strip()


def _extract_forward_generation_payloads(raw_text: str) -> Dict[str, Dict[str, Any]]:
    data = _loads_jsonish(raw_text)
    if not isinstance(data, dict) or not isinstance(data.get("generations"), dict):
        raise ValueError("Forward LLM output must contain a generations object")
    payloads: Dict[str, Dict[str, Any]] = {}
    for generation_id, payload in data["generations"].items():
        if not isinstance(payload, dict):
            raise ValueError(f"generations.{generation_id} must be an object")
        if not isinstance(payload.get("st"), str):
            raise ValueError(f"generations.{generation_id}.st must be a string")
        payloads[str(generation_id)] = payload
    return payloads


def _extract_reverse_nl_payloads(raw_text: str) -> Dict[str, Any]:
    data = _loads_jsonish(raw_text)
    if not isinstance(data, dict) or not isinstance(data.get("instructions"), dict):
        raise ValueError("Reverse LLM output must contain an instructions object")
    payloads: Dict[str, Any] = {}
    for generation_id, payload in data["instructions"].items():
        if not isinstance(payload, dict) or not isinstance(payload.get("nl"), dict):
            raise ValueError(f"instructions.{generation_id}.nl must be an object")
        payloads[str(generation_id)] = payload["nl"]
    return payloads


def generate_tagged_generations_attempt(
    source_text: str,
    *,
    specs: Sequence[GenerationSpec],
    llm_client: Optional[LLMClient],
    config: PipelineConfig,
    attempt: int = 1,
    validation_errors: Sequence[str] = (),
) -> Tuple[List[TaggedGeneration], Dict[str, Any]]:
    """Run one forward LLM attempt for a tagged generation request."""
    if llm_client is None:
        raise RuntimeError("LLM client is required to generate tagged candidates")
    canonical_source = canonicalize_text(source_text).text
    prompt = build_forward_generation_prompt(
        canonical_source,
        specs=specs,
        min_edit_distance_pct=config.min_edit_distance_pct,
    )
    attempt_errors: List[str] = []
    previous_errors = list(validation_errors or [])
    try:
        attempt_prompt = (
            _prompt_with_validation_feedback(prompt, previous_errors)
            if previous_errors
            else prompt
        )
        raw = _call_llm_json(attempt_prompt, llm_client=llm_client, config=config)
        raw_map = _extract_forward_generation_payloads(raw)
    except Exception as exc:
        attempt_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        logger.warning("Forward LLM attempt %s failed: %s", attempt, attempt_errors)
        raise SampleSkipped(
            "forward LLM attempt failed",
            stage="forward_llm",
            forward_attempts=attempt,
            validation_errors=attempt_errors,
        ) from exc

    valid: List[TaggedGeneration] = []
    for spec in specs:
        payload = raw_map.get(spec.generation_id)
        if payload is None:
            attempt_errors.append(f"{spec.generation_id}: missing generation payload")
            continue
        try:
            canonical_st = _repair_common_tag_typos(str(payload["st"]))
            st = _project_and_validate_generation(
                canonical_st,
                source_text=source_text,
                spec=spec,
                min_edit_distance_pct=config.min_edit_distance_pct,
                min_preserved_source_ratio=config.min_preserved_source_ratio,
            )
            if config.forward_nl_mode == "template":
                forward_nl = _template_forward_nl_for_st(st)
            elif config.forward_nl_mode == "llm":
                if not isinstance(payload.get("nl"), dict):
                    raise ValueError(f"generations.{spec.generation_id}.nl must be an object")
                forward_nl = _validate_nl_payload(
                    payload["nl"],
                    simple_count=NL_SIMPLE_COUNT,
                    hard_count=NL_HARD_COUNT,
                )
            else:
                raise ValueError("forward_nl_mode must be 'template' or 'llm'")
        except Exception as exc:
            attempt_errors.append(f"{spec.generation_id}: {type(exc).__name__}: {exc}")
            continue
        valid.append(
            TaggedGeneration(
                generation_id=spec.generation_id,
                st=st,
                canonical_st=canonical_st,
                spec=spec,
                forward_nl=forward_nl,
            )
        )
    if not attempt_errors and len(valid) == len(specs):
        valid.sort(key=lambda item: item.generation_id)
        return valid, {
            "forward_attempts": attempt,
            "forward_retry_attempts": max(0, attempt - 1),
            "forward_validation_errors": _validation_error_distribution(previous_errors),
            "forward_nl_source": config.forward_nl_mode,
        }

    logger.warning(
        "Forward LLM attempt %s invalid; errors=%s raw=%r",
        attempt,
        attempt_errors[:5],
        raw[:500],
    )
    raise SampleSkipped(
        "forward LLM attempt invalid",
        stage="forward_llm",
        forward_attempts=attempt,
        validation_errors=attempt_errors,
    )


def _template_forward_nl_for_st(st: str) -> Dict[str, List[str]]:
    validate_tagged_instruction(st, render_source(st), render_target(st))
    edits = edit_segments(st)
    clauses_en: List[str] = []
    clauses_zh: List[str] = []
    for segment in edits:
        source_piece = segment.source.strip()
        target_piece = segment.target.strip()
        if segment.kind == "sub":
            clauses_en.append(f"replace '{source_piece}' with '{target_piece}'")
            clauses_zh.append(f"把'{source_piece}'改成'{target_piece}'")
        elif segment.kind == "del":
            clauses_en.append(f"delete '{source_piece}'")
            clauses_zh.append(f"删除'{source_piece}'")
        elif segment.kind == "ins":
            clauses_en.append(f"insert '{target_piece}'")
            clauses_zh.append(f"插入'{target_piece}'")
    edit_summary_en = ", and ".join(clauses_en) if clauses_en else "apply the tagged edit"
    edit_summary_zh = "，并".join(clauses_zh) if clauses_zh else "执行标记编辑"
    return {
        "simple": [
            f"Edit the text: {edit_summary_en}.",
            f"Apply this text edit: {edit_summary_en}.",
            f"修改文本：{edit_summary_zh}。",
            f"按要求编辑：{edit_summary_zh}。",
        ],
        "hard": [
            f"Please apply the requested edit while leaving everything else unchanged: {edit_summary_en}.",
            f"Revise only the specified span according to this instruction: {edit_summary_en}.",
            f"请只修改指定内容，其他部分保持不变：{edit_summary_zh}。",
            f"根据指令完成局部编辑，不要改动无关内容：{edit_summary_zh}。",
        ],
    }


def generate_forward_instructions_template(
    generations: Sequence[TaggedGeneration],
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
    forward_nl = {
        generation.generation_id: _template_forward_nl_for_st(generation.st)
        for generation in generations
    }
    return forward_nl, {
        "forward_attempts": 0,
        "forward_retry_attempts": 0,
        "forward_validation_errors": {},
        "source": "template",
    }


def generate_reverse_instructions(
    generations: Sequence[TaggedGeneration],
    *,
    llm_client: Optional[LLMClient],
    config: PipelineConfig,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
    if llm_client is None:
        raise RuntimeError("LLM client is required to generate reverse instructions")
    items: List[Dict[str, Any]] = []
    for generation in generations:
        reverse_st = reverse_tagged_instruction(generation.st)
        source_text = render_source(reverse_st)
        target_text = render_target(reverse_st)
        validate_tagged_instruction(reverse_st, source_text, target_text)
        items.append(
            {
                "generation_id": generation.generation_id,
                "source_text": source_text,
                "target_text": target_text,
                "st": reverse_st,
                "tag_diff": [
                    {
                        "kind": segment.kind,
                        "source_text": segment.source.strip(),
                        "target_text": segment.target.strip(),
                    }
                    for segment in edit_segments(reverse_st)
                ],
            }
        )
    prompt = build_reverse_instruction_prompt(items)
    all_errors: List[str] = []
    max_retries = max(1, int(config.llm_max_retries))
    for attempt in range(1, max_retries + 1):
        attempt_errors: List[str] = []
        try:
            attempt_prompt = (
                _prompt_with_validation_feedback(prompt, all_errors)
                if attempt > 1
                else prompt
            )
            raw = _call_llm_json(attempt_prompt, llm_client=llm_client, config=config)
            raw_map = _extract_reverse_nl_payloads(raw)
        except Exception as exc:
            attempt_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            all_errors.extend(attempt_errors)
            logger.warning("Reverse LLM attempt %s/%s failed: %s", attempt, max_retries, attempt_errors)
            continue
        reverse_nl: Dict[str, Dict[str, List[str]]] = {}
        for generation in generations:
            payload = raw_map.get(generation.generation_id)
            if payload is None:
                attempt_errors.append(f"{generation.generation_id}: missing reverse nl")
                continue
            try:
                reverse_nl[generation.generation_id] = _validate_nl_payload(
                    payload,
                    simple_count=NL_SIMPLE_COUNT,
                    hard_count=NL_HARD_COUNT,
                )
            except Exception as exc:
                attempt_errors.append(f"{generation.generation_id}: {type(exc).__name__}: {exc}")
        if not attempt_errors and len(reverse_nl) == len(generations):
            return reverse_nl, {
                "reverse_attempts": attempt,
                "reverse_retry_attempts": attempt - 1,
                "reverse_validation_errors": _validation_error_distribution(all_errors),
            }
        all_errors.extend(attempt_errors)
        logger.warning(
            "Reverse LLM attempt %s/%s invalid; errors=%s raw=%r",
            attempt,
            max_retries,
            attempt_errors[:5],
            raw[:500],
        )

    raise SampleSkipped(
        "reverse LLM failed after retries",
        stage="reverse_llm",
        reverse_attempts=max_retries,
        validation_errors=all_errors,
    )


def _template_reverse_nl_for_generation(generation: TaggedGeneration) -> Dict[str, List[str]]:
    reverse_st = reverse_tagged_instruction(generation.st)
    validate_tagged_instruction(reverse_st, render_source(reverse_st), render_target(reverse_st))
    edits = edit_segments(reverse_st)
    clauses_en: List[str] = []
    clauses_zh: List[str] = []
    for segment in edits:
        source_piece = segment.source.strip()
        target_piece = segment.target.strip()
        if segment.kind == "sub":
            clauses_en.append(f"replace '{source_piece}' with '{target_piece}'")
            clauses_zh.append(f"把'{source_piece}'改成'{target_piece}'")
        elif segment.kind == "del":
            clauses_en.append(f"delete '{source_piece}'")
            clauses_zh.append(f"删除'{source_piece}'")
        elif segment.kind == "ins":
            clauses_en.append(f"insert '{target_piece}'")
            clauses_zh.append(f"插入'{target_piece}'")
    edit_summary_en = ", and ".join(clauses_en) if clauses_en else "restore the original wording"
    edit_summary_zh = "，并".join(clauses_zh) if clauses_zh else "恢复原文"
    return {
        "simple": [
            f"Reverse the edit: {edit_summary_en}.",
            f"Change the generated text back by applying this reverse edit: {edit_summary_en}.",
            f"反向修改：{edit_summary_zh}。",
            f"把生成文本改回原文：{edit_summary_zh}。",
        ],
        "hard": [
            f"Please restore the original wording by applying the reverse operation: {edit_summary_en}.",
            f"Undo the previous edit while leaving everything else unchanged: {edit_summary_en}.",
            f"请执行反向编辑来恢复原来的表述：{edit_summary_zh}。",
            f"撤销之前的修改，其他内容保持不变：{edit_summary_zh}。",
        ],
    }


def generate_reverse_instructions_template(
    generations: Sequence[TaggedGeneration],
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
    reverse_nl = {
        generation.generation_id: _template_reverse_nl_for_generation(generation)
        for generation in generations
    }
    return reverse_nl, {
        "reverse_attempts": 0,
        "reverse_retry_attempts": 0,
        "reverse_validation_errors": {},
        "source": "template",
    }


def _empty_nl_payload() -> Dict[str, List[str]]:
    return {"simple": [], "hard": []}


def build_tagged_plan_from_st(
    *,
    source_text: str,
    st: str,
    generation_id: str = "000",
    forward_nl: Optional[Dict[str, List[str]]] = None,
    reverse_nl: Optional[Dict[str, List[str]]] = None,
    span_level: str = "arbitrary",
    reverse_nl_mode: str = "template",
) -> TaggedEditPlan:
    """Build a Stage 2+ plan from an externally supplied tagged instruction."""
    target_text = render_target(st)
    validate_tagged_instruction(st, source_text, target_text)
    op_count = sum(op_counts(st).values())
    if op_count < 1:
        raise ValueError("Tagged instruction must contain at least one edit")
    generation = TaggedGeneration(
        generation_id=str(generation_id),
        st=st,
        canonical_st=st,
        spec=GenerationSpec(
            generation_id=str(generation_id),
            span_level=span_level,
            op_count=op_count,
        ),
        forward_nl=forward_nl or _empty_nl_payload(),
    )
    if reverse_nl is not None:
        reverse_nl_by_generation = {str(generation_id): reverse_nl}
        reverse_llm_stats = {
            "reverse_attempts": 0,
            "reverse_retry_attempts": 0,
            "reverse_validation_errors": {},
            "source": "external_st",
        }
    elif reverse_nl_mode == "template":
        reverse_nl_by_generation, reverse_llm_stats = generate_reverse_instructions_template(
            [generation]
        )
    elif reverse_nl_mode == "llm":
        reverse_nl_by_generation = {str(generation_id): _empty_nl_payload()}
        reverse_llm_stats = {
            "reverse_attempts": 0,
            "reverse_retry_attempts": 0,
            "reverse_validation_errors": {},
            "source": "external_st",
        }
    else:
        raise ValueError("reverse_nl_mode must be 'template' or 'llm'")
    return TaggedEditPlan(
        generations=[generation],
        reverse_nl_by_generation=reverse_nl_by_generation,
        forward_llm_stats={
            "forward_attempts": 0,
            "forward_retry_attempts": 0,
            "forward_validation_errors": {},
            "source": "external_st",
        },
        reverse_llm_stats=reverse_llm_stats,
    )


def generate_tagged_edit_plan_with_specs_attempt(
    source_text: str,
    *,
    specs: Sequence[GenerationSpec],
    llm_client: Optional[LLMClient],
    config: PipelineConfig,
    attempt: int = 1,
    validation_errors: Sequence[str] = (),
) -> TaggedEditPlan:
    """Run one forward LLM attempt and build a plan when it validates."""
    generations, forward_llm_stats = generate_tagged_generations_attempt(
        source_text,
        specs=specs,
        llm_client=llm_client,
        config=config,
        attempt=attempt,
        validation_errors=validation_errors,
    )
    if config.reverse_nl_mode == "template":
        reverse_nl_by_generation, reverse_llm_stats = generate_reverse_instructions_template(
            generations
        )
        return TaggedEditPlan(
            generations=list(generations),
            reverse_nl_by_generation=reverse_nl_by_generation,
            forward_llm_stats=forward_llm_stats,
            reverse_llm_stats=reverse_llm_stats,
        )
    if config.reverse_nl_mode != "llm":
        raise ValueError("reverse_nl_mode must be 'template' or 'llm'")
    try:
        reverse_nl_by_generation, reverse_llm_stats = generate_reverse_instructions(
            generations,
            llm_client=llm_client,
            config=config,
        )
    except SampleSkipped as exc:
        raise SampleSkipped(
            str(exc),
            stage=exc.stage,
            forward_attempts=int(forward_llm_stats.get("forward_attempts", 0)),
            reverse_attempts=exc.reverse_attempts,
            validation_errors=exc.validation_errors,
        ) from exc
    return TaggedEditPlan(
        generations=list(generations),
        reverse_nl_by_generation=reverse_nl_by_generation,
        forward_llm_stats=forward_llm_stats,
        reverse_llm_stats=reverse_llm_stats,
    )


def _align_queries(
    aligner: AlignmentClient,
    *,
    audio_bytes: bytes,
    text: str,
    query_texts: Sequence[str],
    language: Optional[str],
    backend: str,
    word_level: bool = False,
) -> QueryAlignmentResult:
    if not query_texts:
        return QueryAlignmentResult(answers=[], backend=backend)
    align_language = _alignment_language(text, language)
    response = aligner.align(
        audio_bytes,
        text,
        language=align_language,
        backend=backend,
        word_level=word_level,
        queries=list(query_texts),
        return_response=True,
    )
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, list) or len(answers) != len(query_texts):
        raise RuntimeError(
            f"Aligner returned {len(answers) if isinstance(answers, list) else 0} "
            f"answers for {len(query_texts)} queries"
        )
    alignments: Optional[List[Dict[str, Any]]] = None
    if word_level:
        raw_alignments = response.get("alignments") if isinstance(response, dict) else None
        if not isinstance(raw_alignments, list):
            raise RuntimeError("Aligner did not return word-level alignments in query mode")
        alignments = [dict(item) for item in raw_alignments]
    response_backend = response.get("backend") if isinstance(response, dict) else None
    return QueryAlignmentResult(
        answers=answers,
        alignments=alignments,
        backend=str(response_backend) if response_backend is not None else backend,
    )


def _alignment_language(text: str, language: Optional[str]) -> str:
    return alignment_language_code(language, text=text)


def _duration_sec(audio_bytes: bytes) -> float:
    audio, sr = wav_bytes_to_mono_float(audio_bytes)
    return len(audio) / sr


def plan_tagged_edit(
    *,
    st: str,
    source_audio_bytes: bytes,
    source_duration: float,
    aligner: AlignmentClient,
    language: Optional[str],
    config: PipelineConfig,
    generated_durations: Optional[Sequence[Optional[float]]] = None,
    return_alignment_result: bool = False,
) -> SpanPlan | Tuple[SpanPlan, QueryAlignmentResult]:
    source_text = render_source(st)
    target_text = render_target(st)
    edits = edit_segments(st)
    queries = build_alignment_queries_from_st(st)
    alignment_result = _align_queries(
        aligner,
        audio_bytes=source_audio_bytes,
        text=source_text,
        query_texts=[query.query for query in queries],
        language=language,
        backend=config.aligner_backend,
        word_level=return_alignment_result,
    )
    plan = plan_spans_from_answers(
        source_text=source_text,
        target_text=target_text,
        edit_segments=edits,
        queries=queries,
        answers=alignment_result.answers,
        source_duration=source_duration,
        speech_duration=nonsilent_speech_duration(source_audio_bytes),
        duration_mode="generation" if generated_durations is not None else config.duration_mode,
        generated_durations=generated_durations,
        expand_radius=config.expand_radius,
        merge_gap=config.merge_gap,
    )
    if return_alignment_result:
        return plan, alignment_result
    return plan


def _sum_spans(spans: Sequence[Sequence[float]]) -> float:
    return sum(max(0.0, float(end) - float(start)) for start, end in spans)


def _conditioning_text_for_prepend(source_text: str, target_text: str) -> str:
    if not source_text or not target_text:
        return source_text + target_text
    left = source_text[-1]
    right = target_text[0]
    if left.isspace() or right.isspace():
        return source_text + target_text
    if re.match(r"[A-Za-z0-9]", left) and re.match(r"[A-Za-z0-9]", right):
        return f"{source_text} {target_text}"
    return source_text + target_text


def _json_sha256(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _debug_generation_dir(
    config: PipelineConfig,
    *,
    sample_id: str,
    generation_id: str,
) -> Optional[str]:
    if not config.debug:
        return None
    root = config.debug_output_dir or os.path.join(config.output_dir, "debug_artifacts")
    return os.path.join(root, sample_id, generation_id)


def _debug_step_dir(debug_dir: Optional[str], step_index: int) -> Optional[str]:
    if not debug_dir:
        return None
    return os.path.join(debug_dir, "steps", f"step_{step_index:03d}")


def _debug_step_artifacts(step_index: int) -> Dict[str, str]:
    prefix = f"steps/step_{step_index:03d}"
    return {
        "source_audio": f"{prefix}/source.wav",
        "condition_audio": f"{prefix}/condition.wav",
        "output_audio": f"{prefix}/output.wav",
        "metadata": f"{prefix}/meta.json",
    }


def _write_debug_manifest(
    *,
    debug_dir: Optional[str],
    manifest: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not debug_dir:
        return None
    os.makedirs(debug_dir, exist_ok=True)
    save_json(os.path.join(debug_dir, "debug.json"), manifest)
    return manifest


def _prepend_policy(
    *,
    span_plan: SpanPlan,
    source_duration: float,
    span_level: str,
    config: PipelineConfig,
) -> Dict[str, Any]:
    target_duration = float(span_plan.target_duration)
    edited_duration = _sum_spans(span_plan.target_spans)
    unmasked_duration = max(0.0, target_duration - edited_duration)
    unmasked_ratio = unmasked_duration / target_duration if target_duration > 0 else 0.0
    effective_fail = (
        config.fail_if_cannot_prepend_context
        or span_level == "sentence"
        or unmasked_duration < config.min_unmasked_context_sec
        or unmasked_ratio <= config.min_unmasked_context_ratio
    )
    metadata = {
        "prepend_context_enabled": config.prepend_context_enabled,
        "fail_if_cannot_prepend_context": config.fail_if_cannot_prepend_context,
        "effective_fail_if_cannot_prepend_context": effective_fail,
        "unmasked_context_sec": round(unmasked_duration, 5),
        "unmasked_context_ratio": round(unmasked_ratio, 5),
        "source_context_duration": round(source_duration, 5),
        "target_timeline_duration": round(target_duration, 5),
        "max_f5_edit_seconds": config.max_f5_edit_seconds,
    }
    return metadata


def synthesize_target_audio(
    *,
    st: str,
    source_audio_bytes: bytes,
    span_plan: SpanPlan,
    f5: F5TTSClient,
    config: PipelineConfig,
    trace_id: str,
    span_level: str = "word",
    debug_step_dir: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    source_waveform, sample_rate = wav_bytes_to_mono_float(source_audio_bytes)
    source_duration = len(source_waveform) / sample_rate
    target_condition = build_target_condition_waveform(
        source_waveform,
        sample_rate,
        span_plan.source_spans,
        span_plan.target_spans,
        span_plan.target_duration,
    )
    target_duration = len(target_condition) / sample_rate
    policy = _prepend_policy(
        span_plan=span_plan,
        source_duration=source_duration,
        span_level=span_level,
        config=config,
    )
    if target_duration > config.max_f5_edit_seconds:
        raise GenerationSkipped(
            f"target timeline exceeds F5 edit limit: {target_duration:.2f}s"
        )

    prepend_used = False
    disabled_reason = ""
    cond_waveform = target_condition
    edit_spans = span_plan.target_spans
    text_prefix = ""
    if config.prepend_context_enabled:
        total_duration = source_duration + target_duration
        if total_duration <= config.max_f5_edit_seconds:
            prepend_used = True
            cond_waveform = np.concatenate([source_waveform, target_condition])
            offset = source_duration
            edit_spans = [
                [round(float(start) + offset, 5), round(float(end) + offset, 5)]
                for start, end in span_plan.target_spans
            ]
            text_prefix = render_source(st)
        elif policy["effective_fail_if_cannot_prepend_context"]:
            raise GenerationSkipped(
                "prepend context is required but would exceed F5 edit limit: "
                f"{total_duration:.2f}s"
            )
        else:
            disabled_reason = "duration_limit"
    elif policy["effective_fail_if_cannot_prepend_context"]:
        raise GenerationSkipped("prepend context is required but disabled by config")
    else:
        disabled_reason = "disabled_by_config"

    target_text = render_target(st)
    text = (
        _conditioning_text_for_prepend(text_prefix, target_text)
        if prepend_used
        else target_text
    )
    f5_trace = {
        "conditioning_waveform_sha256": hashlib.sha256(
            np.asarray(cond_waveform, dtype=np.float32).tobytes()
        ).hexdigest(),
        "text": [text],
        "edit_spans_sec": edit_spans,
        "preserve_unmasked_waveform": True,
        "preserve_spans_sec": edit_spans,
        "input_sample_rate": sample_rate,
        "source_crossfade_sec": config.source_crossfade_sec,
        "regen_crossfade_sec": config.regen_crossfade_sec,
        "steps": config.f5_steps,
        "cfg_strength": config.f5_cfg_strength,
        "sway_sampling_coef": config.f5_sway_sampling_coef,
        "seed": config.seed,
        "output_format": "both",
    }
    f5_result = f5.sample(
        cond_waveform,
        [text],
        edit_spans_sec=edit_spans,
        preserve_unmasked_waveform=True,
        preserve_spans_sec=edit_spans,
        input_sample_rate=sample_rate,
        source_crossfade_sec=config.source_crossfade_sec,
        regen_crossfade_sec=config.regen_crossfade_sec,
        steps=config.f5_steps,
        cfg_strength=config.f5_cfg_strength,
        sway_sampling_coef=config.f5_sway_sampling_coef,
        seed=config.seed,
        output_format="both",
        trace_id=trace_id,
    )
    f5_output_format = "both"
    if isinstance(f5_result, dict) and isinstance(f5_result.get("waveform_tensor"), dict):
        generated_waveform = decode_tensor_response(f5_result["waveform_tensor"]).squeeze()
        generated_sr = int(f5_result.get("sample_rate") or sample_rate)
        if prepend_used:
            start = int(round(source_duration * generated_sr))
            end = start + int(round(target_duration * generated_sr))
            generated_waveform = generated_waveform[start:end]
        wav_bytes = mono_float_to_wav_bytes(generated_waveform, generated_sr)
    else:
        raise RuntimeError("F5 sample must return waveform_tensor for output_format='both'")
    generated_wav_sha256 = hashlib.sha256(wav_bytes).hexdigest()
    if prepend_used and f5_output_format == "both":
        generated_sr = int(f5_result.get("sample_rate") or sample_rate)
    elif not prepend_used:
        generated_waveform, generated_sr = wav_bytes_to_mono_float(wav_bytes)
    metadata = {
        "target_condition_duration": target_duration,
        "condition_duration": len(cond_waveform) / sample_rate,
        "sample_rate": sample_rate,
        "generated_sample_rate": generated_sr,
        "edit_spans_sec": edit_spans,
        "target_edit_spans_sec": span_plan.target_spans,
        "source_spans_sec": span_plan.source_spans,
        "planned_spans": [planned.to_metadata() for planned in span_plan.planned_spans],
        "fix_durations_sec": [planned.fix_duration for planned in span_plan.planned_spans],
        "text_payload": [text],
        "seed": config.seed,
        "cfg_strength": config.f5_cfg_strength,
        "sway_sampling_coef": config.f5_sway_sampling_coef,
        "source_crossfade_sec": config.source_crossfade_sec,
        "regen_crossfade_sec": config.regen_crossfade_sec,
        "f5_output_format": f5_output_format,
        "f5_service_metadata": f5_result.get("metadata") if isinstance(f5_result, dict) else None,
        "f5_payload_hash": _json_sha256(f5_trace),
        "generated_wav_sha256": generated_wav_sha256,
        "preserve_unmasked_waveform": True,
        "prepend_context_used": prepend_used,
        "prepend_context_disabled_reason": disabled_reason,
        **policy,
    }
    if debug_step_dir:
        os.makedirs(debug_step_dir, exist_ok=True)
        save_bytes(os.path.join(debug_step_dir, "source.wav"), source_audio_bytes)
        save_bytes(
            os.path.join(debug_step_dir, "condition.wav"),
            mono_float_to_wav_bytes(cond_waveform, sample_rate),
        )
        save_bytes(os.path.join(debug_step_dir, "output.wav"), wav_bytes)
        metadata["debug_artifacts"] = {
            "source_audio": "source.wav",
            "condition_audio": "condition.wav",
            "output_audio": "output.wav",
            "metadata": "meta.json",
        }
        save_json(os.path.join(debug_step_dir, "meta.json"), metadata)
    return wav_bytes, metadata


def align_generated_target_spans(
    *,
    st: str,
    generated_audio_bytes: bytes,
    generated_duration: float,
    aligner: AlignmentClient,
    language: Optional[str],
    config: PipelineConfig,
    return_alignment_result: bool = False,
) -> List[List[float]] | Tuple[List[List[float]], QueryAlignmentResult]:
    reverse_st = reverse_tagged_instruction(st)
    target_text = render_source(reverse_st)
    queries = build_alignment_queries_from_st(reverse_st)
    alignment_result = _align_queries(
        aligner,
        audio_bytes=generated_audio_bytes,
        text=target_text,
        query_texts=[query.query for query in queries],
        language=language,
        backend=config.aligner_backend,
        word_level=return_alignment_result,
    )
    spans = alignment_answers_to_spans(
        audio_duration=generated_duration,
        queries=queries,
        answers=alignment_result.answers,
    )
    if return_alignment_result:
        return spans, alignment_result
    return spans


def split_tagged_instruction_into_single_op_steps(st: str) -> List[str]:
    """Render ordered single-edit instructions for sequential per-op synthesis."""
    segments = parse_tagged_instruction(st)
    edit_indexes = [idx for idx, segment in enumerate(segments) if segment.is_edit]
    steps: List[str] = []
    for edit_index in edit_indexes:
        parts: List[str] = []
        for idx, segment in enumerate(segments):
            if not segment.is_edit:
                parts.append(render_segment(EQUAL_TAG, segment.source))
            elif idx < edit_index:
                parts.append(render_segment(EQUAL_TAG, segment.target))
            elif idx == edit_index:
                parts.append(render_segment(segment.kind, segment.source, segment.target))
            else:
                parts.append(render_segment(EQUAL_TAG, segment.source))
        step_st = "".join(parts)
        validate_tagged_instruction(step_st, render_source(step_st), render_target(step_st))
        steps.append(step_st)
    return steps


def synthesize_target_audio_per_op(
    *,
    st: str,
    source_audio_bytes: bytes,
    source_duration: float,
    aligner: AlignmentClient,
    f5: F5TTSClient,
    language: Optional[str],
    config: PipelineConfig,
    trace_id: str,
    span_level: str,
    debug_dir: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    steps_st = split_tagged_instruction_into_single_op_steps(st)
    if not steps_st:
        raise GenerationSkipped("per_op synthesis requires at least one edit")

    current_audio = source_audio_bytes
    current_text = render_source(st)
    current_duration = source_duration
    step_results: List[Dict[str, Any]] = []
    for step_index, step_st in enumerate(steps_st):
        step_source = render_source(step_st)
        if step_source != current_text:
            raise RuntimeError(
                "per_op step source mismatch: "
                f"expected={current_text!r} actual={step_source!r}"
            )
        step_config = replace(
            config,
            duration_mode="estimate",
            seed=config.seed + step_index,
        )
        step_plan = plan_tagged_edit(
            st=step_st,
            source_audio_bytes=current_audio,
            source_duration=current_duration,
            aligner=aligner,
            language=language,
            config=step_config,
        )
        step_debug_dir = _debug_step_dir(debug_dir, step_index)
        step_audio, step_synth = synthesize_target_audio(
            st=step_st,
            source_audio_bytes=current_audio,
            span_plan=step_plan,
            f5=f5,
            config=step_config,
            trace_id=f"{trace_id}-op{step_index:03d}",
            span_level=span_level,
            debug_step_dir=step_debug_dir,
        )
        step_target = render_target(step_st)
        step_results.append(
            {
                "step_index": step_index,
                "st": step_st,
                "source_text": step_source,
                "target_text": step_target,
                "op_counts": op_counts(step_st),
                "source_duration": current_duration,
                "target_duration": _duration_sec(step_audio),
                "span_plan": {
                    "source_spans": step_plan.source_spans,
                    "target_spans": step_plan.target_spans,
                    "metadata": step_plan.metadata,
                },
                "synthesis": step_synth,
                "artifacts": _debug_step_artifacts(step_index) if debug_dir else {},
            }
        )
        current_audio = step_audio
        current_text = step_target
        current_duration = step_results[-1]["target_duration"]

    disabled_reasons = [
        step["synthesis"].get("prepend_context_disabled_reason", "")
        for step in step_results
        if step["synthesis"].get("prepend_context_disabled_reason")
    ]
    metadata = {
        "synthesis_mode": "per_op",
        "effective_duration_mode": "estimate",
        "requested_duration_mode": config.duration_mode,
        "f5_call_count": len(step_results),
        "steps": step_results,
        "target_text": current_text,
        "duration": current_duration,
        "prepend_context_disabled_reason": disabled_reasons[0] if disabled_reasons else "",
        "prepend_context_disabled_steps": len(disabled_reasons),
        "preserve_unmasked_waveform": True,
    }
    return current_audio, metadata


def _span_durations(spans: Sequence[Sequence[float]]) -> List[float]:
    return [max(0.0, float(end) - float(start)) for start, end in spans]


def plan_tagged_edit_with_duration_mode(
    *,
    st: str,
    source_audio_bytes: bytes,
    source_duration: float,
    aligner: AlignmentClient,
    f5: F5TTSClient,
    language: Optional[str],
    config: PipelineConfig,
    trace_id: str,
    span_level: str,
    return_alignment_result: bool = False,
) -> Tuple[SpanPlan, Dict[str, Any], Optional[QueryAlignmentResult]]:
    """Plan target spans, using a synthesis probe when requested."""
    if config.duration_mode != "synthesis":
        plan_result = plan_tagged_edit(
            st=st,
            source_audio_bytes=source_audio_bytes,
            source_duration=source_duration,
            aligner=aligner,
            language=language,
            config=config,
            return_alignment_result=return_alignment_result,
        )
        if return_alignment_result:
            span_plan, alignment_result = plan_result
        else:
            span_plan = plan_result
            alignment_result = None
        return span_plan, {}, alignment_result

    estimate_config = replace(config, duration_mode="estimate")
    probe_plan = plan_tagged_edit(
        st=st,
        source_audio_bytes=source_audio_bytes,
        source_duration=source_duration,
        aligner=aligner,
        language=language,
        config=estimate_config,
    )
    probe_bytes, probe_synth = synthesize_target_audio(
        st=st,
        source_audio_bytes=source_audio_bytes,
        span_plan=probe_plan,
        f5=f5,
        config=estimate_config,
        trace_id=f"{trace_id}-duration-probe",
        span_level=span_level,
    )
    probe_duration = _duration_sec(probe_bytes)
    probe_target_spans = align_generated_target_spans(
        st=st,
        generated_audio_bytes=probe_bytes,
        generated_duration=probe_duration,
        aligner=aligner,
        language=language,
        config=config,
    )
    final_plan_result = plan_tagged_edit(
        st=st,
        source_audio_bytes=source_audio_bytes,
        source_duration=source_duration,
        aligner=aligner,
        language=language,
        config=config,
        generated_durations=_span_durations(probe_target_spans),
        return_alignment_result=return_alignment_result,
    )
    if return_alignment_result:
        final_plan, alignment_result = final_plan_result
    else:
        final_plan = final_plan_result
        alignment_result = None
    return (
        final_plan,
        {
            "duration_probe": {
                "duration": probe_duration,
                "target_aligned_spans": probe_target_spans,
                "target_span_durations": _span_durations(probe_target_spans),
                "synthesis": probe_synth,
            }
        },
        alignment_result,
    )


def compute_pair_wdtw_if_enabled(
    *,
    source_audio_bytes: bytes,
    target_audio_bytes: bytes,
    source_text: str,
    target_text: str,
    sample_language: Optional[str],
    config: PipelineConfig,
    wdtw: Optional[WDTWClient] = None,
    original_alignments: Optional[List[Dict[str, Any]]] = None,
    edited_alignments: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not config.evaluate or wdtw is None:
        return None
    compute_kwargs: Dict[str, Any] = {
        "source_audio": source_audio_bytes,
        "target_audio": target_audio_bytes,
        "source_transcript": source_text,
        "target_transcript": target_text,
        "source_alignments": original_alignments,
        "target_alignments": edited_alignments,
        "language": _alignment_language(target_text, sample_language),
        "mode": "preserved",
        "include_segments": False,
    }
    return wdtw.compute(
        **compute_kwargs,
    )


def evaluate_pair_if_enabled(
    *,
    source_audio_bytes: bytes,
    target_audio_bytes: bytes,
    source_text: str,
    target_text: str,
    sample_language: Optional[str],
    config: PipelineConfig,
    asr: Optional[ASRClient] = None,
    wer: Optional[WERClient] = None,
    utmos: Optional[UTMOSClient] = None,
    spk_sim: Optional[SpkSimClient] = None,
    shared_wdtw_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not config.evaluate:
        return {}
    from speech_edit.evaluation.audio import (
        delta_metrics,
        evaluate_audio,
    )

    source_metrics = evaluate_audio(
        source_audio_bytes,
        source_text,
        source_audio_bytes,
        asr_client=asr,
        wer_client=wer,
        utmos_client=utmos,
        spk_sim_client=None,
        sample_language=sample_language,
    )
    target_metrics = evaluate_audio(
        target_audio_bytes,
        target_text,
        source_audio_bytes,
        asr_client=asr,
        wer_client=wer,
        utmos_client=utmos,
        spk_sim_client=spk_sim,
        sample_language=sample_language,
    )
    target_dict = target_metrics.to_dict()
    result = {
        "src": source_metrics.to_dict(),
        "tgt": target_dict,
        "delta": delta_metrics(source_metrics, target_metrics),
    }
    if target_metrics.spk_similarity >= 0:
        result["spk_sim"] = {"similarity": target_metrics.spk_similarity}
    if target_metrics.asr_text:
        metric_language = wer_language_code(sample_language, reference_text=target_text, prefer_auto=True)
        local_result = _compute_local_wer(
            original_text=source_text,
            target_text=target_text,
            hypothesis=target_metrics.asr_text,
            context_k=3,
            language=metric_language,
        )
        result["local_wer_result"] = local_result.to_dict()
    if shared_wdtw_result is not None:
        result["wdtw_result"] = shared_wdtw_result
        if isinstance(shared_wdtw_result, dict):
            result["wdtw"] = shared_wdtw_result.get("wdtw")
    return result


def _normalized_alignment_metadata(
    alignments: Optional[List[Dict[str, Any]]],
    *,
    backend: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not alignments:
        return None
    items: List[Dict[str, Any]] = []
    for item in alignments:
        if not isinstance(item, dict):
            continue
        try:
            start_sec = float(item["start"])
            end_sec = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_sec < start_sec:
            continue
        text = item.get("text", item.get("word", item.get("char", "")))
        items.append({"text": str(text), "span": [start_sec, end_sec]})
    if not items:
        return None
    return {
        "backend": str(backend or ""),
        "level": "word",
        "items": items,
    }


def build_pair_metadata(
    *,
    sample_id: str,
    source_id: str,
    target_id: str,
    source_text: str,
    target_text: str,
    instructions_st: str,
    source_duration: float,
    target_duration: float,
    source_edit_spans: List[List[float]],
    target_edit_spans: List[List[float]],
    eval_results: Dict[str, Any],
    generation_id: str,
    direction: str,
    instructions_nl: Dict[str, List[str]],
    source_metadata: Optional[Dict[str, Any]] = None,
    source_alignment: Optional[List[Dict[str, Any]]] = None,
    target_alignment: Optional[List[Dict[str, Any]]] = None,
    aligner_backend: Optional[str] = None,
    requested_span_level: Optional[str] = None,
    requested_op_count: Optional[int] = None,
) -> Dict[str, Any]:
    instructions = {
        "nl": instructions_nl,
        "st": instructions_st,
    }
    segments = build_segments_metadata(
        source_label=source_text,
        target_label=target_text,
        source_duration=source_duration,
        target_duration=target_duration,
        source_edit_spans=source_edit_spans,
        target_edit_spans=target_edit_spans,
    )
    normalized_source_alignment = _normalized_alignment_metadata(
        source_alignment,
        backend=aligner_backend,
    )
    normalized_target_alignment = _normalized_alignment_metadata(
        target_alignment,
        backend=aligner_backend,
    )
    if normalized_source_alignment is not None:
        segments["src"]["alignment"] = normalized_source_alignment
    if normalized_target_alignment is not None:
        segments["tgt"]["alignment"] = normalized_target_alignment
    edit_distance = _target_edit_distance_stats(
        source_text,
        target_text,
        language=(source_metadata or {}).get("language")
        if isinstance(source_metadata, dict)
        else None,
    )
    return {
        "pair_dir_name": pair_dir_name(source_id, target_id),
        "source_id": source_id,
        "target_id": target_id,
        "generation_id": generation_id,
        "direction": direction,
        "source_text": source_text,
        "target_text": target_text,
        "requested_span_level": requested_span_level,
        "requested_op_count": requested_op_count,
        "target_edit_distance": edit_distance,
        "instructions": instructions,
        "source_metadata": dict(source_metadata or {}),
        "segments": segments,
        "tag_diff": [
            {
                "kind": segment.kind,
                "source_text": segment.source.strip(),
                "target_text": segment.target.strip(),
                "source_char_span": [segment.source_start, segment.source_end],
            }
            for segment in edit_segments(instructions_st)
        ],
        "eval_results": eval_results,
    }


def _persist_sample_outputs(
    *,
    sample_dir: str,
    original_audio_bytes: bytes,
    generated_audio_bytes: Dict[str, bytes],
    pair_results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    save_sample_outputs(
        sample_dir,
        original_audio_bytes=original_audio_bytes,
        generated_audio_bytes=generated_audio_bytes,
        pair_results=pair_results,
        metadata=metadata,
    )


def execute_tagged_plan(
    *,
    segment: "Segment",
    sample_id: str,
    plan: TaggedEditPlan,
    config: PipelineConfig,
    aligner: AlignmentClient,
    f5: F5TTSClient,
    asr: Optional[ASRClient] = None,
    wer: Optional[WERClient] = None,
    utmos: Optional[UTMOSClient] = None,
    spk_sim: Optional[SpkSimClient] = None,
    wdtw: Optional[WDTWClient] = None,
    persist: bool = True,
    include_artifacts: bool = False,
) -> Dict[str, Any]:
    if config.synthesis_mode not in {"one_shot", "per_op"}:
        raise ValueError("synthesis_mode must be 'one_shot' or 'per_op'")
    t0 = time.time()
    source_text = segment.text
    source_duration = segment.duration
    segment_metadata = getattr(segment, "metadata", {}) or {}
    sample_language = segment_metadata.get("language") or segment_metadata.get(
        "sample_language"
    )

    generated_audio_bytes: Dict[str, bytes] = {}
    generation_results: Dict[str, Dict[str, Any]] = {}
    debug_results: Dict[str, Dict[str, Any]] = {}
    pair_results: List[Dict[str, Any]] = []
    should_cache_alignments = include_artifacts or config.evaluate
    alignment_cache: Optional[Dict[str, Any]] = None
    source_word_alignments: Optional[List[Dict[str, Any]]] = None

    if should_cache_alignments:
        alignment_cache = {
            "source_word_alignments": None,
            "generated_word_alignments_by_generation": {},
            "aligner_backend": config.aligner_backend,
        }

    for generation in plan.generations:
        generation_id = generation.generation_id
        st = generation.st
        target_text = render_target(st)
        validate_tagged_instruction(st, source_text, target_text)
        reverse_nl = plan.reverse_nl_by_generation.get(
            generation_id, _empty_nl_payload()
        )
        trace_id = f"text_edit-{sample_id}-{generation_id}-{uuid.uuid4().hex[:6]}"
        debug_dir = _debug_generation_dir(
            config,
            sample_id=sample_id,
            generation_id=generation_id,
        )
        duration_probe_meta: Dict[str, Any] = {}
        if config.synthesis_mode == "per_op":
            estimate_config = replace(config, duration_mode="estimate")
            source_plan_result = plan_tagged_edit(
                st=st,
                source_audio_bytes=segment.audio_bytes,
                source_duration=source_duration,
                aligner=aligner,
                language=sample_language,
                config=estimate_config,
                return_alignment_result=alignment_cache is not None and source_word_alignments is None,
            )
            if isinstance(source_plan_result, tuple):
                source_span_plan, source_alignment_result = source_plan_result
                source_word_alignments = source_alignment_result.alignments
                if alignment_cache is not None:
                    alignment_cache["source_word_alignments"] = source_word_alignments
                    alignment_cache["aligner_backend"] = (
                        source_alignment_result.backend or alignment_cache["aligner_backend"]
                    )
            else:
                source_span_plan = source_plan_result
            generated_bytes, synth_meta = synthesize_target_audio_per_op(
                st=st,
                source_audio_bytes=segment.audio_bytes,
                source_duration=source_duration,
                aligner=aligner,
                f5=f5,
                language=sample_language,
                config=config,
                trace_id=trace_id,
                span_level=generation.spec.span_level,
                debug_dir=debug_dir,
            )
            span_plan = SpanPlan(
                source_spans=source_span_plan.source_spans,
                target_spans=[],
                planned_spans=source_span_plan.planned_spans,
                target_duration=synth_meta["duration"],
                duration_mode="per_op",
                metadata={
                    **source_span_plan.metadata,
                    "synthesis_mode": "per_op",
                    "source_span_duration_mode": "estimate",
                },
            )
        else:
            span_plan, duration_probe_meta, source_alignment_result = plan_tagged_edit_with_duration_mode(
                st=st,
                source_audio_bytes=segment.audio_bytes,
                source_duration=source_duration,
                aligner=aligner,
                f5=f5,
                language=sample_language,
                config=config,
                trace_id=trace_id,
                span_level=generation.spec.span_level,
                return_alignment_result=alignment_cache is not None and source_word_alignments is None,
            )
            if source_alignment_result is not None and source_word_alignments is None:
                source_word_alignments = source_alignment_result.alignments
                if alignment_cache is not None:
                    alignment_cache["source_word_alignments"] = source_word_alignments
                    alignment_cache["aligner_backend"] = (
                        source_alignment_result.backend or alignment_cache["aligner_backend"]
                    )
            step_debug_dir = _debug_step_dir(debug_dir, 0)
            generated_bytes, synth_meta = synthesize_target_audio(
                st=st,
                source_audio_bytes=segment.audio_bytes,
                span_plan=span_plan,
                f5=f5,
                config=config,
                trace_id=trace_id,
                span_level=generation.spec.span_level,
                debug_step_dir=step_debug_dir,
            )
            synth_meta = {
                **synth_meta,
                "synthesis_mode": "one_shot",
                "f5_call_count": 1,
            }
        generated_duration = _duration_sec(generated_bytes)
        generated_audio_bytes[generation_id] = generated_bytes
        generated_word_alignments: Optional[List[Dict[str, Any]]] = None
        if alignment_cache is not None:
            target_aligned_spans, target_alignment_result = align_generated_target_spans(
                st=st,
                generated_audio_bytes=generated_bytes,
                generated_duration=generated_duration,
                aligner=aligner,
                language=sample_language,
                config=config,
                return_alignment_result=True,
            )
            generated_word_alignments = target_alignment_result.alignments
            alignment_cache["generated_word_alignments_by_generation"][generation_id] = (
                generated_word_alignments
            )
            alignment_cache["aligner_backend"] = (
                target_alignment_result.backend or alignment_cache["aligner_backend"]
            )
        else:
            target_aligned_spans = align_generated_target_spans(
                st=st,
                generated_audio_bytes=generated_bytes,
                generated_duration=generated_duration,
                aligner=aligner,
                language=sample_language,
                config=config,
            )
        if config.synthesis_mode == "per_op":
            span_plan = SpanPlan(
                source_spans=span_plan.source_spans,
                target_spans=target_aligned_spans,
                planned_spans=span_plan.planned_spans,
                target_duration=generated_duration,
                duration_mode=span_plan.duration_mode,
                metadata=span_plan.metadata,
            )
        debug_manifest = _write_debug_manifest(
            debug_dir=debug_dir,
            manifest={
                "sample_id": sample_id,
                "generation_id": generation_id,
                "synthesis_mode": config.synthesis_mode,
                "source_text": source_text,
                "target_text": target_text,
                "f5_call_count": synth_meta.get("f5_call_count", 1),
                "debug_dir": debug_dir,
                "steps": synth_meta.get(
                    "steps",
                    [
                        {
                            "step_index": 0,
                            "st": st,
                            "source_text": source_text,
                            "target_text": target_text,
                            "op_counts": op_counts(st),
                            "source_duration": source_duration,
                            "target_duration": generated_duration,
                            "span_plan": {
                                "source_spans": span_plan.source_spans,
                                "target_spans": span_plan.target_spans,
                                "metadata": span_plan.metadata,
                            },
                            "synthesis": synth_meta,
                            "artifacts": _debug_step_artifacts(0) if debug_dir else {},
                        }
                    ],
                ),
                "final": {
                    "duration": generated_duration,
                    "source_spans": span_plan.source_spans,
                    "target_spans": span_plan.target_spans,
                    "target_aligned_spans": target_aligned_spans,
                    "audio_sha256": hashlib.sha256(generated_bytes).hexdigest(),
                },
            },
        )
        if debug_manifest is not None:
            debug_results[generation_id] = debug_manifest
        shared_wdtw_result = compute_pair_wdtw_if_enabled(
            source_audio_bytes=segment.audio_bytes,
            target_audio_bytes=generated_bytes,
            source_text=source_text,
            target_text=target_text,
            sample_language=sample_language,
            config=config,
            wdtw=wdtw,
            original_alignments=source_word_alignments,
            edited_alignments=generated_word_alignments,
        )

        forward_eval = evaluate_pair_if_enabled(
            source_audio_bytes=segment.audio_bytes,
            target_audio_bytes=generated_bytes,
            source_text=source_text,
            target_text=target_text,
            sample_language=sample_language,
            config=config,
            asr=asr,
            wer=wer,
            utmos=utmos,
            spk_sim=spk_sim,
            shared_wdtw_result=shared_wdtw_result,
        )
        forward_pair = build_pair_metadata(
            sample_id=sample_id,
            source_id=ORIGINAL_WAVE_ID,
            target_id=generation_id,
            source_text=source_text,
            target_text=target_text,
            instructions_st=st,
            source_duration=source_duration,
            target_duration=generated_duration,
            source_edit_spans=span_plan.source_spans,
            target_edit_spans=target_aligned_spans
            if config.synthesis_mode == "per_op"
            else span_plan.target_spans,
            eval_results=forward_eval,
            generation_id=generation_id,
            direction="original_to_generated",
            instructions_nl=generation.forward_nl,
            source_metadata=segment_metadata,
            source_alignment=source_word_alignments,
            target_alignment=generated_word_alignments,
            aligner_backend=(
                str(alignment_cache.get("aligner_backend"))
                if alignment_cache is not None and alignment_cache.get("aligner_backend")
                else config.aligner_backend
            ),
            requested_span_level=generation.spec.span_level,
            requested_op_count=generation.spec.op_count,
        )

        reverse_st = reverse_tagged_instruction(st)
        reverse_eval = evaluate_pair_if_enabled(
            source_audio_bytes=generated_bytes,
            target_audio_bytes=segment.audio_bytes,
            source_text=target_text,
            target_text=source_text,
            sample_language=sample_language,
            config=config,
            asr=asr,
            wer=wer,
            utmos=utmos,
            spk_sim=spk_sim,
            shared_wdtw_result=shared_wdtw_result,
        )
        reverse_pair = build_pair_metadata(
            sample_id=sample_id,
            source_id=generation_id,
            target_id=ORIGINAL_WAVE_ID,
            source_text=target_text,
            target_text=source_text,
            instructions_st=reverse_st,
            source_duration=generated_duration,
            target_duration=source_duration,
            source_edit_spans=target_aligned_spans,
            target_edit_spans=span_plan.source_spans,
            eval_results=reverse_eval,
            generation_id=generation_id,
            direction="generated_to_original",
            instructions_nl=reverse_nl,
            source_metadata=segment_metadata,
            source_alignment=generated_word_alignments,
            target_alignment=source_word_alignments,
            aligner_backend=(
                str(alignment_cache.get("aligner_backend"))
                if alignment_cache is not None and alignment_cache.get("aligner_backend")
                else config.aligner_backend
            ),
            requested_span_level=generation.spec.span_level,
            requested_op_count=generation.spec.op_count,
        )
        pair_results.extend([forward_pair, reverse_pair])

        generation_results[generation_id] = {
            "generation_id": generation_id,
            "st": st,
            "canonical_st": generation.canonical_st,
            "reverse_st": reverse_st,
            "target_text": target_text,
            "op_counts": op_counts(st),
            "spec": generation.spec.to_metadata(),
            "forward_nl": generation.forward_nl,
            "reverse_nl": reverse_nl,
            "duration": generated_duration,
            "span_plan": {
                "source_spans": span_plan.source_spans,
                "target_spans": span_plan.target_spans,
                "target_aligned_spans": target_aligned_spans,
                "metadata": span_plan.metadata,
            },
            "synthesis": synth_meta,
            "debug": debug_manifest,
            **duration_probe_meta,
        }

    sample_dir = os.path.join(config.output_dir, sample_id)
    metadata = build_sample_metadata(
        sample_id=sample_id,
        clip_id=segment.clip_id,
        source_text=source_text,
        original_duration=source_duration,
        sample_rate=segment.sample_rate,
        generations=generation_results,
        pair_results=pair_results,
        pipeline_version=PIPELINE_VERSION,
        pipeline_type=PIPELINE_TYPE,
        config={
            "duration_mode": config.duration_mode,
            "synthesis_mode": config.synthesis_mode,
            "debug": config.debug,
            "debug_output_dir": config.debug_output_dir,
            "expand_radius": config.expand_radius,
            "merge_gap": config.merge_gap,
            "num_generations": config.num_generations,
            "span_level_weights": config.span_level_weights,
            "op_count_weights": config.op_count_weights,
            "min_edit_distance_pct": config.min_edit_distance_pct,
            "min_preserved_source_ratio": config.min_preserved_source_ratio,
            "prepend_context_enabled": config.prepend_context_enabled,
            "fail_if_cannot_prepend_context": config.fail_if_cannot_prepend_context,
            "min_unmasked_context_sec": config.min_unmasked_context_sec,
            "min_unmasked_context_ratio": config.min_unmasked_context_ratio,
            "llm_max_retries": config.llm_max_retries,
            "forward_nl_mode": config.forward_nl_mode,
            "reverse_nl_mode": config.reverse_nl_mode,
            "nl_instruction_schema": {
                "simple_total": NL_SIMPLE_COUNT,
                "hard_total": NL_HARD_COUNT,
                "per_language": {"en": 2, "zh": 2},
            },
            "aligner_backend": config.aligner_backend,
            "evaluate": config.evaluate,
        },
        source_metadata=segment_metadata,
    )
    if persist:
        _persist_sample_outputs(
            sample_dir=sample_dir,
            original_audio_bytes=segment.audio_bytes,
            generated_audio_bytes=generated_audio_bytes,
            pair_results=pair_results,
            metadata=metadata,
        )

    result = {
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "generations": len(generated_audio_bytes),
        "pairs": len(pair_results),
        "generation_specs": {
            generation_id: payload["spec"]
            for generation_id, payload in generation_results.items()
        },
        "generation_op_counts": {
            generation_id: payload["op_counts"]
            for generation_id, payload in generation_results.items()
        },
        "llm_forward_attempts": int(plan.forward_llm_stats.get("forward_attempts", 0)),
        "llm_forward_retry_attempts": int(
            plan.forward_llm_stats.get("forward_retry_attempts", 0)
        ),
        "llm_reverse_attempts": int(plan.reverse_llm_stats.get("reverse_attempts", 0)),
        "llm_reverse_retry_attempts": int(
            plan.reverse_llm_stats.get("reverse_retry_attempts", 0)
        ),
        "llm_validation_error_distribution": _combined_validation_errors(
            plan.forward_llm_stats.get("forward_validation_errors", {}),
            plan.reverse_llm_stats.get("reverse_validation_errors", {}),
        ),
        "prepend_context_disabled_generations": sum(
            1
            for payload in generation_results.values()
            if payload.get("synthesis", {}).get("prepend_context_disabled_reason")
        ),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    if debug_results:
        result["debug"] = debug_results
    if include_artifacts:
        result.update(
            {
                "alignment_cache": alignment_cache,
                "generated_audio_bytes": generated_audio_bytes,
                "metadata": metadata,
                "pair_results": pair_results,
                "debug_results": debug_results,
            }
        )
    return result


def _audio_bytes_to_wav_with_meta(audio_bytes: bytes) -> Tuple[bytes, int, float]:
    waveform, sample_rate = wav_bytes_to_mono_float(audio_bytes)
    return mono_float_to_wav_bytes(waveform, sample_rate), sample_rate, len(waveform) / sample_rate


def _refresh_metadata_pair_results(
    metadata: Dict[str, Any],
    pair_results: List[Dict[str, Any]],
    *,
    evaluate: bool,
) -> Dict[str, Any]:
    metadata["edit_pairs"] = [
        {
            key: value
            for key, value in pair.items()
            if key not in {"source_audio_bytes", "target_audio_bytes"}
        }
        for pair in pair_results
    ]
    metadata["pair_count"] = len(pair_results)
    metadata["pairs_with_eval"] = sum(
        1 for pair in pair_results if pair.get("eval_results")
    )
    metadata.setdefault("config", {})["evaluate"] = evaluate
    return metadata


def _evaluate_synthesized_result(
    item: SynthesizedWorkItem,
    *,
    config: PipelineConfig,
    asr: Optional[ASRClient] = None,
    wer: Optional[WERClient] = None,
    utmos: Optional[UTMOSClient] = None,
    spk_sim: Optional[SpkSimClient] = None,
    wdtw: Optional[WDTWClient] = None,
) -> SynthesizedWorkItem:
    if not config.evaluate:
        return item
    result = dict(item.result)
    pair_results = list(result.get("pair_results", []))
    generated_audio_bytes = dict(result.get("generated_audio_bytes", {}))
    metadata = dict(result.get("metadata", {}))
    alignment_cache = dict(result.get("alignment_cache") or {})
    source_word_alignments = alignment_cache.get("source_word_alignments")
    generated_word_alignments_by_generation = dict(
        alignment_cache.get("generated_word_alignments_by_generation") or {}
    )
    segment_metadata = getattr(item.segment, "metadata", {}) or {}
    sample_language = segment_metadata.get("language") or segment_metadata.get(
        "sample_language"
    )
    audio_by_id: Dict[str, bytes] = {ORIGINAL_WAVE_ID: item.segment.audio_bytes}
    audio_by_id.update(generated_audio_bytes)
    shared_wdtw_by_generation: Dict[str, Optional[Dict[str, Any]]] = {}

    for pair in pair_results:
        source_id = str(pair.get("source_id"))
        target_id = str(pair.get("target_id"))
        generation_id = str(pair.get("generation_id") or "")
        source_audio = audio_by_id[source_id]
        target_audio = audio_by_id[target_id]
        shared_wdtw = None
        if generation_id:
            if generation_id not in shared_wdtw_by_generation:
                generation_meta = metadata.get("generations", {}).get(generation_id, {})
                shared_wdtw_by_generation[generation_id] = compute_pair_wdtw_if_enabled(
                    source_audio_bytes=item.segment.audio_bytes,
                    target_audio_bytes=generated_audio_bytes[generation_id],
                    source_text=item.segment.text,
                    target_text=str(generation_meta.get("target_text") or pair.get("target_text") or ""),
                    sample_language=sample_language,
                    config=config,
                    wdtw=wdtw,
                    original_alignments=source_word_alignments,
                    edited_alignments=generated_word_alignments_by_generation.get(generation_id),
                )
            shared_wdtw = shared_wdtw_by_generation[generation_id]
        pair["eval_results"] = evaluate_pair_if_enabled(
            source_audio_bytes=source_audio,
            target_audio_bytes=target_audio,
            source_text=str(pair.get("source_text") or ""),
            target_text=str(pair.get("target_text") or ""),
            sample_language=sample_language,
            config=config,
            asr=asr,
            wer=wer,
            utmos=utmos,
            spk_sim=spk_sim,
            shared_wdtw_result=shared_wdtw,
        )

    result["pair_results"] = pair_results
    result["metadata"] = _refresh_metadata_pair_results(
        metadata,
        pair_results,
        evaluate=config.evaluate,
    )
    return SynthesizedWorkItem(
        sample_id=item.sample_id,
        segment=item.segment,
        attempt_index=item.attempt_index,
        result=result,
    )
