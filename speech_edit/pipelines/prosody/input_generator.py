"""
Input generator for the pitch/rate editing pipeline (Stage I).

Workflow
--------
1. Sample an utterance from the configured source.
2. Select edit kinds and span granularities for generated waves.
3. Sample allowed pitch/rate parameter pools.
4. Call the LLM to generate all tagged waves + dual instructions.
4. Parse and validate the LLM output (against normed_text).
5. Return the complete input specification for Stage II.
"""

import copy
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .prompts import build_prompt, build_reverse_instruction_prompt
from .tag_utils import (
    ORIGINAL_WAVE_ID,
    WaveGroup,
    invert_structural_instruction,
    build_wave_group,
    normalize_pair_instructions_for_wave,
    pair_key,
    parse_tags,
    summarize_tagged_text,
    validate_tagged_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Axis selection
# ---------------------------------------------------------------------------

VALID_EDIT_KINDS = ["pitch", "rate", "mixed"]
VALID_SPAN_LEVELS = ["single_word", "short_phrase", "clause", "sentence"]
VALID_SAMPLE_LANGUAGES = ("en", "zh")
MAX_EDIT_SPANS_PER_WAVE = 3
VALID_REGION_COUNTS = (1, 2, 3)
SHORT_TEXT_MAX_REGIONS = 1
MEDIUM_TEXT_MAX_REGIONS = 2


def select_edit_kind(weights: Optional[Dict[str, float]] = None) -> str:
    """Randomly select an edit kind from normalized weights."""
    if weights is None:
        raise ValueError("Sampling weights must be configured")
    raw_weights = dict(weights)
    unknown = set(raw_weights) - set(VALID_EDIT_KINDS)
    if unknown:
        raise ValueError(f"Unknown edit kinds: {sorted(unknown)}")
    kinds = []
    values = []
    for kind in VALID_EDIT_KINDS:
        weight = float(raw_weights.get(kind, 0.0))
        if weight > 0:
            kinds.append(kind)
            values.append(weight)
    if not kinds:
        raise ValueError("At least one edit kind weight must be positive")
    return random.choices(kinds, weights=values, k=1)[0]


def select_span_level(
    weights: Optional[Dict[str, float]] = None,
    *,
    allow_sentence: bool = True,
) -> str:
    """Randomly select a span granularity from normalized weights."""
    if weights is None:
        raise ValueError("Sampling weights must be configured")
    raw_weights = dict(weights)
    if not allow_sentence:
        raw_weights["sentence"] = 0.0
    unknown = set(raw_weights) - set(VALID_SPAN_LEVELS)
    if unknown:
        raise ValueError(f"Unknown span levels: {sorted(unknown)}")
    levels = []
    values = []
    for level in VALID_SPAN_LEVELS:
        weight = float(raw_weights.get(level, 0.0))
        if weight > 0:
            levels.append(level)
            values.append(weight)
    if not levels:
        raise ValueError("At least one span level weight must be positive")
    return random.choices(levels, weights=values, k=1)[0]


def normalize_region_count_weights(
    weights: Optional[Mapping[Any, Any]],
) -> Optional[Dict[int, float]]:
    """Normalize optional exact-region-count weights.

    ``None`` keeps the legacy 1-N prompt behavior.  A mapping enables exact
    count sampling after edit kind/span selection.
    """
    if weights is None:
        return None
    if not isinstance(weights, Mapping):
        raise ValueError("region_count_weights must be an object")
    normalized: Dict[int, float] = {}
    for raw_key, raw_value in weights.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError):
            raise ValueError("region_count_weights keys must be integers 1, 2, or 3") from None
        if key not in VALID_REGION_COUNTS:
            raise ValueError(f"Unknown region_count_weights key: {raw_key!r}")
        value = float(raw_value)
        if value < 0:
            raise ValueError("region_count_weights values must be non-negative")
        normalized[key] = value
    if not any(value > 0 for value in normalized.values()):
        raise ValueError("region_count_weights must contain at least one positive weight")
    return {count: float(normalized.get(count, 0.0)) for count in VALID_REGION_COUNTS}


def select_region_count(
    *,
    edit_kind: str,
    span_level: str,
    max_regions: int,
    weights: Mapping[int, float],
) -> int:
    if span_level == "sentence":
        return 1
    normalized = dict(weights)
    for count in VALID_REGION_COUNTS:
        if count > max_regions:
            normalized[count] = 0.0
        if edit_kind == "mixed" and count < 2:
            normalized[count] = 0.0
    counts = [count for count in VALID_REGION_COUNTS if normalized.get(count, 0.0) > 0]
    values = [normalized[count] for count in counts]
    if not counts:
        raise ValueError(
            "No legal region_count_weights remain for "
            f"edit_kind={edit_kind!r}, span_level={span_level!r}, max_regions={max_regions}"
        )
    return random.choices(counts, weights=values, k=1)[0]


def validate_edit_axes(edit_kind: str, span_level: str) -> None:
    """Validate the two-axis edit configuration."""
    if edit_kind not in VALID_EDIT_KINDS:
        raise ValueError(f"edit_kind must be one of {VALID_EDIT_KINDS}, got {edit_kind!r}")
    if span_level not in VALID_SPAN_LEVELS:
        raise ValueError(
            f"span_level must be one of {VALID_SPAN_LEVELS}, got {span_level!r}"
        )
    if edit_kind == "mixed" and span_level == "sentence":
        raise ValueError("mixed edit_kind does not support sentence span level")


def normalize_sample_language(language: Optional[str]) -> Optional[str]:
    """Normalize an optional Stage 0 language hint."""
    if language is None or language == "":
        return None
    normalized = str(language).strip().lower()
    if normalized not in VALID_SAMPLE_LANGUAGES:
        raise ValueError(
            f"sample_language must be one of {VALID_SAMPLE_LANGUAGES}, got {language!r}"
        )
    return normalized


def infer_text_language(text: str, language: Optional[str]) -> str:
    """Infer the language used for Stage 1 text-length policies."""
    normalized = normalize_sample_language(language)
    if normalized is not None:
        return normalized
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def count_stage1_text_units(text: str, language: Optional[str]) -> int:
    """Count text units for short-utterance Stage 1 policy decisions."""
    inferred = infer_text_language(text, language)
    if inferred == "zh":
        return len(re.findall(r"[\u4e00-\u9fff]", text))
    return len(re.findall(r"[A-Za-z0-9]+", text))


def stage1_text_policy(text: str, language: Optional[str]) -> Dict[str, Any]:
    """Return text-length bucket and max edit regions for one sampled utterance."""
    inferred = infer_text_language(text, language)
    unit_count = count_stage1_text_units(text, inferred)
    if inferred == "zh":
        if unit_count < 8:
            bucket = "short"
            max_regions = SHORT_TEXT_MAX_REGIONS
        elif unit_count <= 12:
            bucket = "medium"
            max_regions = MEDIUM_TEXT_MAX_REGIONS
        else:
            bucket = "long"
            max_regions = MAX_EDIT_SPANS_PER_WAVE
    else:
        if unit_count < 5:
            bucket = "short"
            max_regions = SHORT_TEXT_MAX_REGIONS
        elif unit_count <= 8:
            bucket = "medium"
            max_regions = MEDIUM_TEXT_MAX_REGIONS
        else:
            bucket = "long"
            max_regions = MAX_EDIT_SPANS_PER_WAVE
    return {
        "language": inferred,
        "text_unit_count": unit_count,
        "text_length_bucket": bucket,
        "effective_max_regions": max_regions,
    }


LLM_SERVICE_TIMEOUT_SEC = 24 * 3600


class LLMCallError(RuntimeError):
    """Raised when the LLM service call itself fails."""


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------


def _positive_weighted_buckets(
    buckets: Mapping[str, Any],
    *,
    field_name: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for name, raw_bucket in buckets.items():
        if not isinstance(raw_bucket, Mapping):
            raise ValueError(f"{field_name}.{name} must be an object")
        bucket = dict(raw_bucket)
        weight = float(bucket.get("weight", 0.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{field_name}.{name}.weight must be non-negative")
        if weight > 0:
            bucket["name"] = str(name)
            bucket["weight"] = weight
            selected.append(bucket)
    if not selected:
        raise ValueError(f"{field_name} must contain at least one positive-weight bucket")
    return selected


def _validate_pitch_buckets(buckets: Mapping[str, Any]) -> None:
    for bucket in _positive_weighted_buckets(
        buckets,
        field_name="parameter_policy.pitch.buckets",
    ):
        values = bucket.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"parameter_policy.pitch.buckets.{bucket['name']}.values must be a non-empty list"
            )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"parameter_policy.pitch.buckets.{bucket['name']}.values must contain integers"
                )
            if value == 0:
                raise ValueError(
                    f"parameter_policy.pitch.buckets.{bucket['name']}.values must not contain 0"
                )


def _validate_rate_buckets(buckets: Mapping[str, Any], *, decimals: int) -> None:
    no_op_low = 1.0 - 0.5 * (10 ** -decimals)
    no_op_high = 1.0 + 0.5 * (10 ** -decimals)
    for bucket in _positive_weighted_buckets(
        buckets,
        field_name="parameter_policy.rate.buckets",
    ):
        ranges = bucket.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(
                f"parameter_policy.rate.buckets.{bucket['name']}.ranges must be a non-empty list"
            )
        for raw_range in ranges:
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ValueError(
                    f"parameter_policy.rate.buckets.{bucket['name']}.ranges must contain [low, high] pairs"
                )
            low = float(raw_range[0])
            high = float(raw_range[1])
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or low <= 0
                or high <= 0
                or low >= high
            ):
                raise ValueError(
                    f"parameter_policy.rate.buckets.{bucket['name']}.ranges must be positive increasing ranges"
                )
            if low <= 1.0 <= high:
                raise ValueError(
                    f"parameter_policy.rate.buckets.{bucket['name']}.ranges must not cross 1.0"
                )
            if low < no_op_high and high > no_op_low:
                raise ValueError(
                    f"parameter_policy.rate.buckets.{bucket['name']}.ranges can round to 1.0"
                )


def normalize_parameter_policy(
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a validated, fully expanded pitch/rate parameter policy."""
    if not isinstance(policy, Mapping):
        raise ValueError("parameter_policy must be an object")
    normalized = copy.deepcopy(dict(policy))
    if "values_per_wave" not in normalized:
        raise ValueError("parameter_policy.values_per_wave must be configured")
    for axis in ("pitch", "rate"):
        if isinstance(normalized.get(axis), Mapping) and normalized[axis].get("buckets") == {}:
            raise ValueError(f"parameter_policy.{axis}.buckets must not be empty")

    values_per_wave = normalized["values_per_wave"]
    if isinstance(values_per_wave, bool) or not isinstance(values_per_wave, int):
        raise ValueError("parameter_policy.values_per_wave must be a positive integer")
    if values_per_wave < 1:
        raise ValueError("parameter_policy.values_per_wave must be >= 1")
    normalized["values_per_wave"] = values_per_wave
    normalized["max_edit_spans_per_wave"] = MAX_EDIT_SPANS_PER_WAVE

    pitch = normalized.get("pitch")
    if not isinstance(pitch, Mapping) or not isinstance(pitch.get("buckets"), Mapping):
        raise ValueError("parameter_policy.pitch.buckets must be an object")
    _validate_pitch_buckets(pitch["buckets"])

    rate = normalized.get("rate")
    if not isinstance(rate, Mapping) or not isinstance(rate.get("buckets"), Mapping):
        raise ValueError("parameter_policy.rate.buckets must be an object")
    if "decimals" not in rate:
        raise ValueError("parameter_policy.rate.decimals must be configured")
    decimals = rate["decimals"]
    if isinstance(decimals, bool) or not isinstance(decimals, int):
        raise ValueError("parameter_policy.rate.decimals must be a non-negative integer")
    if decimals < 0:
        raise ValueError("parameter_policy.rate.decimals must be >= 0")
    normalized["rate"]["decimals"] = decimals
    _validate_rate_buckets(rate["buckets"], decimals=decimals)

    return normalized


def sample_pitch_values(
    num_values: int,
    policy: Mapping[str, Any],
) -> List[int]:
    """Sample integer semitone values for the LLM to choose from."""
    normalized = normalize_parameter_policy(policy)
    buckets = _positive_weighted_buckets(
        normalized["pitch"]["buckets"],
        field_name="parameter_policy.pitch.buckets",
    )
    weights = [bucket["weight"] for bucket in buckets]
    values: List[int] = []
    for _ in range(num_values):
        bucket = random.choices(buckets, weights=weights, k=1)[0]
        values.append(int(random.choice(bucket["values"])))
    return values


def sample_rate_values(
    num_values: int,
    policy: Mapping[str, Any],
) -> List[float]:
    """Sample two-decimal rate factors for the LLM to choose from."""
    normalized = normalize_parameter_policy(policy)
    buckets = _positive_weighted_buckets(
        normalized["rate"]["buckets"],
        field_name="parameter_policy.rate.buckets",
    )
    decimals = int(normalized["rate"]["decimals"])
    weights = [bucket["weight"] for bucket in buckets]
    values: List[float] = []
    for _ in range(num_values):
        bucket = random.choices(buckets, weights=weights, k=1)[0]
        ranges = bucket["ranges"]
        low, high = random.choice(ranges)
        values.append(round(random.uniform(float(low), float(high)), decimals))
    return values


def parameter_policy_metadata(
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    return normalize_parameter_policy(policy)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Stage0Sample:
    """Dataset utterance sampled before any LLM work."""

    clip_id: str
    transcript: str
    audio_bytes: bytes
    sample_rate: int
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineInput:
    """Complete input for one pipeline sample (Stages II-IV)."""

    clip_id: str
    transcript: str  # raw transcript from dataset
    normed_text: str  # normalized spoken-form text (from LLM)
    audio_bytes: bytes  # source audio WAV
    sample_rate: int
    duration: float
    wave_group: WaveGroup
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_llm_json(raw: str) -> Dict:
    """Best-effort parse of LLM JSON output.

    Handles common issues:
    - Markdown code blocks (```json ... ```)
    - Single quotes instead of double quotes
    - Trailing commas
    """
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix single quotes
    try:
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before } or ]
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse LLM JSON output: {text[:200]}")


# ---------------------------------------------------------------------------
# Wave plan helpers
# ---------------------------------------------------------------------------


def _select_wave_configs(
    *,
    num_waves: int,
    edit_kind: Optional[str],
    edit_kind_weights: Optional[Dict[str, float]],
    span_level: Optional[str],
    span_level_weights: Optional[Dict[str, float]],
    max_regions: int = MAX_EDIT_SPANS_PER_WAVE,
    region_count_weights: Optional[Mapping[Any, Any]] = None,
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    normalized_region_count_weights = normalize_region_count_weights(region_count_weights)
    effective_edit_kind_weights = edit_kind_weights
    if edit_kind == "mixed" and max_regions < 2:
        raise ValueError("mixed edit_kind requires at least 2 allowed edit regions")
    if edit_kind is None and max_regions < 2:
        effective_edit_kind_weights = dict(edit_kind_weights)
        effective_edit_kind_weights["mixed"] = 0.0
    for wave_idx in range(1, num_waves + 1):
        selected_kind = edit_kind or select_edit_kind(effective_edit_kind_weights)
        selected_span = span_level or select_span_level(
            span_level_weights,
            allow_sentence=selected_kind != "mixed",
        )
        validate_edit_axes(selected_kind, selected_span)
        target_num_regions = (
            select_region_count(
                edit_kind=selected_kind,
                span_level=selected_span,
                max_regions=max_regions,
                weights=normalized_region_count_weights,
            )
            if normalized_region_count_weights is not None
            else None
        )

        config = {
            "wave_id": str(wave_idx),
            "edit_kind": selected_kind,
            "span_level": selected_span,
            "max_regions": max_regions,
        }
        if target_num_regions is not None:
            config["target_num_regions"] = target_num_regions
        configs.append(config)
    return configs


def _validate_sentence_span(tagged_text: str, normed_text: str) -> None:
    segments = [segment for segment in parse_tags(tagged_text) if segment.is_tagged]
    if len(segments) != 1:
        raise ValueError("Sentence span must contain exactly one tagged region")
    if segments[0].text != normed_text:
        raise ValueError("Sentence span must wrap the entire normalized text")


def _validate_wave_result(
    *,
    wave_id: str,
    result: Dict[str, Any],
    normed_text: str,
    wave_config: Dict[str, Any],
    pitch_values: List[int],
    rate_values: List[float],
) -> Dict[str, int]:
    tagged_text = result["tagged_text"]
    is_valid, err_msg = validate_tagged_text(tagged_text, normed_text)
    if not is_valid:
        raise ValueError(f"Wave {wave_id} validation failed: {err_msg}")

    summary = summarize_tagged_text(tagged_text)
    max_regions = int(wave_config.get("max_regions", MAX_EDIT_SPANS_PER_WAVE))
    if summary["num_regions"] > max_regions:
        raise ValueError(
            f"Wave {wave_id} has {summary['num_regions']} regions; "
            f"max is {max_regions}"
        )
    target_num_regions = wave_config.get("target_num_regions")
    if target_num_regions is not None and summary["num_regions"] != int(target_num_regions):
        raise ValueError(
            f"Wave {wave_id} has {summary['num_regions']} regions; "
            f"target is {int(target_num_regions)}"
        )

    expected_kind = wave_config["edit_kind"]
    expected_span = wave_config["span_level"]
    if expected_kind == "pitch":
        if summary["num_pitch_regions"] < 1:
            raise ValueError(f"Wave {wave_id} expected at least one pitch edit")
        if summary["num_rate_regions"] > 0:
            raise ValueError(f"Wave {wave_id} expected pitch-only edits")
    if expected_kind == "rate":
        if summary["num_rate_regions"] < 1:
            raise ValueError(f"Wave {wave_id} expected at least one rate edit")
        if summary["num_pitch_regions"] > 0:
            raise ValueError(f"Wave {wave_id} expected rate-only edits")
    if expected_kind == "mixed":
        if summary["num_pitch_regions"] < 1 or summary["num_rate_regions"] < 1:
            raise ValueError(f"Wave {wave_id} expected mixed pitch/rate edits")
    if expected_span == "sentence":
        _validate_sentence_span(tagged_text, normed_text)

    allowed_pitch = {int(value) for value in pitch_values}
    allowed_rate = {round(float(value), 2) for value in rate_values}
    for segment in parse_tags(tagged_text):
        if segment.semitones is not None:
            if int(segment.semitones) != segment.semitones:
                raise ValueError(f"Wave {wave_id} pitch value must be an integer")
            if int(segment.semitones) not in allowed_pitch:
                raise ValueError(
                    f"Wave {wave_id} pitch value {segment.semitones} "
                    "is not in pitch_values"
                )
        if segment.factor is not None:
            factor = round(float(segment.factor), 2)
            if abs(float(segment.factor) - factor) > 1e-9:
                raise ValueError(f"Wave {wave_id} rate value must have <=2 decimals")
            if factor not in allowed_rate:
                raise ValueError(
                    f"Wave {wave_id} rate value {segment.factor} is not in rate_values"
                )
    return summary


def _validate_nl_payload(raw: Any, *, label: str) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    precise = raw.get("precise")
    vague = raw.get("vague")
    if not isinstance(precise, str) or not precise.strip():
        raise ValueError(f"{label}.precise must be a non-empty string")
    if not isinstance(vague, str) or not vague.strip():
        raise ValueError(f"{label}.vague must be a non-empty string")
    return {"precise": precise.strip(), "vague": vague.strip()}


def _raw_pair_instructions_for_wave(
    *,
    wave_id: str,
    forward_nl: Optional[Dict[str, Any]],
    reverse_nl: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {}
    if forward_nl is not None:
        raw[pair_key(ORIGINAL_WAVE_ID, wave_id)] = {"nl": forward_nl}
    if reverse_nl is not None:
        raw[pair_key(wave_id, ORIGINAL_WAVE_ID)] = {"nl": reverse_nl}
    return raw


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_input_from_sampled_data(
    sampled: Stage0Sample,
    llm_client,
    *,
    edit_kind: Optional[str] = None,
    edit_kind_weights: Optional[Dict[str, float]] = None,
    span_level: Optional[str] = None,
    span_level_weights: Optional[Dict[str, float]] = None,
    num_waves: int,
    llm_backend: Optional[str],
    llm_model: Optional[str] = None,
    temperature: float,
    sample_language: Optional[str] = None,
    stage1_seed: Optional[int] = None,
    parameter_policy: Mapping[str, Any],
    max_retries: int,
    region_count_weights: Optional[Mapping[Any, Any]] = None,
    forward_nl_mode: str,
    reverse_nl_mode: str,
    raise_on_llm_service_error: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[PipelineInput]:
    """Generate one pipeline input from an already sampled utterance.

    Stage 1 owns all random edit configuration, allowed parameter sampling, LLM
    generation, validation, and WaveGroup construction.  Passing *stage1_seed*
    makes this work deterministic per sampled item even when multiple Stage 1
    workers run concurrently.
    """
    if stage1_seed is not None:
        random_state = random.getstate()
        random.seed(stage1_seed)
    else:
        random_state = None

    try:
        return _generate_input_from_sampled_data_impl(
            sampled,
            llm_client,
            edit_kind=edit_kind,
            edit_kind_weights=edit_kind_weights,
            span_level=span_level,
            span_level_weights=span_level_weights,
            num_waves=num_waves,
            llm_backend=llm_backend,
            llm_model=llm_model,
            temperature=temperature,
            sample_language=sample_language,
            parameter_policy=parameter_policy,
            max_retries=max_retries,
            region_count_weights=region_count_weights,
            forward_nl_mode=forward_nl_mode,
            reverse_nl_mode=reverse_nl_mode,
            raise_on_llm_service_error=raise_on_llm_service_error,
            diagnostics=diagnostics,
        )
    finally:
        if random_state is not None:
            random.setstate(random_state)


def _generate_input_from_sampled_data_impl(
    sampled: Stage0Sample,
    llm_client,
    *,
    edit_kind: Optional[str] = None,
    edit_kind_weights: Optional[Dict[str, float]] = None,
    span_level: Optional[str] = None,
    span_level_weights: Optional[Dict[str, float]] = None,
    num_waves: int,
    llm_backend: Optional[str],
    llm_model: Optional[str] = None,
    temperature: float,
    sample_language: Optional[str] = None,
    parameter_policy: Mapping[str, Any],
    max_retries: int,
    region_count_weights: Optional[Mapping[Any, Any]] = None,
    forward_nl_mode: str,
    reverse_nl_mode: str,
    raise_on_llm_service_error: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[PipelineInput]:
    if forward_nl_mode not in {"template", "llm"}:
        raise ValueError("forward_nl_mode must be 'template' or 'llm'")
    if reverse_nl_mode not in {"template", "llm"}:
        raise ValueError("reverse_nl_mode must be 'template' or 'llm'")
    transcript = sampled.transcript
    language = normalize_sample_language(
        sample_language or sampled.metadata.get("language")
    )
    text_policy = stage1_text_policy(transcript, language)
    if language is None:
        language = text_policy["language"]
    stage1_diagnostics: Dict[str, Any] = {
        "required_wave_ids": [str(index) for index in range(1, num_waves + 1)],
        "validation_attempts": [],
        "text_unit_count": text_policy["text_unit_count"],
        "text_length_bucket": text_policy["text_length_bucket"],
        "effective_max_regions": text_policy["effective_max_regions"],
        "language": language,
        "forward_nl_mode": forward_nl_mode,
        "reverse_nl_mode": reverse_nl_mode,
    }
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(stage1_diagnostics)

    if num_waves < 1:
        raise ValueError("num_waves must be >= 1")

    if span_level is not None and span_level not in VALID_SPAN_LEVELS:
        raise ValueError(
            f"span_level must be one of {VALID_SPAN_LEVELS}, got {span_level!r}"
        )
    if edit_kind is not None and edit_kind not in VALID_EDIT_KINDS:
        raise ValueError(
            f"edit_kind must be one of {VALID_EDIT_KINDS}, got {edit_kind!r}"
        )
    if edit_kind is not None and span_level is not None:
        validate_edit_axes(edit_kind, span_level)

    try:
        wave_configs = _select_wave_configs(
            num_waves=num_waves,
            edit_kind=edit_kind,
            edit_kind_weights=edit_kind_weights,
            span_level=span_level,
            span_level_weights=span_level_weights,
            max_regions=int(text_policy["effective_max_regions"]),
            region_count_weights=region_count_weights,
        )
    except ValueError as exc:
        logger.warning("Invalid edit controls: %s", exc)
        stage1_diagnostics["last_error"] = str(exc)
        stage1_diagnostics["error_type"] = "invalid_edit_controls"
        if diagnostics is not None:
            diagnostics.update(stage1_diagnostics)
        return None

    stage1_diagnostics["wave_configs"] = wave_configs
    if diagnostics is not None:
        diagnostics.update(stage1_diagnostics)

    for config in wave_configs:
        logger.info(
            "Wave %s/%d: edit_kind=%s span=%s max_regions=%s",
            config["wave_id"],
            num_waves,
            config["edit_kind"],
            config["span_level"],
            config["max_regions"],
        )

    try:
        normalized_parameter_policy = parameter_policy_metadata(parameter_policy)
    except ValueError as exc:
        logger.warning("Invalid parameter policy: %s", exc)
        stage1_diagnostics["last_error"] = str(exc)
        stage1_diagnostics["error_type"] = "invalid_parameter_policy"
        if diagnostics is not None:
            diagnostics.update(stage1_diagnostics)
        return None

    num_parameter_values = num_waves * int(normalized_parameter_policy["values_per_wave"])
    pitch_values = sample_pitch_values(num_parameter_values, normalized_parameter_policy)
    rate_values = sample_rate_values(num_parameter_values, normalized_parameter_policy)
    parameter_values = {
        "pitch_values": pitch_values,
        "rate_values": rate_values,
    }

    tagged_texts: Dict[str, str] = {}
    pair_instructions: Dict[str, Dict[str, Any]] = {}
    normed_text: Optional[str] = None
    span_levels_used: List[str] = []
    edit_kinds_used: List[str] = []
    wave_specs: Dict[str, Dict[str, Any]] = {}

    llm_result: Optional[Dict[str, Any]] = None
    for attempt in range(max_retries):
        attempt_diagnostics: Dict[str, Any] = {"attempt": attempt + 1}
        try:
            result = _call_llm_for_edits(
                transcript,
                llm_client,
                wave_configs=wave_configs,
                pitch_values=pitch_values,
                rate_values=rate_values,
                language=language,
                backend=llm_backend,
                model=llm_model,
                temperature=temperature,
                forward_nl_mode=forward_nl_mode,
                diagnostics=attempt_diagnostics,
            )
            candidate_normed_text = result["normed_text"]
            candidate_waves = result["waves"]
            candidate_tagged_texts: Dict[str, str] = {}
            candidate_pair_instructions: Dict[str, Dict[str, Any]] = {}
            candidate_wave_specs: Dict[str, Dict[str, Any]] = {}
            for config in wave_configs:
                wave_id = config["wave_id"]
                wave_result = candidate_waves[wave_id]
                edit_counts = _validate_wave_result(
                    wave_id=wave_id,
                    result=wave_result,
                    normed_text=candidate_normed_text,
                    wave_config=config,
                    pitch_values=pitch_values,
                    rate_values=rate_values,
                )
                candidate_tagged_texts[wave_id] = wave_result["tagged_text"]
                candidate_wave_specs[wave_id] = {
                    "edit_kind": config["edit_kind"],
                    "span_granularity": config["span_level"],
                    "max_regions": config.get("max_regions", MAX_EDIT_SPANS_PER_WAVE),
                    "target_num_regions": config.get("target_num_regions"),
                    **edit_counts,
                }

            reverse_nl_by_wave: Dict[str, Dict[str, str]] = {}
            if reverse_nl_mode == "llm":
                reverse_nl_by_wave = _call_llm_for_reverse_instructions(
                    candidate_normed_text,
                    candidate_tagged_texts,
                    llm_client,
                    language=language,
                    backend=llm_backend,
                    model=llm_model,
                    temperature=temperature,
                    diagnostics=attempt_diagnostics,
                )

            for config in wave_configs:
                wave_id = config["wave_id"]
                wave_result = candidate_waves[wave_id]
                raw_pair_instructions = _raw_pair_instructions_for_wave(
                    wave_id=wave_id,
                    forward_nl=wave_result.get("forward_nl"),
                    reverse_nl=reverse_nl_by_wave.get(wave_id),
                )
                candidate_pair_instructions[wave_id] = normalize_pair_instructions_for_wave(
                    candidate_normed_text,
                    wave_id,
                    wave_result["tagged_text"],
                    raw_pair_instructions,
                    allow_filled_reverse=reverse_nl_mode == "template",
                )

            normed_text = candidate_normed_text
            tagged_texts = candidate_tagged_texts
            pair_instructions = candidate_pair_instructions
            wave_specs = candidate_wave_specs
            llm_result = result
            attempt_diagnostics["status"] = "accepted"
            stage1_diagnostics["parsed_wave_ids"] = attempt_diagnostics.get(
                "parsed_wave_ids",
                list(candidate_waves.keys()),
            )
            stage1_diagnostics["missing_wave_ids"] = []
            break
        except LLMCallError as exc:
            logger.error("LLM service failed: %s", exc)
            stage1_diagnostics["last_error"] = str(exc)
            stage1_diagnostics["error_type"] = "llm_service_failed"
            if diagnostics is not None:
                diagnostics.update(stage1_diagnostics)
            if raise_on_llm_service_error:
                raise
            return None
        except Exception as exc:
            attempt_diagnostics["status"] = "rejected"
            attempt_diagnostics["error_message"] = str(exc)
            stage1_diagnostics["last_error"] = str(exc)
            stage1_diagnostics["error_type"] = type(exc).__name__
            logger.warning(
                "LLM output rejected (validation attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                exc,
            )
        finally:
            stage1_diagnostics["validation_attempts"].append(attempt_diagnostics)
            for key in (
                "prompt_length",
                "raw_response_length",
                "raw_response_preview",
                "parsed_wave_ids",
                "missing_wave_ids",
                "reverse_prompt_length",
                "reverse_raw_response_length",
                "reverse_raw_response_preview",
            ):
                if key in attempt_diagnostics:
                    stage1_diagnostics[key] = attempt_diagnostics[key]
            if diagnostics is not None:
                diagnostics.update(stage1_diagnostics)

    if llm_result is None or normed_text is None:
        logger.error("LLM validation failed for clip %s", sampled.clip_id)
        if diagnostics is not None:
            diagnostics.update(stage1_diagnostics)
        return None

    edit_kinds_used = [config["edit_kind"] for config in wave_configs]
    span_levels_used = [config["span_level"] for config in wave_configs]
    try:
        wave_group = build_wave_group(
            normed_text,
            tagged_texts,
            metadata={
                "num_waves": num_waves,
                "wave_specs": wave_specs,
                "pair_instructions": pair_instructions,
                "edit_kinds_used": edit_kinds_used,
                "span_granularities_used": span_levels_used,
                "parameter_values": parameter_values,
                "parameter_policy": normalized_parameter_policy,
                "region_count_weights": (
                    normalize_region_count_weights(region_count_weights)
                    if region_count_weights is not None
                    else None
                ),
                "language": language,
                "forward_nl_source": forward_nl_mode,
                "reverse_nl_source": reverse_nl_mode,
                "stage1_diagnostics": stage1_diagnostics,
            },
        )
    except Exception as e:
        logger.warning("Wave-group construction failed: %s", e)
        stage1_diagnostics["last_error"] = str(e)
        stage1_diagnostics["error_type"] = "wave_group_construction_failed"
        if diagnostics is not None:
            diagnostics.update(stage1_diagnostics)
        return None

    logger.info("Generated wave group: %d waves", len(wave_group.tagged_texts))

    return PipelineInput(
        clip_id=sampled.clip_id,
        transcript=transcript,
        normed_text=normed_text,
        audio_bytes=sampled.audio_bytes,
        sample_rate=sampled.sample_rate,
        duration=sampled.duration,
        wave_group=wave_group,
        metadata={
            "language": language,
            "stage0_info": dict(sampled.metadata),
            "llm_backend": llm_backend or "default",
            "llm_model": llm_model or "default",
            "llm_temperature": temperature,
            "edit_kind": edit_kind,
            "edit_kind_weights": edit_kind_weights,
            "edit_kinds_used": edit_kinds_used,
            "span_granularity": span_level,
            "span_granularity_weights": span_level_weights,
            "span_granularities_used": span_levels_used,
            "wave_specs": wave_specs,
            "pair_instructions": pair_instructions,
            "parameter_values": parameter_values,
            "parameter_policy": normalized_parameter_policy,
            "region_count_weights": (
                normalize_region_count_weights(region_count_weights)
                if region_count_weights is not None
                else None
            ),
            "num_waves": num_waves,
            "forward_nl_source": forward_nl_mode,
            "reverse_nl_source": reverse_nl_mode,
            "stage1_diagnostics": stage1_diagnostics,
        },
    )


def _call_llm_for_edits(
    transcript: str,
    llm_client,
    *,
    wave_configs: List[Dict[str, Any]],
    pitch_values: List[int],
    rate_values: List[float],
    language: Optional[str] = None,
    backend: Optional[str],
    model: Optional[str] = None,
    temperature: float,
    forward_nl_mode: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the LLM and parse its response.

    Returns:
        Dict with keys: normed_text and waves. Each wave contains tagged_text
        and optional forward natural-language instructions.

    Raises:
        ValueError: If the response cannot be parsed or is missing fields.
    """
    prompt = build_prompt(
        transcript,
        wave_configs=wave_configs,
        pitch_values=pitch_values,
        rate_values=rate_values,
        language=language,
        forward_nl_mode=forward_nl_mode,
    )
    if diagnostics is not None:
        diagnostics["required_wave_ids"] = [config["wave_id"] for config in wave_configs]
        diagnostics["prompt_length"] = len(prompt)

    kwargs = {
        "json_mode": True,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if backend is not None:
        kwargs["backend"] = backend
    if model is not None:
        kwargs["model"] = model

    t0 = time.time()
    try:
        raw_response = llm_client.generate(
            prompt,
            timeout=LLM_SERVICE_TIMEOUT_SEC,
            **kwargs,
        )
    except Exception as exc:
        raise LLMCallError(str(exc)) from exc
    elapsed = time.time() - t0
    logger.debug("LLM response (%.1fs): %s", elapsed, raw_response[:200])
    if diagnostics is not None:
        diagnostics["raw_response_length"] = len(raw_response)
        diagnostics["raw_response_preview"] = raw_response[:800]

    # Parse JSON
    data = _parse_llm_json(raw_response)

    normed_text = data.get("normed_text")
    waves = data.get("waves")

    if not normed_text:
        raise ValueError("LLM response missing 'normed_text' field")
    if not isinstance(waves, dict):
        raise ValueError("LLM response missing 'waves' object")
    if diagnostics is not None:
        parsed_wave_ids = [str(key) for key in waves.keys()]
        required_wave_ids = [config["wave_id"] for config in wave_configs]
        diagnostics["parsed_wave_ids"] = parsed_wave_ids
        diagnostics["missing_wave_ids"] = [
            wave_id for wave_id in required_wave_ids if wave_id not in waves
        ]

    parsed_waves: Dict[str, Dict[str, Any]] = {}
    for config in wave_configs:
        wave_id = config["wave_id"]
        wave_data = waves.get(wave_id)
        if not isinstance(wave_data, dict):
            raise ValueError(f"LLM response missing wave {wave_id!r}")
        tagged_text = wave_data.get("tagged_text")
        if not tagged_text:
            raise ValueError(f"Wave {wave_id} missing 'tagged_text' field")
        parsed_wave = {"tagged_text": str(tagged_text)}
        if forward_nl_mode == "llm":
            parsed_wave["forward_nl"] = _validate_nl_payload(
                wave_data.get("forward_nl"),
                label=f"Wave {wave_id} forward_nl",
            )
        parsed_waves[wave_id] = parsed_wave

    return {
        "normed_text": str(normed_text),
        "waves": parsed_waves,
    }


def _call_llm_for_reverse_instructions(
    plain_text: str,
    tagged_texts: Dict[str, str],
    llm_client,
    *,
    language: Optional[str] = None,
    backend: Optional[str],
    model: Optional[str] = None,
    temperature: float,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    waves = [
        {
            "wave_id": str(wave_id),
            "tagged_text": tagged_text,
            "reverse_st": invert_structural_instruction(tagged_text),
        }
        for wave_id, tagged_text in tagged_texts.items()
    ]
    prompt = build_reverse_instruction_prompt(
        plain_text=plain_text,
        waves=waves,
        language=language,
    )
    if diagnostics is not None:
        diagnostics["reverse_prompt_length"] = len(prompt)

    kwargs = {
        "json_mode": True,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if backend is not None:
        kwargs["backend"] = backend
    if model is not None:
        kwargs["model"] = model

    try:
        raw_response = llm_client.generate(
            prompt,
            timeout=LLM_SERVICE_TIMEOUT_SEC,
            **kwargs,
        )
    except Exception as exc:
        raise LLMCallError(str(exc)) from exc
    if diagnostics is not None:
        diagnostics["reverse_raw_response_length"] = len(raw_response)
        diagnostics["reverse_raw_response_preview"] = raw_response[:800]

    data = _parse_llm_json(raw_response)
    reverse_instructions = data.get("reverse_instructions")
    if not isinstance(reverse_instructions, dict):
        raise ValueError("Reverse LLM response missing 'reverse_instructions' object")

    parsed: Dict[str, Dict[str, str]] = {}
    for wave_id in tagged_texts:
        item = reverse_instructions.get(str(wave_id))
        if not isinstance(item, dict):
            raise ValueError(f"Reverse LLM response missing wave {wave_id!r}")
        parsed[str(wave_id)] = _validate_nl_payload(
            item.get("nl"),
            label=f"Wave {wave_id} reverse nl",
        )
    return parsed
