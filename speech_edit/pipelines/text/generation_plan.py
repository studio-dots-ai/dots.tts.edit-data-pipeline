"""Generation-level sampling for text edit pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


VALID_SPAN_LEVELS = ("word", "phrase", "clause", "sentence", "arbitrary")
VALID_OP_COUNTS = (1, 2, 3)

@dataclass(frozen=True)
class GenerationSpec:
    generation_id: str
    span_level: str
    op_count: int
    require_del_ins: bool = False

    @property
    def max_operations(self) -> int:
        if self.span_level == "sentence":
            return 1
        if self.span_level == "clause":
            return 2
        return 3

    def to_prompt_dict(self) -> Dict[str, Any]:
        instruction: str
        payload = {
            "generation_id": self.generation_id,
            "span_level": self.span_level,
            "op_count": self.op_count,
            "allowed_tags": ["sub", "del", "ins"],
        }
        if self.span_level == "sentence":
            payload["required_kind"] = "sub"
            instruction = (
                "Use exactly one <sub> edit and put the full canonical source text "
                "inside the sub tag."
            )
        elif self.span_level == "arbitrary":
            instruction = (
                "Use the requested number of non-overlapping edits. Each edit may "
                "target a word, phrase, or clause span."
            )
        else:
            instruction = (
                f"Use exactly {self.op_count} non-overlapping edits at {self.span_level} granularity."
            )
        if self.require_del_ins:
            payload["require_del_ins"] = True
            instruction += " Include at least one <ins> or <del> edit."
        payload["instruction"] = instruction
        return payload

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "span_level": self.span_level,
            "op_count": self.op_count,
            "max_operations": self.max_operations,
            "require_del_ins": self.require_del_ins,
        }


def _normalize_weights(
    weights: Optional[Mapping[Any, Any]],
    *,
    valid_keys: tuple[Any, ...],
    field_name: str,
) -> Dict[Any, float]:
    raw = dict(weights)
    unknown = set(raw) - set(valid_keys)
    if unknown:
        raise ValueError(f"Unknown {field_name}: {sorted(unknown)}")
    normalized: Dict[Any, float] = {}
    for key in valid_keys:
        value = float(raw.get(key, 0.0))
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative")
        normalized[key] = value
    if not any(value > 0 for value in normalized.values()):
        raise ValueError(f"{field_name} must contain at least one positive weight")
    return normalized


def parse_span_level_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("span level weights must be a JSON object")
    return {str(key): float(value) for key, value in data.items()}


def parse_op_count_weights(raw: Optional[str]) -> Optional[Dict[int, float]]:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("op count weights must be a JSON object")
    return {int(key): float(value) for key, value in data.items()}


def select_span_level(
    weights: Optional[Mapping[str, float]] = None,
    *,
    rng: Optional[random.Random] = None,
) -> str:
    normalized = _normalize_weights(
        weights,
        valid_keys=VALID_SPAN_LEVELS,
        field_name="span_level_weights",
    )
    levels = [level for level in VALID_SPAN_LEVELS if normalized[level] > 0]
    values = [normalized[level] for level in levels]
    chooser = rng or random
    return chooser.choices(levels, weights=values, k=1)[0]


def max_op_count_for_span_level(span_level: str) -> int:
    if span_level not in VALID_SPAN_LEVELS:
        raise ValueError(f"Unknown span_level: {span_level!r}")
    if span_level == "sentence":
        return 1
    if span_level == "clause":
        return 2
    return 3


def select_op_count(
    span_level: str,
    weights: Optional[Mapping[int, float]] = None,
    *,
    rng: Optional[random.Random] = None,
) -> int:
    if span_level == "sentence":
        return 1
    normalized = _normalize_weights(
        weights,
        valid_keys=VALID_OP_COUNTS,
        field_name="op_count_weights",
    )
    max_count = max_op_count_for_span_level(span_level)
    for count in VALID_OP_COUNTS:
        if count > max_count:
            normalized[count] = 0.0
    counts = [count for count in VALID_OP_COUNTS if normalized[count] > 0]
    values = [normalized[count] for count in counts]
    if not counts:
        raise ValueError(f"No legal op_count weight remains for span_level={span_level!r}")
    chooser = rng or random
    return chooser.choices(counts, weights=values, k=1)[0]


def select_generation_specs(
    *,
    num_generations: int,
    span_level_weights: Optional[Mapping[str, float]] = None,
    op_count_weights: Optional[Mapping[int, float]] = None,
    require_del_ins_prob: float,
    rng: Optional[random.Random] = None,
) -> List[GenerationSpec]:
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1")
    if require_del_ins_prob < 0.0 or require_del_ins_prob > 1.0:
        raise ValueError("require_del_ins_prob must be between 0 and 1")
    specs: List[GenerationSpec] = []
    chooser = rng or random
    for index in range(num_generations):
        span_level = select_span_level(span_level_weights, rng=rng)
        op_count = select_op_count(span_level, op_count_weights, rng=rng)
        require_del_ins = (
            index == 0
            and span_level != "sentence"
            and chooser.random() < require_del_ins_prob
        )
        specs.append(
            GenerationSpec(
                generation_id=f"{index:03d}",
                span_level=span_level,
                op_count=op_count,
                require_del_ins=require_del_ins,
            )
        )
    return specs
