from __future__ import annotations

"""Plan profile-controlled emotion edits from sampled text and speaker audio."""

import json
import logging
import math
import random
import re
import time
import html
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from speech_edit.data.language import classify_text_language

if TYPE_CHECKING:
    from speech_edit.clients.services import LLMClient
    from speech_edit.data.parquet.reader import Segment

from .prompts import (
    build_rewrite_prompt,
    get_prompt,
)
from .synthesis_profile import (
    EMOTION_CONDITION_DESC,
    EMOTION_CONDITION_TYPE_LEVEL,
    normalize_synthesis_profile,
    validate_tagged_texts_for_profile,
)
from .tag_utils import (
    EMOTION_CONTROL_ALPHA,
    EMOTION_CONTROL_DESCRIPTION,
    EMOTION_TAGS,
    EditPair,
    TaggedTextGroup,
    render_emo_tag,
    detect_boundaries,
    describe_tag_diff,
    get_all_pairs,
    parse_tags,
    strip_tags,
    texts_equivalent_ignoring_safe_whitespace,
    validate_boundaries_at_punctuation,
    validate_gradual_variation,
    validate_tagged_text,
)

logger = logging.getLogger(__name__)


def _resolve_rng(rng: Optional[Any] = None) -> Any:
    """Return a random-like object exposing random/choices/shuffle/randint."""
    return rng if rng is not None else random

# ---------------------------------------------------------------------------
# Probability of rewriting the generation text
TEXT_REWRITE_PROB = 0.0

# Bound the LLM response size for tagged text generation.
STAGE1_LLM_MAX_TOKENS = 4096

# Stage I text budget defaults.  English uses whitespace/punctuation-delimited
# words; Chinese uses punctuation-stripped CJK characters as text units.
MAX_STAGE1_TEXT_WORDS_EN = 120
MAX_STAGE1_TEXT_WORDS_ZH = 180

# Stage-I emotion-control defaults.  Each emotion sample draws one plan,
# then the LLM must stay inside that plan.
ALLOWED_EMOTION_COUNT = 5
EMOTION_EDIT_MODES = ("type_change", "intensity_change")
EMOTION_PROMPT_MODE_PROBS = {
    "level": 0.5,
    "description": 0.5,
}


@dataclass(frozen=True)
class EmotionEditPlan:
    """External random control for one emotion emotion edit sample."""

    allowed_emotions: List[str]
    edit_mode: str
    emotion_prompt_mode: str
    emotion_tag_weights: Dict[str, float]
    strict: bool = False
    region_count: Optional[int] = None
    emotion_matrix: Optional[Dict[str, List[str]]] = None
    change_masks: Optional[Dict[str, List[int]]] = None
    region_levels: Optional[List[int]] = None
    coverage_mode: Optional[str] = None

    def to_prompt_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "allowed_emotions": self.allowed_emotions,
            "edit_mode": self.edit_mode,
            "emotion_prompt_mode": self.emotion_prompt_mode,
        }
        if self.strict:
            result.update(
                {
                    "strict": True,
                    "region_count": self.region_count,
                    "emotion_matrix": self.emotion_matrix,
                    "change_masks": self.change_masks,
                    "region_levels": self.region_levels,
                    "coverage_mode": self.coverage_mode,
                }
            )
        return result

    def to_metadata_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "allowed_emotions": self.allowed_emotions,
            "edit_mode": self.edit_mode,
            "emotion_prompt_mode": self.emotion_prompt_mode,
            "emotion_tag_weights": self.emotion_tag_weights,
        }
        if self.strict:
            result.update(self.to_prompt_dict())
        return result


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------


def _strip_code_blocks(text: str) -> str:
    """Strip markdown code blocks from LLM output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _fix_malformed_closing_tags(text: str) -> str:
    """Fix LLM-generated malformed closing tags like </calm, level=1>.

    The closing tag should only have the tag name, not attributes.
    E.g., </calm, level=1> -> </calm>
    """
    return re.sub(r"</(\w+)[^>]*>", r"</\1>", text)


def _absorb_intertag_whitespace(text: str) -> Tuple[str, int]:
    """Move whitespace-only gaps between adjacent emo tags into the left tag."""
    repaired, count = re.subn(
        r"</emo>(\s+)(?=<emo(?:\s|>))",
        lambda match: f"{match.group(1)}</emo>",
        text,
    )
    return repaired, count


def parse_llm_json(text: str) -> dict:
    """Parse LLM JSON output with cascading error recovery."""
    text = _strip_code_blocks(text)
    # Fix malformed closing tags inside JSON string values
    text = _fix_malformed_closing_tags(text)

    fixes = [
        lambda t: t,
        lambda t: t + "}",
        lambda t: t + "]}",
        lambda t: t + "}]}",
        lambda t: t + '"]}',
        lambda t: t + '"}]}',
        lambda t: t + '"}',
        lambda t: t + '"}}}',
        lambda t: re.sub(r",\s*([}\]])", r"\1", t),  # trailing commas
    ]

    last_err = None
    for fix_fn in fixes:
        try:
            parsed = json.loads(fix_fn(text))
            # If result is a list, take the first element
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]
            return parsed
        except json.JSONDecodeError as e:
            last_err = e
            continue

    raise ValueError(
        f"Cannot parse LLM output as JSON after all fix attempts: {last_err}\n"
        f"Raw output (first 500 chars): {text[:500]}"
    )


def _prompt_with_validation_feedback(prompt: str, errors: Tuple[str, ...] | List[str]) -> str:
    if not errors:
        return prompt
    trimmed = []
    for error in list(errors)[-12:]:
        one_line = " ".join(str(error).split())
        trimmed.append(one_line[:500])
    bullet_list = "\n".join(f"- {error}" for error in trimmed)
    return f"""{prompt}

## Retry Feedback
The previous attempt failed validation. Regenerate the entire JSON object from scratch and fix every issue below.
Do not copy invalid emotion tags, invalid duration values, or invalid text from the failed attempt.

Validation errors:
{bullet_list}
""".strip()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def _normalize_plain_text(text: str) -> str:
    """Normalize plain text to avoid LLM stripping characters.

    - Normalize curly/smart quotes to straight quotes.
    - Strip surrounding matched quotation marks.
    - Strip orphan leading/trailing quotes.
    - Normalize whitespace.
    """
    text = text.strip()
    # Normalize curly/smart quotes to straight
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("«", '"').replace("»", '"')
    # Normalize dashes
    text = text.replace("—", "--").replace("–", "--")
    # Remove surrounding matched quotes
    quote_pairs = [('"', '"'), ("'", "'")]
    for open_q, close_q in quote_pairs:
        if text.startswith(open_q) and text.endswith(close_q) and len(text) > 2:
            text = text[len(open_q) : -len(close_q)].strip()
    # Strip orphan leading/trailing quotes that LLMs tend to drop
    text = text.strip("\"'")
    # Normalize internal whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_stage1_text_units(text: str, language: Optional[str]) -> int:
    """Count text units used by Stage 0 text-length budgeting."""
    normalized_language = (language or classify_text_language(text)).lower()
    if normalized_language == "zh":
        return len(re.findall(r"[\u4e00-\u9fff]", text))
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))


def build_stage1_text_unit_caps(
    *,
    max_stage1_text_words_en: Optional[int] = MAX_STAGE1_TEXT_WORDS_EN,
    max_stage1_text_words_zh: Optional[int] = MAX_STAGE1_TEXT_WORDS_ZH,
) -> Dict[str, int]:
    """Build language-keyed text unit caps for Stage 0 sampling."""
    caps: Dict[str, int] = {}
    if max_stage1_text_words_en is not None:
        caps["en"] = int(max_stage1_text_words_en)
    if max_stage1_text_words_zh is not None:
        caps["zh"] = int(max_stage1_text_words_zh)
    return caps


def _tokenize_for_similarity(text: str) -> List[str]:
    """Tokenize text coarsely for normalization drift checks."""
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\w\s]", text)


_ALLOWED_EXPANSION_SOURCE_TOKENS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "st",
    "jr",
    "sr",
}


def _extract_semantic_tokens(text: str) -> List[str]:
    """Extract ordered lexical tokens while ignoring punctuation and digits."""
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", text)
    normalized = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z]+", token):
            normalized.append(token.lower())
        else:
            normalized.append(token)
    return normalized


def _is_ordered_subsequence(required: List[str], actual: List[str]) -> bool:
    """Return True if required tokens appear in order within actual tokens."""
    idx = 0
    for token in actual:
        if idx < len(required) and token == required[idx]:
            idx += 1
    return idx == len(required)


def _is_reasonable_normed_text(raw_text: str, normed_text: str) -> bool:
    """Allow surface normalization while rejecting broad semantic rewriting."""
    raw_norm = _normalize_plain_text(raw_text)
    normed_norm = _normalize_plain_text(normed_text)
    if not normed_norm:
        return False
    if raw_norm == normed_norm:
        return True

    raw_tokens = _tokenize_for_similarity(raw_norm)
    normed_tokens = _tokenize_for_similarity(normed_norm)
    if not raw_tokens or not normed_tokens:
        return False

    length_ratio = len(normed_norm) / max(len(raw_norm), 1)
    if not 0.5 <= length_ratio <= 3.0:
        return False

    raw_semantic_tokens = _extract_semantic_tokens(raw_norm)
    normed_semantic_tokens = _extract_semantic_tokens(normed_norm)
    if not normed_semantic_tokens:
        return False

    required_tokens = [
        token
        for token in raw_semantic_tokens
        if token not in _ALLOWED_EXPANSION_SOURCE_TOKENS
    ]
    return _is_ordered_subsequence(required_tokens, normed_semantic_tokens)


# ---------------------------------------------------------------------------
# Utterance selection
# ---------------------------------------------------------------------------


def build_plain_text_from_segments(
    gen_segments: List[Segment],
    *,
    rng: Optional[Any] = None,
) -> str:
    """Build normalized Stage 1 plain text from sampled generation segments."""
    rng = _resolve_rng(rng)

    gen_texts = [seg.text for seg in gen_segments]
    rng.shuffle(gen_texts)
    concatenated_text = " ".join(gen_texts)
    return _normalize_plain_text(concatenated_text)


# ---------------------------------------------------------------------------
# Sub-type selection
# ---------------------------------------------------------------------------


def _weighted_sample_without_replacement(
    items: List[str],
    weights: List[float],
    k: int,
    *,
    rng: Optional[Any] = None,
) -> List[str]:
    """Sample ``k`` unique items according to weights."""
    rng = _resolve_rng(rng)
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    if k > len(items):
        raise ValueError(f"Cannot sample {k} unique items from {len(items)} items")
    if any(weight < 0 for weight in weights):
        raise ValueError("Weights must be non-negative")
    if sum(1 for weight in weights if weight > 0) < k:
        raise ValueError(f"At least {k} emotion weights must be positive")

    remaining_items = list(items)
    remaining_weights = list(weights)
    selected: List[str] = []
    for _ in range(k):
        choice = rng.choices(remaining_items, weights=remaining_weights, k=1)[0]
        idx = remaining_items.index(choice)
        selected.append(choice)
        del remaining_items[idx]
        del remaining_weights[idx]
    return selected


def _condition_from_emotion_prompt_mode(emotion_prompt_mode: str) -> str:
    if emotion_prompt_mode == "level":
        return EMOTION_CONDITION_TYPE_LEVEL
    if emotion_prompt_mode == "description":
        return EMOTION_CONDITION_DESC
    raise ValueError(f"Unknown emotion_prompt_mode: {emotion_prompt_mode}")


def normalize_emotion_edit_mode_probs(
    raw_weights: Optional[Mapping[str, Any]],
    *,
    emotion_condition_mode: str,
) -> Dict[str, float]:
    """Return edit-mode weights for one emotion condition mode.

    ``emotion_edit_mode_probs`` is intentionally condition-scoped so desc
    prompts can support intensity edits in code without sampling them by
    default.
    """
    if not isinstance(raw_weights, Mapping):
        raise TypeError("emotion_edit_mode_probs must be a mapping")

    legal_conditions = {EMOTION_CONDITION_TYPE_LEVEL, EMOTION_CONDITION_DESC}
    unknown = []
    for condition_key, value in raw_weights.items():
        if condition_key not in legal_conditions:
            unknown.append(str(condition_key))
            continue
        if not isinstance(value, Mapping):
            unknown.append(str(condition_key))
            continue
        for edit_key in value:
            if edit_key not in EMOTION_EDIT_MODES:
                unknown.append(f"{condition_key}.{edit_key}")
    if unknown:
        raise ValueError(
            "Unknown emotion_edit_mode_probs keys: " + ", ".join(sorted(unknown))
        )

    if emotion_condition_mode not in legal_conditions:
        raise ValueError(f"Unknown emotion_condition_mode: {emotion_condition_mode}")

    condition_weights = raw_weights.get(emotion_condition_mode, {})
    if not isinstance(condition_weights, Mapping):
        raise TypeError(
            f"emotion_edit_mode_probs.{emotion_condition_mode} must be a mapping"
        )

    weights: Dict[str, float] = {}
    for edit_mode in EMOTION_EDIT_MODES:
        value = float(condition_weights.get(edit_mode, 0.0))
        if value < 0:
            raise ValueError(
                f"emotion_edit_mode_probs.{emotion_condition_mode}.{edit_mode} "
                "must be non-negative"
            )
        weights[edit_mode] = value
    if sum(weights.values()) <= 0:
        raise ValueError(
            f"emotion_edit_mode_probs.{emotion_condition_mode} must contain "
            "non-negative weights with positive total"
        )
    return weights


def _normalize_emotion_target_weights(
    emotion_tag_weights: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    raw = emotion_tag_weights
    if not isinstance(raw, Mapping):
        raise TypeError("emotion_tag_weights must be a mapping")
    unknown = sorted(set(raw) - set(EMOTION_TAGS))
    if unknown:
        raise ValueError(f"Unknown emotion_tag_weights keys: {unknown}")
    weights = {emotion: float(raw.get(emotion, 0.0)) for emotion in EMOTION_TAGS}
    if any(value <= 0 for value in weights.values()):
        raise ValueError(
            "strict emotion planning requires every emotion_tag_weights value to be positive"
        )
    total = sum(weights.values())
    return {emotion: value / total for emotion, value in weights.items()}


def build_no_self_transition_matrix(
    emotion_tag_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Build a reversible no-self transition matrix with the requested stationary mix."""
    target = _normalize_emotion_target_weights(emotion_tag_weights)
    max_probability = max(target.values())
    if max_probability >= 0.5:
        raise ValueError(
            "A no-self transition matrix requires every target emotion probability "
            "to be below 0.5"
        )

    def residual(total_factor: float) -> float:
        return (
            sum(
                (
                    total_factor
                    - math.sqrt(total_factor * total_factor - 4.0 * probability)
                )
                / 2.0
                for probability in target.values()
            )
            - total_factor
        )

    lower = 2.0 * math.sqrt(max_probability) * (1.0 + 1e-12)
    upper = max(1.0, lower * 2.0)
    while residual(upper) > 0:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if residual(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    total_factor = (lower + upper) / 2.0
    factors = {
        emotion: (
            total_factor
            - math.sqrt(total_factor * total_factor - 4.0 * probability)
        )
        / 2.0
        for emotion, probability in target.items()
    }
    matrix: Dict[str, Dict[str, float]] = {}
    for source in EMOTION_TAGS:
        row = {
            target_emotion: (
                0.0
                if target_emotion == source
                else factors[source]
                * factors[target_emotion]
                / target[source]
            )
            for target_emotion in EMOTION_TAGS
        }
        row_total = sum(row.values())
        matrix[source] = {
            emotion: probability / row_total
            for emotion, probability in row.items()
        }
    return matrix


def _select_strict_region_count(
    sub_type: str,
    region_count_weights: Optional[Mapping[str, Any]],
    *,
    rng: Any,
) -> int:
    legal_counts = {1} if sub_type == "single" else {2, 3}
    if sub_type not in {"single", "multi_emo_normal", "multi_emo_frequent"}:
        raise ValueError(f"Strict emotion planning does not support sub_type={sub_type!r}")
    if not isinstance(region_count_weights, Mapping) or sub_type not in region_count_weights:
        raise ValueError(f"emotion_region_count_weights.{sub_type} must be configured")
    raw = region_count_weights[sub_type]
    if not isinstance(raw, Mapping):
        raise TypeError(f"emotion_region_count_weights.{sub_type} must be a mapping")
    normalized = {int(key): float(value) for key, value in raw.items()}
    if set(normalized) - legal_counts:
        raise ValueError(
            f"emotion_region_count_weights.{sub_type} supports only "
            f"{sorted(legal_counts)}"
        )
    counts = sorted(legal_counts)
    weights = [normalized.get(count, 0.0) for count in counts]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(
            f"emotion_region_count_weights.{sub_type} must have positive total weight"
        )
    return int(rng.choices(counts, weights=weights, k=1)[0])


def _strict_change_masks(
    region_count: int,
    *,
    require_full_coverage: bool,
    rng: Any,
) -> Dict[str, List[int]]:
    if region_count == 1:
        return {"1->2": [0], "2->3": [0]}
    candidates = [
        tuple(indices)
        for size in range(1, region_count)
        for indices in combinations(range(region_count), size)
    ]
    if require_full_coverage:
        pairs = [
            (first, second)
            for first in candidates
            for second in candidates
            if set(first) | set(second) == set(range(region_count))
        ]
        first, second = rng.choice(pairs)
    else:
        first = rng.choice(candidates)
        second = rng.choice(candidates)
    return {"1->2": list(first), "2->3": list(second)}


def _sample_strict_emotion_matrix(
    target_weights: Mapping[str, float],
    transition_matrix: Mapping[str, Mapping[str, float]],
    change_masks: Mapping[str, List[int]],
    region_count: int,
    *,
    rng: Any,
) -> Dict[str, List[str]]:
    emotions = list(EMOTION_TAGS)
    weights = [target_weights[emotion] for emotion in emotions]
    first = rng.choices(emotions, weights=weights, k=region_count)

    def transition(
        source: List[str],
        changed_regions: List[int],
    ) -> List[str]:
        result = list(source)
        for region in changed_regions:
            current = source[region]
            result[region] = rng.choices(
                emotions,
                weights=[transition_matrix[current][emotion] for emotion in emotions],
                k=1,
            )[0]
        return result

    second = transition(first, change_masks["1->2"])
    for _ in range(100):
        third = transition(second, change_masks["2->3"])
        if third != first:
            return {"1": first, "2": second, "3": third}
    raise RuntimeError("Could not sample a strict emotion chain with distinct variants 1 and 3")


def select_emotion_edit_plan(
    *,
    emotion_edit_mode_probs: Mapping[str, Any],
    emotion_prompt_mode_probs: Optional[Dict[str, float]] = None,
    emotion_tag_weights: Dict[str, float],
    strict: bool = False,
    sub_type: Optional[str] = None,
    emotion_region_count_weights: Optional[Mapping[str, Any]] = None,
    rng: Optional[Any] = None,
) -> EmotionEditPlan:
    """Draw one externally controlled emotion edit plan for a Stage-I sample."""
    rng = _resolve_rng(rng)

    if emotion_prompt_mode_probs is None:
        emotion_prompt_mode_probs = EMOTION_PROMPT_MODE_PROBS
    valid_prompt_modes = ["level", "description"]
    prompt_mode_weights = [
        float(emotion_prompt_mode_probs.get(mode, 0.0)) for mode in valid_prompt_modes
    ]
    if any(weight < 0 for weight in prompt_mode_weights) or sum(prompt_mode_weights) <= 0:
        raise ValueError(
            "emotion_prompt_mode_probs must contain non-negative weights with positive total"
        )
    emotion_prompt_mode = rng.choices(
        valid_prompt_modes, weights=prompt_mode_weights, k=1
    )[0]
    emotion_condition_mode = _condition_from_emotion_prompt_mode(emotion_prompt_mode)

    normalized_edit_weights = normalize_emotion_edit_mode_probs(
        emotion_edit_mode_probs,
        emotion_condition_mode=emotion_condition_mode,
    )
    valid_modes = list(EMOTION_EDIT_MODES)
    mode_weights = [normalized_edit_weights[mode] for mode in valid_modes]
    edit_mode = rng.choices(valid_modes, weights=mode_weights, k=1)[0]

    normalized_weights = {
        emotion: float(emotion_tag_weights.get(emotion, 0.0))
        for emotion in EMOTION_TAGS
    }
    if strict:
        if not sub_type:
            raise ValueError("sub_type is required for strict emotion planning")
        target_weights = _normalize_emotion_target_weights(emotion_tag_weights)
        region_count = _select_strict_region_count(
            sub_type,
            emotion_region_count_weights,
            rng=rng,
        )
        coverage_mode = "all" if sub_type == "multi_emo_normal" else "partial"
        change_masks = _strict_change_masks(
            region_count,
            require_full_coverage=coverage_mode == "all",
            rng=rng,
        )
        transition_matrix = build_no_self_transition_matrix(emotion_tag_weights)
        emotion_matrix = _sample_strict_emotion_matrix(
            target_weights,
            transition_matrix,
            change_masks,
            region_count,
            rng=rng,
        )
        return EmotionEditPlan(
            allowed_emotions=list(EMOTION_TAGS),
            edit_mode="type_change",
            emotion_prompt_mode=emotion_prompt_mode,
            emotion_tag_weights=target_weights,
            strict=True,
            region_count=region_count,
            emotion_matrix=emotion_matrix,
            change_masks=change_masks,
            region_levels=[int(rng.randint(1, 3)) for _ in range(region_count)],
            coverage_mode=coverage_mode,
        )

    allowed_emotions = _weighted_sample_without_replacement(
        list(EMOTION_TAGS),
        [normalized_weights[emotion] for emotion in EMOTION_TAGS],
        ALLOWED_EMOTION_COUNT,
        rng=rng,
    )

    return EmotionEditPlan(
        allowed_emotions=allowed_emotions,
        edit_mode=edit_mode,
        emotion_prompt_mode=emotion_prompt_mode,
        emotion_tag_weights=normalized_weights,
    )


def _edit_pair_has_tag_diff(
    tagged_texts: Dict[str, str],
    edit_pair: EditPair,
) -> bool:
    """Return whether a pair has a parseable tag-control delta."""
    source_text = tagged_texts.get(edit_pair.source_id)
    target_text = tagged_texts.get(edit_pair.target_id)
    if source_text is None or target_text is None:
        return False
    try:
        return bool(describe_tag_diff(source_text, target_text))
    except Exception:
        return False


def _is_intensity_diff_for_plan(diff: Dict[str, Any], emotion_edit_plan: EmotionEditPlan) -> bool:
    if emotion_edit_plan.emotion_prompt_mode == "level":
        return (
            diff.get("source_tag") == diff.get("target_tag")
            and diff.get("source_level") is not None
            and diff.get("target_level") is not None
            and diff.get("source_level") != diff.get("target_level")
        )
    if emotion_edit_plan.emotion_prompt_mode == "description":
        return (
            diff.get("source_emotion_type_ref") == diff.get("target_emotion_type_ref")
            and bool(diff.get("source_emotion_type_ref"))
            and diff.get("source_emotion_desc") != diff.get("target_emotion_desc")
        )
    raise ValueError(f"Unknown emotion_prompt_mode: {emotion_edit_plan.emotion_prompt_mode}")


def _diffs_match_emotion_plan(
    diffs: List[Dict[str, Any]],
    emotion_edit_plan: Optional[EmotionEditPlan],
) -> bool:
    if not diffs:
        return False
    if emotion_edit_plan is None:
        return True
    if emotion_edit_plan.edit_mode == "type_change":
        if not emotion_edit_plan.strict:
            return True
        if emotion_edit_plan.emotion_prompt_mode == "description":
            return all(
                bool(diff.get("source_emotion_type_ref"))
                and bool(diff.get("target_emotion_type_ref"))
                and diff.get("source_emotion_type_ref")
                != diff.get("target_emotion_type_ref")
                for diff in diffs
            )
        return all(
            bool(diff.get("source_tag"))
            and bool(diff.get("target_tag"))
            and diff.get("source_tag") != diff.get("target_tag")
            for diff in diffs
        )
    if emotion_edit_plan.edit_mode != "intensity_change":
        raise ValueError(f"Unknown emotion edit mode: {emotion_edit_plan.edit_mode}")
    return all(_is_intensity_diff_for_plan(diff, emotion_edit_plan) for diff in diffs)


def _sorted_tagged_text_keys(tagged_texts: Dict[str, str]) -> List[str]:
    """Return stable text ids, preserving numeric variant ordering when present."""
    return sorted(
        tagged_texts.keys(),
        key=lambda key: (0, int(key)) if str(key).isdigit() else (1, str(key)),
    )


def _build_edit_pairs_from_diff(
    tagged_texts: Dict[str, str],
    llm_result: Dict[str, Any],
    *,
    natural_language_instruction_mode: str,
    emotion_edit_plan: Optional[EmotionEditPlan] = None,
) -> List[EditPair]:
    """Build emotion edit pairs during Plan validation.

    Structured instructions are always derived from the local tag diff. The
    natural-language instruction can either come from the LLM or from the same
    deterministic template path.
    """
    mode = natural_language_instruction_mode.strip().lower()
    if mode not in {"llm", "template"}:
        raise ValueError(
            "natural_language_instruction_mode must be 'llm' or 'template', "
            f"got {natural_language_instruction_mode!r}"
        )

    st_inst_raw = llm_result.get("st_inst", {})
    if st_inst_raw is None:
        st_inst_raw = {}
    if not isinstance(st_inst_raw, dict):
        raise ValueError("st_inst must be a mapping when provided")
    desc_intensity_mode = (
        emotion_edit_plan is not None
        and emotion_edit_plan.edit_mode == "intensity_change"
        and emotion_edit_plan.emotion_prompt_mode == "description"
    )

    if mode == "llm":
        nl_inst_raw = llm_result.get("nl_inst", {})
        if not isinstance(nl_inst_raw, dict) or not nl_inst_raw:
            raise ValueError("No nl_inst in LLM response")
        candidate_pairs: List[Tuple[str, str, str]] = []
        for pair_key, nl_instruction in nl_inst_raw.items():
            parts = str(pair_key).split("->")
            if len(parts) != 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if src in tagged_texts and tgt in tagged_texts:
                instruction = str(nl_instruction or "").strip()
                if instruction:
                    candidate_pairs.append((src, tgt, instruction))
    elif desc_intensity_mode:
        if not st_inst_raw:
            raise ValueError("No st_inst in LLM response for desc intensity_change")
        candidate_pairs = []
        for pair_key in st_inst_raw:
            parts = str(pair_key).split("->")
            if len(parts) != 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if src in tagged_texts and tgt in tagged_texts:
                candidate_pairs.append((src, tgt, ""))
    else:
        keys = _sorted_tagged_text_keys(tagged_texts)
        candidate_pairs = [
            (src, tgt, "")
            for src, tgt in get_all_pairs(keys, bidirectional=True)
        ]

    edit_pairs: List[EditPair] = []
    for src, tgt, nl_instruction in candidate_pairs:
        source_tagged = tagged_texts[src]
        target_tagged = tagged_texts[tgt]
        try:
            diffs = describe_tag_diff(source_tagged, target_tagged)
            if not _diffs_match_emotion_plan(diffs, emotion_edit_plan):
                continue
            st_instruction = build_template_st_instruction(
                source_tagged,
                target_tagged,
                emotion_edit_plan=emotion_edit_plan,
            )
            if desc_intensity_mode:
                pair_key = f"{src}->{tgt}"
                st_instruction = _validate_desc_intensity_st_instruction(
                    st_inst_raw.get(pair_key, ""),
                    source_tagged=source_tagged,
                    pair_key=pair_key,
                )
            if mode == "template":
                nl_instruction = build_template_nl_instruction(
                    source_tagged,
                    target_tagged,
                )
        except Exception as exc:
            raise ValueError(
                f"Failed to build tag-diff instructions for pair {src}->{tgt}: {exc}"
            ) from exc
        edit_pairs.append(
            EditPair(
                source_id=src,
                target_id=tgt,
                nl_instruction=nl_instruction,
                st_instruction=st_instruction,
            )
        )

    if not edit_pairs:
        raise ValueError("No edit pairs contained a tag diff")
    return edit_pairs


def summarize_prompted_edit_modes(
    edit_pairs: List[EditPair],
    emotion_edit_plan: Optional[EmotionEditPlan],
) -> Dict[str, int]:
    """Return the sampled edit intent distribution, not a post-hoc tag classifier."""
    if emotion_edit_plan is None:
        return {}
    return {emotion_edit_plan.edit_mode: len(edit_pairs)}


def _validate_emotion_edit_plan_tags(
    tagged_texts: Dict[str, str],
    emotion_edit_plan: EmotionEditPlan,
) -> None:
    allowed = set(emotion_edit_plan.allowed_emotions)
    for text_id, tagged_text in tagged_texts.items():
        for segment in parse_tags(tagged_text):
            if segment.tag_name is None:
                continue
            if segment.tag_type not in {"emotion", "none"}:
                raise ValueError(
                    f"Tagged text {text_id} uses non-emotion tag {segment.tag_name!r}; "
                    "style tags are disabled for this pipeline"
                )
            if emotion_edit_plan.emotion_prompt_mode == "description":
                if segment.tag_name == "none":
                    continue
                if segment.emotion_control_mode != EMOTION_CONTROL_DESCRIPTION:
                    raise ValueError(
                        f"Tagged text {text_id} must use <emo desc=...> in description mode"
                    )
                if not segment.emotion_type_ref:
                    raise ValueError(
                        f"Tagged text {text_id} must include type_ref in description mode"
                    )
                if segment.emotion_type_ref not in allowed:
                    raise ValueError(
                        f"Tagged text {text_id} uses type_ref {segment.emotion_type_ref!r} "
                        f"outside allowed_emotions={emotion_edit_plan.allowed_emotions}"
                    )
                continue
            if emotion_edit_plan.emotion_prompt_mode != "level":
                raise ValueError(
                    f"Unknown emotion_prompt_mode: {emotion_edit_plan.emotion_prompt_mode}"
                )
            if segment.emotion_control_mode != EMOTION_CONTROL_ALPHA and segment.tag_name != "none":
                raise ValueError(
                    f"Tagged text {text_id} must use <emo type=... level=...> in level mode"
                )
            if segment.tag_name != "none" and segment.emotion_level is None:
                raise ValueError(
                    f"Tagged text {text_id} must use explicit level in level mode"
                )
            if segment.emotion_type not in allowed and segment.tag_name != "none":
                raise ValueError(
                    f"Tagged text {text_id} uses emotion {segment.emotion_type!r} "
                    f"outside allowed_emotions={emotion_edit_plan.allowed_emotions}"
                )
    if emotion_edit_plan.strict:
        _validate_strict_emotion_matrix(tagged_texts, emotion_edit_plan)


def _validate_strict_emotion_matrix(
    tagged_texts: Dict[str, str],
    emotion_edit_plan: EmotionEditPlan,
) -> None:
    expected_keys = {"1", "2", "3"}
    if set(tagged_texts) != expected_keys:
        raise ValueError(
            f"Strict emotion plan requires tagged_texts keys {sorted(expected_keys)}, "
            f"got {sorted(tagged_texts)}"
        )
    region_count = emotion_edit_plan.region_count
    matrix = emotion_edit_plan.emotion_matrix
    masks = emotion_edit_plan.change_masks
    levels = emotion_edit_plan.region_levels
    if not region_count or not matrix or not masks or not levels:
        raise ValueError("Strict emotion plan is missing its region matrix")

    parsed: Dict[str, List[Any]] = {}
    for text_id in ("1", "2", "3"):
        segments = [
            segment
            for segment in parse_tags(tagged_texts[text_id])
            if segment.tag_name is not None
        ]
        if len(segments) != region_count:
            raise ValueError(
                f"Tagged text {text_id} must contain exactly {region_count} regions; "
                f"got {len(segments)}"
            )
        expected_emotions = matrix[text_id]
        for region, (segment, expected_emotion) in enumerate(
            zip(segments, expected_emotions)
        ):
            if segment.evaluation_tag != expected_emotion:
                raise ValueError(
                    f"Tagged text {text_id} region {region + 1} must use "
                    f"{expected_emotion!r}, got {segment.evaluation_tag!r}"
                )
            if emotion_edit_plan.emotion_prompt_mode == "level":
                if segment.emotion_level != levels[region]:
                    raise ValueError(
                        f"Tagged text {text_id} region {region + 1} must keep "
                        f"level={levels[region]}"
                    )
            elif not segment.emotion_desc:
                raise ValueError(
                    f"Tagged text {text_id} region {region + 1} must include desc"
                )
            else:
                _validate_strict_emotion_description(
                    segment.emotion_desc,
                    expected_emotion=expected_emotion,
                    text_id=text_id,
                    region=region,
                )
        parsed[text_id] = segments

    reference_segments = parsed["1"]
    for text_id in ("2", "3"):
        for region, (reference, candidate) in enumerate(
            zip(reference_segments, parsed[text_id])
        ):
            if not texts_equivalent_ignoring_safe_whitespace(
                reference.text, candidate.text
            ):
                raise ValueError(
                    f"Tagged text {text_id} changes region {region + 1} text or boundary"
                )

    for source_id, target_id, mask_key in (
        ("1", "2", "1->2"),
        ("2", "3", "2->3"),
    ):
        actual_changed = {
            region
            for region in range(region_count)
            if matrix[source_id][region] != matrix[target_id][region]
        }
        expected_changed = set(masks[mask_key])
        if actual_changed != expected_changed:
            raise ValueError(
                f"Emotion matrix {mask_key} changes regions "
                f"{sorted(actual_changed)}, expected {sorted(expected_changed)}"
            )
        if region_count > 1 and len(actual_changed) >= region_count:
            raise ValueError(f"Strict multi-region transition {mask_key} must leave a region unchanged")
        if emotion_edit_plan.emotion_prompt_mode == "description":
            for region in set(range(region_count)) - expected_changed:
                if (
                    parsed[source_id][region].emotion_desc
                    != parsed[target_id][region].emotion_desc
                ):
                    raise ValueError(
                        f"Unchanged region {region + 1} must keep the same desc in {mask_key}"
                    )

    if emotion_edit_plan.emotion_prompt_mode == "description":
        for region in range(region_count):
            for source_id, target_id in combinations(("1", "2", "3"), 2):
                same_emotion = (
                    matrix[source_id][region] == matrix[target_id][region]
                )
                same_description = (
                    parsed[source_id][region].emotion_desc
                    == parsed[target_id][region].emotion_desc
                )
                if same_emotion != same_description:
                    relationship = (
                        "reuse the same desc" if same_emotion else "use different desc values"
                    )
                    raise ValueError(
                        f"Region {region + 1} variants {source_id}/{target_id} must "
                        f"{relationship}"
                    )

    if emotion_edit_plan.coverage_mode == "all":
        covered = set(masks["1->2"]) | set(masks["2->3"])
        if covered != set(range(region_count)):
            raise ValueError(
                "multi_emo_normal must change every region at least once across the chain"
            )


def _validate_strict_emotion_description(
    description: str,
    *,
    expected_emotion: str,
    text_id: str,
    region: int,
) -> None:
    normalized = " ".join(str(description).strip().lower().split())
    words = re.findall(r"[a-z]+(?:['-][a-z]+)*", normalized)
    cjk_chars = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", normalized)
    wrapper_words = {
        expected_emotion.lower(),
        "a",
        "an",
        "the",
        "very",
        "slightly",
        "deeply",
        "softly",
        "tone",
        "voice",
        "vocal",
        "speaking",
        "speech",
        "sound",
        "manner",
        "style",
        "emotion",
        "delivery",
    }
    chinese_labels = {
        "happy": ("开心", "快乐", "高兴", "喜悦"),
        "angry": ("生气", "愤怒"),
        "sad": ("悲伤", "伤心", "哀伤"),
        "afraid": ("害怕", "恐惧"),
        "disgusted": ("厌恶", "嫌弃", "反感"),
        "melancholic": ("忧郁", "惆怅", "伤感"),
        "surprised": ("惊讶", "吃惊", "惊奇"),
        "calm": ("平静", "冷静", "镇定"),
    }
    compact_cjk = "".join(cjk_chars)
    chinese_label_wrappers = {
        f"{prefix}{label}{suffix}"
        for label in chinese_labels.get(expected_emotion.lower(), ())
        for prefix in ("", "很", "非常", "稍微", "略带", "带有")
        for suffix in (
            "",
            "语气",
            "声音",
            "嗓音",
            "情绪",
            "风格",
            "的语气",
            "的声音",
            "的嗓音",
            "的情绪",
            "的风格",
        )
    }
    if normalized == expected_emotion.lower() or (
        words and set(words).issubset(wrapper_words)
    ) or compact_cjk in chinese_label_wrappers:
        raise ValueError(
            f"Tagged text {text_id} region {region + 1} desc must be a natural "
            f"vocal-delivery phrase, not a label wrapper: {description!r}"
        )
    if len(words) < 3 and len(cjk_chars) < 4:
        raise ValueError(
            f"Tagged text {text_id} region {region + 1} desc is too short; "
            "use roughly 3-8 English words or 4-16 Chinese characters"
        )


def _filter_edit_pairs_by_emotion_plan(
    tagged_texts: Dict[str, str],
    edit_pairs: List[EditPair],
    emotion_edit_plan: EmotionEditPlan,
) -> List[EditPair]:
    if emotion_edit_plan.emotion_prompt_mode not in {"description", "level"}:
        raise ValueError(
            f"Unknown emotion_prompt_mode: {emotion_edit_plan.emotion_prompt_mode}"
        )
    if emotion_edit_plan.edit_mode not in EMOTION_EDIT_MODES:
        raise ValueError(f"Unknown emotion edit mode: {emotion_edit_plan.edit_mode}")

    filtered = [
        edit_pair
        for edit_pair in edit_pairs
        if _edit_pair_has_tag_diff(tagged_texts, edit_pair)
    ]
    if not filtered:
        raise ValueError(
            f"No edit pairs contained a tag diff for prompted edit mode "
            f"{emotion_edit_plan.edit_mode!r}"
        )
    return filtered


# ---------------------------------------------------------------------------
# Text rewriting
# ---------------------------------------------------------------------------


def maybe_rewrite_text(
    text: str,
    llm_client: LLMClient,
    rewrite_prob: float = TEXT_REWRITE_PROB,
    llm_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
    *,
    rng: Optional[Any] = None,
) -> Tuple[str, bool]:
    """Possibly rewrite text via LLM, keeping style and topic.

    Returns:
        (text, was_rewritten)
    """
    rng = _resolve_rng(rng)
    if rng.random() >= rewrite_prob:
        return text, False

    prompt = build_rewrite_prompt(text)
    rewritten = llm_client.generate(
        prompt,
        backend=llm_backend,
        model=llm_model,
        temperature=0.9,
        max_tokens=len(text) * 3,  # generous but bounded
    )
    rewritten = rewritten.strip()

    if not rewritten or len(rewritten) > len(text):
        raise ValueError("Rewritten text must be nonempty and no longer than the input")

    logger.info(f"Text rewritten: {text[:50]}... -> {rewritten[:50]}...")
    return rewritten, True


# ---------------------------------------------------------------------------
# LLM generation + validation
# ---------------------------------------------------------------------------


def _generate_tagged_texts(
    plain_text: str,
    sub_type: str,
    llm_client: LLMClient,
    llm_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
    *,
    emotion_edit_plan: Optional[EmotionEditPlan] = None,
    synthesis_profile: Optional[Dict[str, Any]] = None,
    natural_language_instruction_mode: str = "llm",
    validation_errors: Tuple[str, ...] | List[str] = (),
    diagnostics: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, str], List[EditPair], str]:
    """Generate tagged texts and instructions from the sampled profile.

    The LLM decides tag boundaries for all sub-types. No pre-segmentation
    is needed.

    Returns:
        (tagged_texts_dict, edit_pairs_list, canonical_text)
    """
    prompt = get_prompt(
        sub_type,
        plain_text,
        emotion_edit_plan=(
            emotion_edit_plan.to_prompt_dict()
            if emotion_edit_plan is not None
            else None
        ),
        natural_language_instruction_mode=natural_language_instruction_mode,
    )
    feedback_errors = validation_errors
    if emotion_edit_plan is not None and emotion_edit_plan.strict:
        feedback_errors = list(validation_errors)[-1:]
    prompt = _prompt_with_validation_feedback(prompt, feedback_errors)
    if diagnostics is not None:
        diagnostics["llm_prompt"] = prompt

    max_tokens = STAGE1_LLM_MAX_TOKENS

    generate_start = time.time()
    try:
        raw = llm_client.generate(
            prompt,
            backend=llm_backend,
            model=llm_model,
            temperature=0.8,
            max_tokens=max_tokens,
            json_mode=True,
        )
    finally:
        if diagnostics is not None:
            diagnostics["llm_generate_sec"] = diagnostics.get(
                "llm_generate_sec", 0.0
            ) + (time.time() - generate_start)
    if diagnostics is not None:
        diagnostics["llm_raw_response"] = raw

    result = parse_llm_json(raw)

    # Extract tagged texts first. For we no longer require the LLM's
    # internal normalization to match the raw input text. The generated data only
    # needs one canonical spoken-form text that all tagged variants share.
    tagged_texts = result.get("tagged_texts", {})
    if not tagged_texts:
        raise ValueError("No tagged_texts in LLM response")

    fixed_tagged_texts: Dict[str, str] = {}
    stripped_by_key: Dict[str, str] = {}
    for key, tt in list(tagged_texts.items()):
        tt = _fix_malformed_closing_tags(tt)
        tt, whitespace_repair_count = _absorb_intertag_whitespace(tt)
        if diagnostics is not None and whitespace_repair_count:
            diagnostics["intertag_whitespace_repairs"] = int(
                diagnostics.get("intertag_whitespace_repairs", 0)
            ) + whitespace_repair_count
        fixed_tagged_texts[key] = tt

        stripped = strip_tags(tt)
        if not _normalize_plain_text(stripped):
            raise ValueError(f"Tagged text {key} is empty after stripping tags")
        stripped_by_key[key] = stripped

    canonical_candidates = list(dict.fromkeys(stripped_by_key.values()))
    canonical_text: Optional[str] = None
    first_canonical_candidate = canonical_candidates[0]
    if all(
        re.sub(r"\s+", "", first_canonical_candidate)
        == re.sub(r"\s+", "", candidate)
        and texts_equivalent_ignoring_safe_whitespace(
            first_canonical_candidate, candidate
        )
        for candidate in canonical_candidates[1:]
    ):
        canonical_text = first_canonical_candidate
        tagged_texts = fixed_tagged_texts
    elif sub_type == "single":
        repair_start = time.time()
        repaired_tagged_texts, repaired_canonical_text = (
            _repair_single_tagged_texts_by_majority(fixed_tagged_texts)
        )
        repair_elapsed = time.time() - repair_start
        if diagnostics is not None:
            diagnostics["repair_sec"] = diagnostics.get("repair_sec", 0.0) + repair_elapsed
        if repaired_tagged_texts is None or repaired_canonical_text is None:
            raise ValueError(
                "tagged texts are not internally consistent after stripping tags: "
                f"stripped_texts={stripped_by_key}"
            )
        tagged_texts = repaired_tagged_texts
        canonical_text = repaired_canonical_text
        if diagnostics is not None:
            diagnostics["single_repair_applied"] = True
    else:
        raise ValueError(
            "tagged texts are not internally consistent after stripping tags: "
            f"stripped_texts={stripped_by_key}"
        )

    assert canonical_text is not None
    if (
        emotion_edit_plan is not None
        and emotion_edit_plan.strict
        and not texts_equivalent_ignoring_safe_whitespace(plain_text, canonical_text)
    ):
        raise ValueError(
            "Strict tagged texts must preserve the input text; whitespace-only "
            "differences are allowed except removed English word boundaries"
        )

    llm_canonical_text = result.get("canonical_text")
    if llm_canonical_text is None:
        llm_canonical_text = result.get("normed_text", "")
    if isinstance(llm_canonical_text, str) and llm_canonical_text.strip():
        llm_canonical_text = _normalize_plain_text(llm_canonical_text)
        if llm_canonical_text != canonical_text:
            logger.info(
                "canonical_text differs from canonical tagged text; using canonical text"
            )

    # Validate each tagged text against the canonical text derived from the
    # tagged outputs themselves.
    for key, tt in tagged_texts.items():
        valid, err = validate_tagged_text(tt, canonical_text)
        if not valid:
            raise ValueError(f"Tagged text {key} invalid: {err}")

    if emotion_edit_plan is not None:
        _validate_emotion_edit_plan_tags(tagged_texts, emotion_edit_plan)
    if synthesis_profile is not None:
        validate_tagged_texts_for_profile(
            tagged_texts,
            synthesis_profile,
            require_explicit_index_level=True,
        )

    # Validate gradual variation for multi-tag types
    if (
        sub_type != "single"
        and len(tagged_texts) >= 2
        and not (emotion_edit_plan is not None and emotion_edit_plan.strict)
    ):
        valid, err = validate_gradual_variation(tagged_texts)
        if not valid:
            logger.warning(f"Gradual variation check failed: {err}")
            # Don't fail — still usable data, just not perfectly gradual

    # Validate boundary positions for normal frequency (skip for frequent)
    if sub_type == "multi_emo_normal":
        for key, tt in tagged_texts.items():
            valid, err = validate_boundaries_at_punctuation(tt, canonical_text)
            if not valid:
                logger.warning(f"Boundary check failed for text {key}: {err}")

    # For multi_emo_frequent: validate that we got exactly 3 tagged texts
    if (
        sub_type == "multi_emo_frequent"
        and len(tagged_texts) != 3
        and not (emotion_edit_plan is not None and emotion_edit_plan.strict)
    ):
        logger.warning(
            f"multi_emo_frequent expected 3 tagged texts, got {len(tagged_texts)}"
        )

    edit_pairs = _build_edit_pairs_from_diff(
        tagged_texts,
        result,
        natural_language_instruction_mode=natural_language_instruction_mode,
        emotion_edit_plan=emotion_edit_plan,
    )

    if emotion_edit_plan is not None:
        edit_pairs = _filter_edit_pairs_by_emotion_plan(
            tagged_texts,
            edit_pairs,
            emotion_edit_plan,
        )

    return tagged_texts, edit_pairs, canonical_text


def _repair_single_tagged_texts_by_majority(
    tagged_texts: Dict[str, str],
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Repair single-tag variants only when at least 2/3 agree on plain text."""
    surface_to_keys: Dict[str, List[str]] = {}
    rebuilt_segments = {}

    for key, tagged_text in tagged_texts.items():
        segments = parse_tags(tagged_text)
        tagged_segments = [segment for segment in segments if segment.tag_name is not None]
        untagged_segments = [segment for segment in segments if segment.tag_name is None]
        if len(tagged_segments) != 1:
            return None, None
        if any(segment.text.strip() for segment in untagged_segments):
            return None, None

        tagged_segment = tagged_segments[0]
        surface_text = tagged_segment.text
        if not _normalize_plain_text(surface_text):
            return None, None
        surface_to_keys.setdefault(surface_text, []).append(key)
        rebuilt_segments[key] = tagged_segment

    canonical_text, keys = max(
        surface_to_keys.items(), key=lambda item: len(item[1])
    )
    if len(keys) <= len(tagged_texts) / 2:
        return None, None

    repaired = {}
    for key, tagged_segment in rebuilt_segments.items():
        if tagged_segment.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
            repaired[key] = render_emo_tag(
                text=canonical_text,
                emotion_desc=tagged_segment.emotion_desc,
                emotion_type_ref=tagged_segment.emotion_type_ref,
                dur=tagged_segment.dur,
            )
        else:
            repaired[key] = render_emo_tag(
                text=canonical_text,
                emotion_type=tagged_segment.emotion_type or tagged_segment.tag_name,
                level=tagged_segment.emotion_level,
                alpha=tagged_segment.alpha if tagged_segment.emotion_level is None else None,
                dur=tagged_segment.dur,
            )
    return repaired, canonical_text


def _tag_label(
    tag: object,
    level: object,
    alpha: object,
    emotion_desc: object = None,
    dur: object = None,
) -> str:
    if emotion_desc:
        label = f"emotion_desc={emotion_desc}"
        if dur is not None:
            label += f" dur={dur}s"
        return label
    label = str(tag or "none")
    if level is not None:
        label += f" level={level}"
    elif alpha is not None:
        label += f" alpha={alpha}"
    if dur is not None:
        label += f" dur={dur}s"
    return label


def build_template_nl_instruction(source_tagged: str, target_tagged: str) -> str:
    """Build a natural-language instruction from the tag delta."""
    diffs = describe_tag_diff(source_tagged, target_tagged)
    if not diffs:
        return "Keep the emotion and style unchanged."

    clauses = []
    for diff in diffs:
        text = str(diff.get("segment_text", "")).strip()
        source = _tag_label(
            diff.get("source_tag"),
            diff.get("source_level"),
            diff.get("source_alpha"),
            diff.get("source_emotion_desc"),
            diff.get("source_dur"),
        )
        target = _tag_label(
            diff.get("target_tag"),
            diff.get("target_level"),
            diff.get("target_alpha"),
            diff.get("target_emotion_desc"),
            diff.get("target_dur"),
        )
        if text:
            clauses.append(f'change "{text}" from {source} to {target}')
        else:
            clauses.append(f"change one segment from {source} to {target}")
    if len(clauses) == 1:
        return f"Please {clauses[0]}."
    return "Please apply these emotion edits: " + "; ".join(clauses) + "."


def _escape_attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def _render_intensity_delta_tag(*, text: str, emotion_type: str, action: str) -> str:
    if action not in {"inc", "dec"}:
        raise ValueError(f"Unknown intensity action: {action}")
    return (
        f'<emo type="{_escape_attr(emotion_type)}" level="{action}">'
        f"{text}</emo>"
    )


def _validate_desc_intensity_st_instruction(
    st_instruction: object,
    *,
    source_tagged: str,
    pair_key: str,
) -> str:
    instruction = str(st_instruction or "").strip()
    if not instruction:
        raise ValueError(f"Missing st_inst for desc intensity pair {pair_key}")
    source_text = strip_tags(source_tagged)
    if not texts_equivalent_ignoring_safe_whitespace(
        source_text, strip_tags(instruction)
    ):
        raise ValueError(
            f"st_inst for desc intensity pair {pair_key} must preserve canonical text"
        )
    changed_segments = [
        segment
        for segment in parse_tags(instruction)
        if segment.tag_name is not None
    ]
    if not changed_segments:
        raise ValueError(
            f"st_inst for desc intensity pair {pair_key} must mark at least one span"
        )
    for segment in changed_segments:
        if segment.emotion_control_mode != EMOTION_CONTROL_DESCRIPTION:
            raise ValueError(
                f"st_inst for desc intensity pair {pair_key} must use <emo desc=...>"
            )
        if segment.emotion_type_ref:
            raise ValueError(
                f"st_inst for desc intensity pair {pair_key} must not include type_ref"
            )
    return instruction


def build_template_st_instruction(
    source_tagged: str,
    target_tagged: str,
    *,
    emotion_edit_plan: Optional[EmotionEditPlan] = None,
) -> str:
    """Build WDTW-compatible structured markup for changed emotion spans."""
    diffs = describe_tag_diff(source_tagged, target_tagged)
    diff_by_index = {
        int(diff["segment_idx"]): diff
        for diff in diffs
        if isinstance(diff, dict) and diff.get("segment_idx") is not None
    }
    changed_indexes = {
        index for index in diff_by_index
    }
    target_segments = [seg for seg in parse_tags(target_tagged) if seg.tag_name is not None]
    if not changed_indexes:
        return strip_tags(target_tagged)

    parts: List[str] = []
    for index, segment in enumerate(target_segments):
        text = html.escape(segment.text, quote=False)
        if index in changed_indexes:
            if segment.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
                parts.append(
                    render_emo_tag(
                        text=text,
                        emotion_desc=segment.emotion_desc,
                        dur=segment.dur,
                    )
                )
            elif (
                emotion_edit_plan is not None
                and emotion_edit_plan.edit_mode == "intensity_change"
                and emotion_edit_plan.emotion_prompt_mode == "level"
            ):
                diff = diff_by_index[index]
                source_level = int(diff["source_level"])
                target_level = int(diff["target_level"])
                action = "inc" if target_level > source_level else "dec"
                parts.append(
                    _render_intensity_delta_tag(
                        text=text,
                        emotion_type=str(segment.emotion_type or segment.tag_name),
                        action=action,
                    )
                )
            else:
                parts.append(
                    render_emo_tag(
                        text=text,
                        emotion_type=segment.emotion_type or segment.tag_name,
                        level=segment.emotion_level,
                        alpha=(
                            segment.effective_alpha
                            if segment.emotion_level is None
                            else None
                        ),
                        dur=segment.dur,
                    )
                )
        else:
            parts.append(text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Plan-stage entry point
# ---------------------------------------------------------------------------


def generate_tagged_text_group_attempt(
    speaker_prompt: Segment,
    gen_segments: List[Segment],
    language: str,
    llm_client: LLMClient,
    *,
    synthesis_profile: Dict[str, Any],
    stage1_seed: int,
    natural_language_instruction_mode: str,
    rewrite_prob: float,
    emotion_edit_mode_probs: Mapping[str, Any],
    emotion_tag_weights: Dict[str, float],
    strict_emotion_plan: bool,
    emotion_region_count_weights: Mapping[str, Any],
    max_stage1_text_words_en: int,
    max_stage1_text_words_zh: int,
    attempt: int = 1,
    validation_errors: Tuple[str, ...] | List[str] = (),
    llm_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> TaggedTextGroup:
    """Generate one profile-controlled group; the plan stage owns retries."""
    rng = random.Random(stage1_seed)
    normalized_profile = normalize_synthesis_profile(synthesis_profile)
    sub_type = normalized_profile["synthesis_mode"]
    emotion_prompt_mode_probs = {
        "description": float(normalized_profile["emotion_condition_mode"] == EMOTION_CONDITION_DESC),
        "level": float(normalized_profile["emotion_condition_mode"] == EMOTION_CONDITION_TYPE_LEVEL),
    }
    natural_language_instruction_mode = natural_language_instruction_mode.strip().lower()
    if natural_language_instruction_mode not in {"llm", "template"}:
        raise ValueError(
            "natural_language_instruction_mode must be 'llm' or 'template', "
            f"got {natural_language_instruction_mode!r}"
        )
    stage1_diagnostics: Dict[str, object] = {
        "rewrite_sec": 0.0,
        "llm_generate_sec": 0.0,
        "postprocess_sec": 0.0,
        "repair_sec": 0.0,
        "generation_attempts": 0,
        "last_generation_error": None,
        "single_repair_applied": False,
        "intertag_whitespace_repairs": 0,
    }

    selected_window = normalized_profile.get("selected_text_window")
    if isinstance(selected_window, dict) and selected_window.get("text"):
        plain_text = _normalize_plain_text(str(selected_window["text"]))
    else:
        plain_text = build_plain_text_from_segments(gen_segments, rng=rng)

    # Remember original plain text before potential rewriting
    original_plain_text = plain_text

    # Maybe rewrite text
    rewrite_start = time.time()
    plain_text, was_rewritten = maybe_rewrite_text(
        plain_text,
        llm_client,
        rewrite_prob=rewrite_prob,
        llm_backend=llm_backend,
        llm_model=llm_model,
        rng=rng,
    )
    stage1_diagnostics["rewrite_sec"] = time.time() - rewrite_start

    gen_duration = sum(seg.duration for seg in gen_segments)
    emotion_edit_plan = select_emotion_edit_plan(
        emotion_edit_mode_probs=emotion_edit_mode_probs,
        emotion_prompt_mode_probs=emotion_prompt_mode_probs,
        emotion_tag_weights=emotion_tag_weights,
        strict=strict_emotion_plan,
        sub_type=sub_type,
        emotion_region_count_weights=emotion_region_count_weights,
        rng=rng,
    )

    stage1_diagnostics["generation_attempts"] = attempt
    attempt_start = time.time()
    try:
        tagged_texts, edit_pairs, normed_text = _generate_tagged_texts(
            plain_text, sub_type, llm_client, llm_backend, llm_model,
            emotion_edit_plan=emotion_edit_plan,
            synthesis_profile=normalized_profile,
            natural_language_instruction_mode=natural_language_instruction_mode,
            validation_errors=validation_errors,
            diagnostics=stage1_diagnostics,
        )
    finally:
        stage1_diagnostics["postprocess_sec"] = max(
            time.time() - attempt_start
            - float(stage1_diagnostics["llm_generate_sec"])
            - float(stage1_diagnostics["repair_sec"]),
            0.0,
        )

    # Detect boundaries on the final normalized text.
    boundary_start = time.time()
    boundaries = detect_boundaries(normed_text, mode="punctuation")
    stage1_diagnostics["postprocess_sec"] = float(
        stage1_diagnostics["postprocess_sec"]
    ) + (time.time() - boundary_start)

    prompted_edit_mode = (
        emotion_edit_plan.edit_mode if emotion_edit_plan is not None else None
    )
    pair_change_summary = summarize_prompted_edit_modes(edit_pairs, emotion_edit_plan)
    llm_prompt = stage1_diagnostics.get("llm_prompt")
    llm_raw_response = stage1_diagnostics.get("llm_raw_response")
    profile_with_prompt = dict(normalized_profile)
    if llm_prompt is not None:
        profile_with_prompt["llm_prompt"] = llm_prompt
    if llm_raw_response is not None:
        profile_with_prompt["llm_raw_response"] = llm_raw_response
    stage1_text_units = count_stage1_text_units(normed_text, language)
    stage1_text_caps = build_stage1_text_unit_caps(
        max_stage1_text_words_en=max_stage1_text_words_en,
        max_stage1_text_words_zh=max_stage1_text_words_zh,
    )
    stage1_text_cap = stage1_text_caps.get(language)
    return TaggedTextGroup(
        sub_type=sub_type,
        plain_text=normed_text,
        speaker_prompt_audio=speaker_prompt.audio_bytes,
        speaker_prompt_text=speaker_prompt.text,
        tagged_texts=tagged_texts,
        edit_pairs=edit_pairs,
        boundaries=boundaries,
        metadata={
            "language": language,
            "was_rewritten": was_rewritten,
            "original_plain_text": original_plain_text,
            "rewritten_plain_text": plain_text,
            "stage1_diagnostics": {
                key: (round(value, 3) if isinstance(value, float) else value)
                for key, value in stage1_diagnostics.items()
                if key not in {"llm_prompt", "llm_raw_response"}
            },
            "llm_prompt": llm_prompt,
            "gen_duration_sec": round(gen_duration, 2),
            "stage1_text_units": stage1_text_units,
            "stage1_text_cap": stage1_text_cap,
            "stage1_text_cap_language": language,
            "num_boundaries": len(boundaries),
            "num_tagged_texts": len(tagged_texts),
            "num_edit_pairs": len(edit_pairs),
            "emotion_edit_plan": (
                emotion_edit_plan.to_metadata_dict()
                if emotion_edit_plan is not None
                else None
            ),
            "emotion_condition_mode": profile_with_prompt.get("emotion_condition_mode"),
            "ted_estimated_duration_soft_limit_sec": profile_with_prompt.get("ted_estimated_duration_soft_limit_sec"),
            "ted_max_generated_duration_sec": profile_with_prompt.get("ted_max_generated_duration_sec"),
            "ted_max_mel_tokens": profile_with_prompt.get("ted_max_mel_tokens"),
            "pair_change_summary": pair_change_summary,
            "prompted_edit_mode": prompted_edit_mode,
            "natural_language_instruction_mode": natural_language_instruction_mode,
            "speaker_prompt_clip_id": speaker_prompt.clip_id,
            "speaker_prompt_metadata": dict(speaker_prompt.metadata or {}),
            "gen_segment_clip_ids": [s.clip_id for s in gen_segments],
            "synthesis_profile": profile_with_prompt,
            "synthesis_backend": profile_with_prompt.get("synthesis_backend"),
            "synthesis_mode": profile_with_prompt.get("synthesis_mode"),
        },
    )
