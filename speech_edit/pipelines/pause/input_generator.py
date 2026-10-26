"""
Stage I: Input Generation for pause editing pipeline.

Orchestrates:
1. Feasible pause-target selection
2. LLM-based pause edit profile generation
3. Validation and output
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple


if TYPE_CHECKING:
    from speech_edit.clients.services import LLMClient

from .prompts import get_prompt
from .tag_utils import (
    PauseEditProfile,
    build_pause_nl_instructions,
    parse_pause_tags,
    pause_boundaries,
    reverse_operation,
    validate_pause_tags,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------


# Fixed pause-level mapping exposed to the LLM prompt.
DURATION_POOL_SIZE = 3
PAUSE_LEVEL_DURATIONS_MS = [200, 500, 800]


class PauseProfileValidationError(ValueError):
    """Raised when all generated pause profiles fail validation."""

    def __init__(self, message: str, *, validation_errors: Optional[List[str]] = None):
        self.validation_errors = list(validation_errors or [])
        if self.validation_errors:
            detail_text = "; ".join(self.validation_errors[:12])
            message = f"{message} Validation details: {detail_text}"
        super().__init__(message)


@dataclass(frozen=True)
class PauseProfileTarget:
    profile_id: int
    boundary_types: List[str]
    levels: List[int]

    @property
    def num_operations(self) -> int:
        return len(self.levels)

    def to_prompt_dict(self) -> Dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "num_operations": self.num_operations,
            "boundary_types": list(self.boundary_types),
            "levels": list(self.levels),
        }


def _coerce_weight_key(raw_key: object, allowed_keys: set[object]) -> object:
    if raw_key in allowed_keys:
        return raw_key
    text = str(raw_key)
    for allowed in allowed_keys:
        if text == str(allowed):
            return allowed
    raise ValueError(f"Unsupported weight key {raw_key!r}; allowed keys: {sorted(allowed_keys, key=str)}")


def _normalize_weights(
    raw_weights: Optional[Dict],
    allowed_keys: set,
) -> Dict:
    source = raw_weights
    weights: Dict = {}
    for raw_key, raw_value in dict(source).items():
        key = _coerce_weight_key(raw_key, allowed_keys)
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Weight for {key!r} must be finite and non-negative, got {value}")
        weights[key] = value
    if not any(value > 0 for value in weights.values()):
        raise ValueError(f"At least one weight must be positive: {source}")
    return weights


def _weighted_choice(weights: Dict, rng: random.Random):
    total = sum(float(value) for value in weights.values())
    threshold = rng.random() * total
    cumulative = 0.0
    last_key = None
    for key, value in weights.items():
        if value <= 0:
            continue
        last_key = key
        cumulative += float(value)
        if threshold < cumulative:
            return key
    return last_key


def sample_pause_profile_targets(
    *,
    plain_text: str,
    num_profiles: int,
    num_operation_weights: Optional[Dict] = None,
    boundary_type_weights: Optional[Dict] = None,
    pause_level_weights: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> List[PauseProfileTarget]:
    """Condition the configured target distribution on feasible text boundaries."""
    if num_profiles < 1:
        raise ValueError("num_profiles must be >= 1")
    rng = random.Random(seed)
    operation_weights = _normalize_weights(
        num_operation_weights,
        {1, 2, 3},
    )
    boundary_weights = _normalize_weights(
        boundary_type_weights,
        {"punct", "nonpunct"},
    )
    level_weights = _normalize_weights(
        pause_level_weights,
        {1, 2, 3},
    )

    available_types = [kind for _, kind in pause_boundaries(plain_text)]
    boundary_total = sum(boundary_weights.values())
    feasible_weights = {}
    for count, operation_weight in operation_weights.items():
        if operation_weight <= 0:
            continue
        for sequence in product(boundary_weights, repeat=int(count)):
            # Greedy subsequence matching uses each physical boundary once.
            positions = iter(available_types)
            if not all(any(actual == wanted for actual in positions) for wanted in sequence):
                continue
            weight = operation_weight * math.prod(
                boundary_weights[kind] / boundary_total for kind in sequence
            )
            if weight > 0:
                feasible_weights[sequence] = weight
    if not feasible_weights:
        raise ValueError("No positive-weight feasible pause target for the supplied text")

    targets: List[PauseProfileTarget] = []
    for profile_id in range(1, num_profiles + 1):
        boundary_types = list(_weighted_choice(feasible_weights, rng))
        num_operations = len(boundary_types)
        levels = [
            int(_weighted_choice(level_weights, rng))
            for _ in range(num_operations)
        ]
        targets.append(
            PauseProfileTarget(
                profile_id=profile_id,
                boundary_types=boundary_types,
                levels=levels,
            )
        )
    return targets


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


def parse_llm_json(text: str) -> dict:
    """Parse LLM JSON output with cascading error recovery."""
    text = _strip_code_blocks(text)

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
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')
    # Normalize dashes
    text = text.replace("\u2014", "--").replace("\u2013", "--")
    # Remove surrounding matched quotes
    quote_pairs = [('"', '"'), ("'", "'")]
    for open_q, close_q in quote_pairs:
        if text.startswith(open_q) and text.endswith(close_q) and len(text) > 2:
            text = text[len(open_q) : -len(close_q)].strip()
    # Strip orphan leading/trailing quotes
    text = text.strip("\"'")
    # Normalize internal whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_semantic_tokens(text: str) -> List[str]:
    """Extract ordered lexical tokens while ignoring punctuation and digits."""
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", text)
    return [t.lower() if t.isascii() else t for t in tokens]


def _is_ordered_subsequence(required: List[str], actual: List[str]) -> bool:
    """Return True if required tokens appear in order within actual tokens."""
    idx = 0
    for token in actual:
        if idx < len(required) and token == required[idx]:
            idx += 1
    return idx == len(required)


_ALLOWED_EXPANSION_SOURCE_TOKENS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr",
}


def _is_reasonable_normed_text(raw_text: str, normed_text: str) -> bool:
    """Allow surface normalization while rejecting broad semantic rewriting."""
    raw_norm = _normalize_plain_text(raw_text)
    normed_norm = _normalize_plain_text(normed_text)
    if not normed_norm:
        return False

    # Allow surface normalization of the raw transcript before tagging.
    from .tag_utils import _normalize_for_comparison
    if _normalize_for_comparison(raw_norm) == _normalize_for_comparison(normed_norm):
        return True

    length_ratio = len(normed_norm) / max(len(raw_norm), 1)
    if not 0.5 <= length_ratio <= 3.0:
        return False

    raw_semantic_tokens = _extract_semantic_tokens(raw_norm)
    normed_semantic_tokens = _extract_semantic_tokens(normed_norm)
    if not normed_semantic_tokens:
        return False

    required_tokens = [
        t for t in raw_semantic_tokens
        if t not in _ALLOWED_EXPANSION_SOURCE_TOKENS
    ]
    return _is_ordered_subsequence(required_tokens, normed_semantic_tokens)


# ---------------------------------------------------------------------------
# Duration pool sampling
# ---------------------------------------------------------------------------


def sample_duration_pool(n: int = DURATION_POOL_SIZE) -> List[int]:
    """Return the fixed pause-level mapping in milliseconds.

    The current strategy no longer asks the LLM to invent precise pause
    durations. Instead it chooses one of three discrete levels that map to
    stable durations exposed here.
    """
    if n < 1:
        return []
    return PAUSE_LEVEL_DURATIONS_MS[:n]


# ---------------------------------------------------------------------------
# Utterance selection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LLM generation + validation
# ---------------------------------------------------------------------------


def _validation_errors_from_exception(exc: BaseException) -> List[str]:
    raw = getattr(exc, "validation_errors", None)
    if raw:
        return [str(item) for item in raw]
    return [f"{type(exc).__name__}: {exc}"]


def _dedupe_validation_errors(validation_errors: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for item in validation_errors:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _validate_profile_with_reason(
    profile_data: dict,
    normed_text: str,
    duration_pool: List[int],
    max_operations: int,
    *,
    profile_target: Optional[PauseProfileTarget] = None,
) -> tuple[Optional[PauseEditProfile], Optional[str]]:
    profile_id = profile_data.get("profile_id")
    operation = str(profile_data.get("operation", "") or "")
    if not operation:
        return None, f"Profile {profile_id}: empty operation"
    if profile_target is not None:
        try:
            actual_profile_id = int(profile_id)
        except (TypeError, ValueError):
            return None, f"Profile {profile_id}: profile_id must be {profile_target.profile_id}"
        if actual_profile_id != profile_target.profile_id:
            return None, (
                f"Profile {profile_id}: expected profile_id "
                f"{profile_target.profile_id}"
            )

    valid, err = validate_pause_tags(operation, normed_text)
    if not valid:
        return None, f"Profile {profile_id}: tag validation failed: {err}"

    tags = parse_pause_tags(operation, normed_text)
    if profile_target is None and not 1 <= len(tags) <= max_operations:
        return None, (
            f"Profile {profile_id}: expected between 1 and {max_operations} "
            f"pause insertion tags, got {len(tags)}"
        )
    if profile_target is not None and len(tags) != profile_target.num_operations:
        return None, (
            f"Profile {profile_id}: expected exactly "
            f"{profile_target.num_operations} pause insertion tags, got {len(tags)}"
        )
    valid_durations = set(duration_pool)
    for tag in tags:
        if tag.delta_dur_ms not in valid_durations:
            return None, (
                f"Profile {profile_id}: unsupported mapped duration {tag.delta_dur_ms}ms "
                f"(allowed: {sorted(valid_durations)})"
            )

    if profile_target is not None:
        actual_levels = [tag.level for tag in tags]
        if actual_levels != profile_target.levels:
            return None, (
                f"Profile {profile_id}: expected pause levels "
                f"{profile_target.levels}, got {actual_levels}"
            )
        actual_boundary_types = [
            "punct" if tag.punctuation else "nonpunct"
            for tag in tags
        ]
        if actual_boundary_types != profile_target.boundary_types:
            return None, (
                f"Profile {profile_id}: expected boundary types "
                f"{profile_target.boundary_types}, got {actual_boundary_types}"
            )

    st_inst = {
        "increase_pause": [operation],
        "decrease_pause": [reverse_operation(operation)],
    }
    nl_inst = {
        direction: build_pause_nl_instructions(instructions[0])
        for direction, instructions in st_inst.items()
    }

    profile = PauseEditProfile(
        profile_id=profile_id,
        operation=operation,
        st_inst=st_inst,
        nl_inst=nl_inst,
        tags=tags,
    )
    return profile, None


def _validate_profile(
    profile_data: dict,
    normed_text: str,
    duration_pool: List[int],
    max_operations: int,
    *,
    profile_target: Optional[PauseProfileTarget] = None,
) -> Optional[PauseEditProfile]:
    """Validate and parse a single profile from LLM output.

    Args:
        profile_data: Raw profile dict from LLM JSON.
        normed_text: The normalized text.
        duration_pool: Fixed level-to-duration mapping for this call.
        max_operations: Maximum allowed pause insertions for this profile.

    Returns:
        PauseEditProfile if valid, None if invalid.
    """
    profile, reason = _validate_profile_with_reason(
        profile_data,
        normed_text,
        duration_pool,
        max_operations,
        profile_target=profile_target,
    )
    if profile is None and reason is not None:
        logger.warning(reason)
    return profile


def _generate_profiles(
    plain_text: str,
    duration_pool: List[int],
    profile_targets: List[PauseProfileTarget],
    llm_client: LLMClient,
    llm_backend: str = "gemini",
    llm_model: Optional[str] = None,
    *,
    validation_errors: Optional[Iterable[str]] = None,
) -> Tuple[List[PauseEditProfile], str]:
    """Generate pause edit profiles via LLM.

    Returns:
        (profiles_list, normed_text)
    """
    prompt = get_prompt(
        plain_text=plain_text,
        duration_pool=duration_pool,
        profile_targets=[target.to_prompt_dict() for target in profile_targets],
        validation_errors=validation_errors,
    )

    raw = llm_client.generate(
        prompt,
        backend=llm_backend,
        model=llm_model,
        temperature=0.8,
        max_tokens=4096,
        json_mode=True,
    )

    result = parse_llm_json(raw)

    # Extract normed_text
    normed_text = result.get("normed_text")
    if not isinstance(normed_text, str) or not normed_text.strip():
        raise PauseProfileValidationError("LLM response requires nonempty normed_text")
    normed_text = normed_text.strip()
    if not _is_reasonable_normed_text(plain_text, normed_text):
        raise PauseProfileValidationError("normed_text changes the input's lexical content")

    # Extract profiles
    profiles_raw = result.get("profiles", [])
    if not profiles_raw:
        raise PauseProfileValidationError(
            "No profiles in LLM response",
            validation_errors=["LLM response missing non-empty profiles array"],
        )

    profiles = []
    profile_errors: List[str] = []

    raw_by_profile_id: Dict[int, dict] = {}
    duplicate_ids: List[int] = []
    for prof_data in profiles_raw:
        if not isinstance(prof_data, dict):
            profile_errors.append(f"Invalid profile entry: {prof_data!r}")
            continue
        try:
            profile_id = int(prof_data.get("profile_id"))
        except (TypeError, ValueError):
            profile_errors.append(f"Invalid profile_id: {prof_data.get('profile_id')!r}")
            continue
        if profile_id in raw_by_profile_id:
            duplicate_ids.append(profile_id)
            continue
        raw_by_profile_id[profile_id] = prof_data
    for duplicate_id in duplicate_ids:
        profile_errors.append(f"Duplicate profile_id {duplicate_id}")

    max_operations = max(target.num_operations for target in profile_targets)
    for target in profile_targets:
        prof_data = raw_by_profile_id.get(target.profile_id)
        if prof_data is None:
            profile_errors.append(f"Missing profile_id {target.profile_id}")
            continue
        profile, error = _validate_profile_with_reason(
            prof_data,
            normed_text,
            duration_pool,
            max_operations,
            profile_target=target,
        )
        if profile is not None:
            profiles.append(profile)
        elif error is not None:
            profile_errors.append(error)

    if len(profiles) != len(profile_targets):
        raise PauseProfileValidationError(
            "No valid profiles after validation",
            validation_errors=_dedupe_validation_errors(profile_errors),
        )

    return profiles, normed_text


def generate_profiles_with_retries(
    *,
    plain_text: str,
    duration_pool: List[int],
    profile_targets: List[PauseProfileTarget],
    llm_client: LLMClient,
    llm_backend: str = "gemini",
    llm_model: Optional[str] = None,
    max_retries: int,
) -> Tuple[List[PauseEditProfile], str]:
    profiles = None
    normed_text = plain_text
    last_error: Exception | None = None
    validation_errors: List[str] = []

    for attempt in range(1, max_retries + 1):
        try:
            profiles, normed_text = _generate_profiles(
                plain_text=plain_text,
                duration_pool=duration_pool,
                profile_targets=profile_targets,
                llm_client=llm_client,
                llm_backend=llm_backend,
                llm_model=llm_model,
                validation_errors=_dedupe_validation_errors(validation_errors),
            )
            return profiles, normed_text
        except Exception as exc:
            last_error = exc
            validation_errors.extend(_validation_errors_from_exception(exc))
            logger.warning(
                "Generation attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )

    message = f"Failed to generate after {max_retries} attempts. Last error: {last_error}"
    deduped_errors = _dedupe_validation_errors(validation_errors)
    if deduped_errors:
        raise PauseProfileValidationError(
            message,
            validation_errors=deduped_errors,
        ) from last_error
    raise RuntimeError(message) from last_error


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
