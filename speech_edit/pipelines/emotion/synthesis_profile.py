"""Synthesis profile helpers for emotion/style edit samples."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .tag_utils import EMOTION_CONTROL_ALPHA, EMOTION_CONTROL_DESCRIPTION, parse_tags

SYNTHESIS_BACKEND_INDEX = "index"
SYNTHESIS_BACKEND_TED = "ted"
SYNTHESIS_BACKENDS = {SYNTHESIS_BACKEND_INDEX, SYNTHESIS_BACKEND_TED}

SYNTHESIS_MODE_SINGLE = "single"
SYNTHESIS_MODE_MULTI_NORMAL = "multi_emo_normal"
SYNTHESIS_MODE_MULTI_FREQUENT = "multi_emo_frequent"
SYNTHESIS_MODES = {
    SYNTHESIS_MODE_SINGLE,
    SYNTHESIS_MODE_MULTI_NORMAL,
    SYNTHESIS_MODE_MULTI_FREQUENT,
}

EMOTION_CONDITION_TYPE_LEVEL = "type_level"
EMOTION_CONDITION_DESC = "desc"
EMOTION_CONDITION_MODES = {
    EMOTION_CONDITION_TYPE_LEVEL,
    EMOTION_CONDITION_DESC,
}

EMOTION_CONTROL_BY_CONDITION = {
    EMOTION_CONDITION_TYPE_LEVEL: EMOTION_CONTROL_ALPHA,
    EMOTION_CONDITION_DESC: EMOTION_CONTROL_DESCRIPTION,
}

TED_MEL_TOKEN_SECONDS = 0.02
TED_TAGGED_DURATION_MAX_SEC = 17.0  # Maximum duration supported by tagged-duration synthesis.
TED_TAGGED_DURATION_MAX_MEL_TOKENS = int(
    round(TED_TAGGED_DURATION_MAX_SEC / TED_MEL_TOKEN_SECONDS)
)

SYNTHESIS_PROFILE_ORDER = (
    (SYNTHESIS_MODE_SINGLE, EMOTION_CONDITION_TYPE_LEVEL),
    (SYNTHESIS_MODE_SINGLE, EMOTION_CONDITION_DESC),
    (SYNTHESIS_MODE_MULTI_NORMAL, EMOTION_CONDITION_TYPE_LEVEL),
    (SYNTHESIS_MODE_MULTI_NORMAL, EMOTION_CONDITION_DESC),
    (SYNTHESIS_MODE_MULTI_FREQUENT, EMOTION_CONDITION_DESC),
)

_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SENTENCE_RE = re.compile(r"[^.!?;。！？；]+[.!?;。！？；]*")


@dataclass(frozen=True)
class TextUnit:
    text: str
    source_segment_index: int
    unit_index: int
    clip_id: Optional[str]
    estimated_duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextWindow:
    start_unit: int
    end_unit: int
    text: str
    estimated_duration_sec: float
    units: Tuple[TextUnit, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_unit": self.start_unit,
            "end_unit": self.end_unit,
            "text": self.text,
            "estimated_duration_sec": round(self.estimated_duration_sec, 3),
            "unit_texts": [unit.text for unit in self.units],
            "source_segment_indexes": [
                unit.source_segment_index for unit in self.units
            ],
            "source_clip_ids": [unit.clip_id for unit in self.units],
        }


def normalize_synthesis_backend(value: Optional[str], *, allow_infer: bool = False) -> str:
    backend = (value or "infer").strip().lower()
    if allow_infer and backend in {"", "infer", "auto"}:
        return "infer"
    if backend in SYNTHESIS_BACKENDS:
        return backend
    raise ValueError(
        f"synthesis_backend must be one of {sorted(SYNTHESIS_BACKENDS)}"
        + (", infer" if allow_infer else "")
        + f"; got {value!r}"
    )


def normalize_synthesis_mode(value: Optional[str], *, allow_infer: bool = False) -> str:
    mode = (value or "infer").strip().lower()
    if allow_infer and mode in {"", "infer", "auto"}:
        return "infer"
    if mode in {"multi_normal", "normal"}:
        return SYNTHESIS_MODE_MULTI_NORMAL
    if mode in {"multi_frequent", "frequent"}:
        return SYNTHESIS_MODE_MULTI_FREQUENT
    if mode in SYNTHESIS_MODES:
        return mode
    raise ValueError(
        f"synthesis_mode must be one of {sorted(SYNTHESIS_MODES)}"
        + (", infer" if allow_infer else "")
        + f"; got {value!r}"
    )


def normalize_emotion_condition_mode(
    value: Optional[str],
    *,
    allow_infer: bool = False,
) -> str:
    condition = (value or "infer").strip().lower()
    if allow_infer and condition in {"", "infer", "auto"}:
        return "infer"
    if condition in EMOTION_CONDITION_MODES:
        return condition
    raise ValueError(
        "emotion_condition_mode must be one of "
        f"{sorted(EMOTION_CONDITION_MODES)}"
        + (", infer" if allow_infer else "")
        + f"; got {value!r}"
    )


def resolve_backend_for_profile(
    synthesis_mode: str,
    emotion_condition_mode: str,
) -> str:
    mode = normalize_synthesis_mode(synthesis_mode)
    condition = normalize_emotion_condition_mode(emotion_condition_mode)
    if (
        mode == SYNTHESIS_MODE_MULTI_FREQUENT
        and condition == EMOTION_CONDITION_TYPE_LEVEL
    ):
        raise ValueError("multi_emo_frequent + type_level is not a legal profile")
    if mode == SYNTHESIS_MODE_MULTI_FREQUENT:
        return SYNTHESIS_BACKEND_TED
    return SYNTHESIS_BACKEND_INDEX


def profile_uses_ted_soft_limit(
    synthesis_mode: str,
    emotion_condition_mode: str,
) -> bool:
    return (
        resolve_backend_for_profile(synthesis_mode, emotion_condition_mode)
        == SYNTHESIS_BACKEND_TED
    )


def estimate_text_duration_sec(
    text: str,
    *,
    en_word_sec: float,
    cjk_char_sec: float,
) -> float:
    """Estimate spoken duration using additive Latin-word and CJK-char counts."""
    latin_words = len(_LATIN_WORD_RE.findall(text or ""))
    cjk_chars = len(_CJK_CHAR_RE.findall(text or ""))
    return latin_words * float(en_word_sec) + cjk_chars * float(cjk_char_sec)


def ted_max_mel_tokens_from_duration_sec(duration_sec: float) -> int:
    """Convert the user-facing TED hard generation cap in seconds to mel tokens."""
    return max(1, int(round(float(duration_sec) / TED_MEL_TOKEN_SECONDS)))


def _normalize_language(language: Optional[str], *, text: str = "") -> str:
    raw = str(language or "").strip().lower().replace("_", "-")
    if raw in {"zh", "zho", "chi", "cmn", "cn", "chinese", "mandarin", "zh-cn"}:
        return "zh"
    if raw in {"en", "eng", "english", "en-us", "en-gb"}:
        return "en"
    return "zh" if _CJK_CHAR_RE.search(text or "") else "en"


def count_profile_text_units(text: str, language: Optional[str]) -> int:
    """Count units for profile-window text caps.

    Chinese/code-switching windows count both CJK characters and Latin words so
    mixed text cannot bypass the cap by using the non-primary script.
    """
    normalized = _normalize_language(language, text=text)
    latin_words = len(_LATIN_WORD_RE.findall(text or ""))
    cjk_chars = len(_CJK_CHAR_RE.findall(text or ""))
    if normalized == "zh":
        return cjk_chars + latin_words
    return latin_words + cjk_chars


def _text_unit_cap_for_language(
    language: Optional[str],
    *,
    plan_text_unit_cap_en: Optional[int],
    plan_text_unit_cap_zh: Optional[int],
) -> Optional[int]:
    normalized = _normalize_language(language)
    if normalized == "zh":
        return int(plan_text_unit_cap_zh) if plan_text_unit_cap_zh is not None else None
    return int(plan_text_unit_cap_en) if plan_text_unit_cap_en is not None else None


def split_text_units_ordered(
    segments: Sequence[Any],
    *,
    duration_estimation_en_word_sec: float,
    duration_estimation_cjk_char_sec: float,
) -> List[TextUnit]:
    """Split sampled segments into stable, ordered sentence-like text units."""
    units: List[TextUnit] = []
    for segment_index, segment in enumerate(segments):
        raw_text = str(getattr(segment, "text", "") or "")
        pieces = [m.group(0).strip() for m in _SENTENCE_RE.finditer(raw_text)]
        pieces = [piece for piece in pieces if piece]
        if not pieces and raw_text.strip():
            pieces = [raw_text.strip()]
        for unit_index, text in enumerate(pieces):
            units.append(
                TextUnit(
                    text=text,
                    source_segment_index=segment_index,
                    unit_index=unit_index,
                    clip_id=getattr(segment, "clip_id", None),
                    estimated_duration_sec=estimate_text_duration_sec(
                        text,
                        en_word_sec=duration_estimation_en_word_sec,
                        cjk_char_sec=duration_estimation_cjk_char_sec,
                    ),
                )
            )
    return units


def enumerate_contiguous_text_windows(units: Sequence[TextUnit]) -> List[TextWindow]:
    windows: List[TextWindow] = []
    for start in range(len(units)):
        for end in range(start + 1, len(units) + 1):
            window_units = tuple(units[start:end])
            text = " ".join(unit.text for unit in window_units).strip()
            if not text:
                continue
            estimated = sum(unit.estimated_duration_sec for unit in window_units)
            windows.append(
                TextWindow(
                    start_unit=start,
                    end_unit=end,
                    text=text,
                    estimated_duration_sec=estimated,
                    units=window_units,
                )
            )
    return windows


def legal_synthesis_tuples_for_duration(
    estimated_duration_sec: float,
    *,
    ted_estimated_duration_soft_limit_sec: float,
) -> List[Tuple[str, str]]:
    legal: List[Tuple[str, str]] = []
    for mode, condition in SYNTHESIS_PROFILE_ORDER:
        if profile_uses_ted_soft_limit(mode, condition) and estimated_duration_sec > float(
            ted_estimated_duration_soft_limit_sec
        ):
            continue
        legal.append((mode, condition))
    return legal


def build_synthesis_profile(
    *,
    synthesis_mode: str,
    emotion_condition_mode: str,
    selected_text_window: Optional[Mapping[str, Any]] = None,
    estimated_duration_sec: Optional[float] = None,
    ted_estimated_duration_soft_limit_sec: float,
    ted_max_generated_duration_sec: float,
) -> Dict[str, Any]:
    mode = normalize_synthesis_mode(synthesis_mode)
    condition = normalize_emotion_condition_mode(emotion_condition_mode)
    backend = resolve_backend_for_profile(mode, condition)
    profile: Dict[str, Any] = {
        "synthesis_backend": backend,
        "synthesis_mode": mode,
        "emotion_condition_mode": condition,
        "emotion_control_mode": EMOTION_CONTROL_BY_CONDITION[condition],
        "ted_estimated_duration_soft_limit_sec": round(
            float(ted_estimated_duration_soft_limit_sec), 3
        ),
        "ted_max_generated_duration_sec": round(
            float(ted_max_generated_duration_sec), 3
        ),
        "ted_max_mel_tokens": ted_max_mel_tokens_from_duration_sec(
            ted_max_generated_duration_sec
        ),
    }
    if selected_text_window is not None:
        profile["selected_text_window"] = dict(selected_text_window)
    if estimated_duration_sec is not None:
        profile["estimated_duration_sec"] = round(float(estimated_duration_sec), 3)
    return profile


def normalize_synthesis_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    selected_text_window = profile.get("selected_text_window")
    estimated = profile.get("estimated_duration_sec")
    if estimated is None and isinstance(selected_text_window, Mapping):
        estimated = selected_text_window.get("estimated_duration_sec")
    normalized = build_synthesis_profile(
        synthesis_mode=str(profile.get("synthesis_mode") or ""),
        emotion_condition_mode=str(profile.get("emotion_condition_mode") or ""),
        selected_text_window=(
            dict(selected_text_window)
            if isinstance(selected_text_window, Mapping)
            else None
        ),
        estimated_duration_sec=float(estimated) if estimated is not None else None,
        ted_estimated_duration_soft_limit_sec=float(
            profile["ted_estimated_duration_soft_limit_sec"]
        ),
        ted_max_generated_duration_sec=float(
            profile["ted_max_generated_duration_sec"]
        ),
    )
    for debug_key in ("llm_prompt", "llm_raw_response"):
        if debug_key in profile:
            normalized[debug_key] = profile.get(debug_key)
    return normalized


def _profile_weight_value(
    raw_weights: Mapping[str, Any],
    mode: str,
    condition: str,
) -> Optional[float]:
    flat_key = f"{mode}.{condition}"
    if flat_key in raw_weights:
        return float(raw_weights[flat_key])
    nested = raw_weights.get(mode)
    if isinstance(nested, Mapping) and condition in nested:
        return float(nested[condition])
    return None


def normalize_synthesis_profile_weights(
    raw_weights: Optional[Mapping[str, Any]],
) -> Dict[Tuple[str, str], float]:
    if not isinstance(raw_weights, Mapping):
        raise TypeError("synthesis_profile_weights must be a mapping")

    legal_flat_keys = {f"{mode}.{condition}" for mode, condition in SYNTHESIS_PROFILE_ORDER}
    legal_modes = {mode for mode, _condition in SYNTHESIS_PROFILE_ORDER}
    unknown = []
    for key, value in raw_weights.items():
        if key in legal_flat_keys:
            continue
        if key in legal_modes:
            if not isinstance(value, Mapping):
                unknown.append(str(key))
                continue
            for condition_key in value:
                if f"{key}.{condition_key}" not in legal_flat_keys:
                    unknown.append(f"{key}.{condition_key}")
            continue
        unknown.append(str(key))
        continue
    if unknown:
        raise ValueError(
            "Unknown synthesis_profile_weights keys: " + ", ".join(sorted(unknown))
        )

    weights: Dict[Tuple[str, str], float] = {}
    for mode, condition in SYNTHESIS_PROFILE_ORDER:
        value = _profile_weight_value(raw_weights, mode, condition)
        if value is None:
            value = 0.0
        if value < 0:
            raise ValueError(
                f"synthesis_profile_weights.{mode}.{condition} must be non-negative"
            )
        weights[(mode, condition)] = float(value)

    return weights


def _copy_segment_with_text(
    segment: Any,
    *,
    text: str,
    clip_suffix: str,
    duration_estimation_en_word_sec: float,
    duration_estimation_cjk_char_sec: float,
) -> Any:
    estimated_duration = estimate_text_duration_sec(
        text,
        en_word_sec=duration_estimation_en_word_sec,
        cjk_char_sec=duration_estimation_cjk_char_sec,
    )
    metadata = dict(getattr(segment, "metadata", {}) or {})
    metadata.update(
        {
            "synthesis_window_source_clip_id": getattr(segment, "clip_id", None),
            "synthesis_window_original_text": getattr(segment, "text", None),
        }
    )
    clip_id = f"{getattr(segment, 'clip_id', 'segment')}{clip_suffix}"
    if is_dataclass(segment):
        return replace(
            segment,
            clip_id=clip_id,
            text=text,
            duration=estimated_duration,
            metadata=metadata,
        )
    cloned = copy.copy(segment)
    cloned.clip_id = clip_id
    cloned.text = text
    cloned.duration = estimated_duration
    cloned.metadata = metadata
    return cloned


def window_to_segments(
    segments: Sequence[Any],
    window: TextWindow,
    *,
    duration_estimation_en_word_sec: float,
    duration_estimation_cjk_char_sec: float,
) -> List[Any]:
    selected: List[Any] = []
    for window_pos, unit in enumerate(window.units):
        source = segments[unit.source_segment_index]
        suffix = f"#win{window.start_unit + window_pos}"
        selected.append(
            _copy_segment_with_text(
                source,
                text=unit.text,
                clip_suffix=suffix,
                duration_estimation_en_word_sec=duration_estimation_en_word_sec,
                duration_estimation_cjk_char_sec=duration_estimation_cjk_char_sec,
            )
        )
    return selected


def select_profile_and_segments(
    gen_segments: Sequence[Any],
    *,
    rng: Any,
    emotion_condition_mode: Optional[str] = None,
    synthesis_mode: Optional[str] = None,
    synthesis_profile_weights: Optional[Mapping[str, Any]] = None,
    language: Optional[str] = None,
    plan_text_unit_cap_en: Optional[int] = None,
    plan_text_unit_cap_zh: Optional[int] = None,
    duration_estimation_en_word_sec: float,
    duration_estimation_cjk_char_sec: float,
    ted_estimated_duration_soft_limit_sec: float,
    ted_max_generated_duration_sec: float,
) -> Tuple[Dict[str, Any], List[Any]]:
    """Pick one legal profile by weight, then one legal text window uniformly."""
    forced_condition = normalize_emotion_condition_mode(
        emotion_condition_mode, allow_infer=True
    )
    forced_mode = normalize_synthesis_mode(synthesis_mode, allow_infer=True)
    units = split_text_units_ordered(
        gen_segments,
        duration_estimation_en_word_sec=duration_estimation_en_word_sec,
        duration_estimation_cjk_char_sec=duration_estimation_cjk_char_sec,
    )
    windows = enumerate_contiguous_text_windows(units)
    if not windows:
        raise ValueError("Cannot build synthesis profile from empty generation text")

    text_unit_cap = _text_unit_cap_for_language(
        language,
        plan_text_unit_cap_en=plan_text_unit_cap_en,
        plan_text_unit_cap_zh=plan_text_unit_cap_zh,
    )
    if text_unit_cap is not None:
        windows = [
            window
            for window in windows
            if count_profile_text_units(window.text, language) <= text_unit_cap
        ]
        if not windows:
            raise ValueError(
                f"No text window satisfies plan text unit cap {text_unit_cap} "
                f"for language={language!r}"
            )

    profile_to_windows: Dict[Tuple[str, str], List[TextWindow]] = {
        profile: [] for profile in SYNTHESIS_PROFILE_ORDER
    }
    for window in windows:
        for mode, condition in legal_synthesis_tuples_for_duration(
            window.estimated_duration_sec,
            ted_estimated_duration_soft_limit_sec=ted_estimated_duration_soft_limit_sec,
        ):
            if forced_condition != "infer" and condition != forced_condition:
                continue
            if forced_mode != "infer" and mode != forced_mode:
                continue
            profile_to_windows.setdefault((mode, condition), []).append(window)

    profile_weights = normalize_synthesis_profile_weights(synthesis_profile_weights)
    available_profiles: List[Tuple[str, str]] = []
    available_weights: List[float] = []
    for profile in SYNTHESIS_PROFILE_ORDER:
        windows_for_profile = profile_to_windows.get(profile) or []
        weight = profile_weights.get(profile, 0.0)
        if windows_for_profile and weight > 0.0:
            available_profiles.append(profile)
            available_weights.append(weight)
    if not available_profiles:
        raise ValueError(
            "No legal synthesis profile for requested backend/mode, profile weights, "
            "and sampled text"
        )

    mode, condition = rng.choices(available_profiles, weights=available_weights, k=1)[0]
    window = rng.choice(profile_to_windows[(mode, condition)])
    selected_window = window.to_dict()
    selected_window["text_unit_count"] = count_profile_text_units(window.text, language)
    if text_unit_cap is not None:
        selected_window["text_unit_cap"] = text_unit_cap
    selected_window["duration_estimation_en_word_sec"] = float(
        duration_estimation_en_word_sec
    )
    selected_window["duration_estimation_cjk_char_sec"] = float(
        duration_estimation_cjk_char_sec
    )
    if language:
        selected_window["language"] = language
    profile = build_synthesis_profile(
        synthesis_mode=mode,
        emotion_condition_mode=condition,
        selected_text_window=selected_window,
        estimated_duration_sec=window.estimated_duration_sec,
        ted_estimated_duration_soft_limit_sec=ted_estimated_duration_soft_limit_sec,
        ted_max_generated_duration_sec=ted_max_generated_duration_sec,
    )
    return profile, window_to_segments(
        gen_segments,
        window,
        duration_estimation_en_word_sec=duration_estimation_en_word_sec,
        duration_estimation_cjk_char_sec=duration_estimation_cjk_char_sec,
    )


def validate_tagged_texts_for_profile(
    tagged_texts: Mapping[str, str],
    profile: Mapping[str, Any],
    *,
    require_explicit_index_level: bool = False,
) -> None:
    normalized = normalize_synthesis_profile(profile)
    mode = normalized["synthesis_mode"]
    condition = normalized["emotion_condition_mode"]
    resolve_backend_for_profile(mode, condition)

    for text_id, tagged_text in tagged_texts.items():
        tagged_segments = [
            segment for segment in parse_tags(tagged_text) if segment.tag_name is not None
        ]
        if not tagged_segments:
            raise ValueError(f"Tagged text {text_id} has no tagged segments")

        if condition == EMOTION_CONDITION_TYPE_LEVEL:
            for segment in tagged_segments:
                if segment.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
                    raise ValueError(
                        f"Tagged text {text_id} uses desc but emotion_condition_mode=type_level"
                    )
                if segment.dur is not None:
                    raise ValueError(
                        f"Tagged text {text_id} uses dur but emotion_condition_mode=type_level"
                    )
                if (
                    require_explicit_index_level
                    and segment.tag_name != "none"
                    and segment.emotion_level is None
                ):
                    raise ValueError(
                        f"Tagged text {text_id} must use explicit level for type_level mode"
                    )
            continue

        for segment in tagged_segments:
            if segment.tag_type == "none":
                if segment.dur is not None:
                    raise ValueError(
                        f"Tagged text {text_id} uses duration tags, which are not supported for emotion editing"
                    )
                continue
            if segment.emotion_control_mode != EMOTION_CONTROL_DESCRIPTION:
                raise ValueError(
                    f"Tagged text {text_id} must use desc tags for emotion_condition_mode={condition}"
                )
            if not segment.emotion_type_ref:
                raise ValueError(
                    f"Tagged text {text_id} must include type_ref for desc mode"
                )
            if segment.dur is not None:
                raise ValueError(
                    f"Tagged text {text_id} uses duration tags, which are not supported for emotion editing"
                )


def service_mode_from_contract(
    *,
    is_whole_sentence_edit: bool,
    requested_mode: Optional[str] = None,
) -> str:
    mode = normalize_synthesis_mode(requested_mode, allow_infer=True)
    if mode != "infer":
        return mode
    if is_whole_sentence_edit:
        return SYNTHESIS_MODE_SINGLE
    return SYNTHESIS_MODE_MULTI_NORMAL
