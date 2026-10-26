"""Tagged-text adapter for TED-TTS service requests.

The data pipeline uses canonical ``<emo>`` tags, while TED-TTS expects parallel
``text`` and ``emo_text`` strings split by ``|``.  Keep that translation inside
the TED-TTS tool boundary so callers can continue to speak in tags.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from typing import List, Optional


EMOTION_TAGS = {
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
}
NEUTRAL_TAG = "none"

LEVEL_TO_ALPHA = {
    "afraid": {1: 0.8, 2: 0.9, 3: 1.0},
    "angry": {1: 0.72, 2: 0.85, 3: 1.0},
    "calm": {1: 0.8, 2: 0.9, 3: 1.0},
    "disgusted": {1: 0.8, 2: 0.9, 3: 1.0},
    "happy": {1: 0.83, 2: 0.93, 3: 1.0},
    "melancholic": {1: 0.8, 2: 0.9, 3: 1.0},
    "sad": {1: 0.8, 2: 0.9, 3: 1.0},
    "surprised": {1: 0.72, 2: 0.8, 3: 0.88},
}
DEFAULT_LEVEL = 2

_TAG_PATTERN = re.compile(r"<emo\b([^>]*)>(.*?)</emo>", re.DOTALL | re.IGNORECASE)
_ATTR_PATTERN = re.compile(
    r"([a-zA-Z_][\w-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\"'\s>]+))"
)

_EMOTION_DESCRIPTIONS = {
    "happy": "高兴 happy: {intensity} cheerful, bright, warm, smiling voice",
    "angry": "愤怒 angry: {intensity} tense, sharp, forceful, pressed voice",
    "sad": "悲伤 sad: {intensity} low, soft, slow, downcast voice",
    "afraid": "恐惧 afraid: {intensity} nervous, cautious, slightly trembling voice",
    "disgusted": "反感 disgusted: {intensity} displeased, cold, repelled voice",
    "melancholic": "低落 melancholic: {intensity} gloomy, restrained, wistful, low-energy voice",
    "surprised": "惊讶 surprised: {intensity} alert, lifted pitch, sudden energy",
    "calm": "自然 calm: {intensity} neutral, steady, relaxed, no strong emotion",
    NEUTRAL_TAG: "自然 calm: neutral, steady, relaxed, no strong emotion",
}

_INTENSITY_WORDS = {
    1: "slightly",
    2: "clearly",
    3: "strongly",
}


@dataclass
class TaggedSegment:
    tag_name: str
    text: str
    alpha: Optional[float]
    level: Optional[int]
    emotion_description: str
    emotion_control_mode: str
    emotion_type_ref: Optional[str] = None
    dur: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaggedAdaptation:
    text: str
    emo_text: str
    segments: List[TaggedSegment]
    target_seconds: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "emo_text": self.emo_text,
            "segments": [segment.to_dict() for segment in self.segments],
            "target_seconds": self.target_seconds,
        }


def _nearest_level(tag_name: str, alpha: Optional[float], level: Optional[int]) -> int:
    if level is not None:
        return max(1, min(3, level))
    if alpha is None:
        return DEFAULT_LEVEL
    level_map = LEVEL_TO_ALPHA.get(tag_name)
    if not level_map:
        return DEFAULT_LEVEL
    return min(level_map, key=lambda candidate: abs(level_map[candidate] - alpha))


def _sanitize_pipe(value: str) -> str:
    return value.replace("|", " ").strip()


def _parse_attrs(raw_attrs: str) -> dict:
    attrs = {}
    cursor = 0
    raw_attrs = raw_attrs.strip()
    while cursor < len(raw_attrs):
        if raw_attrs[cursor].isspace():
            cursor += 1
            continue
        match = _ATTR_PATTERN.match(raw_attrs, cursor)
        if not match:
            raise ValueError(f"Malformed <emo> attributes: {raw_attrs!r}")
        key = match.group(1).lower()
        value = next(group for group in match.groups()[1:] if group is not None)
        if key in attrs:
            raise ValueError(f"Duplicate <emo> attribute: {key}")
        attrs[key] = html.unescape(value)
        cursor = match.end()
    return attrs


def _parse_dur(attrs: dict) -> Optional[float]:
    if "dur" not in attrs:
        return None
    try:
        dur = float(attrs["dur"])
    except ValueError as exc:
        raise ValueError("dur must be a positive number of seconds") from exc
    if dur <= 0:
        raise ValueError("dur must be a positive number of seconds")
    return dur


def render_emotion_description(
    tag_name: str,
    *,
    alpha: Optional[float] = None,
    level: Optional[int] = None,
) -> str:
    """Render a TED-TTS emotion-text prompt from one pipeline tag."""
    normalized = tag_name.lower().strip()
    if normalized == NEUTRAL_TAG:
        template = _EMOTION_DESCRIPTIONS[NEUTRAL_TAG]
        return template
    if normalized not in EMOTION_TAGS:
        raise ValueError(f"Unsupported TED-TTS emotion tag: {tag_name}")
    resolved_level = _nearest_level(normalized, alpha, level)
    intensity = _INTENSITY_WORDS[resolved_level]
    return _EMOTION_DESCRIPTIONS[normalized].format(intensity=intensity)


def adapt_tagged_text(tagged_text: str) -> TaggedAdaptation:
    """Convert canonical ``<emo>`` tagged text into TED-TTS text/emo_text."""
    segments: List[TaggedSegment] = []
    text_parts: List[str] = []
    emo_parts: List[str] = []
    last_end = 0

    for match in _TAG_PATTERN.finditer(tagged_text):
        untagged = tagged_text[last_end : match.start()]
        if untagged.strip():
            raise ValueError(
                "TED-TTS tagged synthesis requires every non-whitespace span to be tagged"
            )

        attrs = _parse_attrs(match.group(1))
        text = match.group(2)
        unknown_attrs = sorted(
            set(attrs) - {"type", "alpha", "level", "desc", "type_ref", "dur"}
        )
        if unknown_attrs:
            raise ValueError(f"Unsupported <emo> attributes: {unknown_attrs}")
        alpha = None
        level = None
        desc = attrs.get("desc")
        type_ref = attrs.get("type_ref")
        if type_ref is not None:
            type_ref = type_ref.lower().strip()
            if type_ref not in EMOTION_TAGS:
                raise ValueError(
                    f"Unsupported TED-TTS emotion type_ref: {type_ref}"
                )
        dur = _parse_dur(attrs)
        if desc:
            if "type" in attrs or "alpha" in attrs or "level" in attrs:
                raise ValueError("<emo desc=...> cannot be combined with type, alpha, or level")
            tag_name = "emo"
            description = desc.strip()
            control_mode = "description"
        else:
            if type_ref is not None:
                raise ValueError("<emo type_ref=...> requires desc")
            tag_name = attrs.get("type", "").lower().strip()
            if not tag_name:
                raise ValueError("<emo> must include either desc or type")
            if tag_name == NEUTRAL_TAG:
                if "alpha" in attrs or "level" in attrs:
                    raise ValueError('<emo type="none"> cannot include alpha or level')
            elif tag_name not in EMOTION_TAGS:
                raise ValueError(f"Unsupported TED-TTS emotion tag: {tag_name}")
            if "alpha" in attrs and "level" in attrs:
                raise ValueError("<emo> cannot include both alpha and level")
            if "alpha" in attrs:
                alpha = float(attrs["alpha"])
            elif "level" in attrs:
                level_token = attrs["level"].lower().strip()
                if level_token in {"", "none"}:
                    level = None
                else:
                    parsed_level = int(level_token)
                    if parsed_level == 0:
                        tag_name = NEUTRAL_TAG
                    else:
                        level = parsed_level
            description = render_emotion_description(tag_name, alpha=alpha, level=level)
            control_mode = "alpha"
        clean_text = _sanitize_pipe(text)
        clean_description = _sanitize_pipe(description)
        if not clean_text:
            raise ValueError("TED-TTS tagged synthesis does not support empty segments")

        segments.append(
            TaggedSegment(
                tag_name=tag_name,
                text=clean_text,
                alpha=alpha,
                level=level,
                emotion_description=clean_description,
                emotion_control_mode=control_mode,
                emotion_type_ref=type_ref,
                dur=dur,
            )
        )
        text_parts.append(clean_text)
        emo_parts.append(clean_description)
        last_end = match.end()

    trailing = tagged_text[last_end:]
    if trailing.strip():
        raise ValueError(
            "TED-TTS tagged synthesis requires every non-whitespace span to be tagged"
        )
    if not segments:
        raise ValueError("No tagged segments found")

    dur_flags = [segment.dur is not None for segment in segments]
    if any(dur_flags) and not all(dur_flags):
        raise ValueError("TED-TTS duration control requires dur on every tagged segment")
    target_seconds = [segment.dur for segment in segments] if all(dur_flags) else None

    return TaggedAdaptation(
        text="|".join(text_parts),
        emo_text="|".join(emo_parts),
        segments=segments,
        target_seconds=target_seconds,
    )
