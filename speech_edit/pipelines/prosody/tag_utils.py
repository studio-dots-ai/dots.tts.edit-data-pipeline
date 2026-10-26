"""Tag utilities for the pitch/rate editing pipeline.

Supported tag format
--------------------
  <pitch, semitones=3>word</pitch>
  <rate, factor=1.3>word</rate>

Pitch and rate tags must be non-overlapping. Applying pitch and rate to the
same text span is intentionally invalid for this pipeline.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PITCH_TAG = "pitch"
RATE_TAG = "rate"
VALID_TAG_TYPES = {PITCH_TAG, RATE_TAG}

PITCH_MIN_SEMITONES = -12
PITCH_MAX_SEMITONES = 12
RATE_MIN_FACTOR = 0.3
RATE_MAX_FACTOR = 3.0

ALL_PUNCTUATION = set(".,!?;:-–—…，。！？、；：()[]{}\"'")
ORIGINAL_WAVE_ID = "original"
RATE_VALUE_PLACEHOLDER = "[[RATE_VALUE_PH]]"


def _is_han_character(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    return 0x4E00 <= codepoint <= 0x9FFF


@dataclass
class TaggedSegment:
    """A plain-text span with an optional pitch or rate tag."""

    text: str
    tag_type: Optional[str] = None
    semitones: Optional[float] = None
    factor: Optional[float] = None

    def __post_init__(self) -> None:
        if self.tag_type is None:
            if self.semitones is not None or self.factor is not None:
                raise ValueError("Untagged segments cannot carry edit parameters")
            return
        if self.tag_type not in VALID_TAG_TYPES:
            raise ValueError(f"Unknown tag_type: {self.tag_type!r}")
        if self.semitones is not None and self.factor is not None:
            raise ValueError("Pitch and rate cannot be applied to the same segment")
        if self.tag_type == PITCH_TAG:
            if self.semitones is None or self.factor is not None:
                raise ValueError("Pitch segments require semitones only")
        if self.tag_type == RATE_TAG:
            if self.factor is None or self.semitones is not None:
                raise ValueError("Rate segments require factor only")

    @property
    def is_tagged(self) -> bool:
        return self.tag_type is not None

    @property
    def param_str(self) -> str:
        if self.semitones is not None:
            sign = "+" if self.semitones > 0 else ""
            return f"{sign}{self.semitones}st"
        if self.factor is not None:
            return f"{self.factor}x"
        return ""


@dataclass
class TextBoundary:
    """A candidate text boundary where edit tags can switch."""

    char_idx: int
    has_punctuation: bool


@dataclass
class WaveEditPair:
    """One editing direction between two wave ids."""

    source_id: str
    target_id: str
    plain_text: str
    source_tagged_text: str
    target_tagged_text: str
    instructions: Dict[str, Any]
    tag_diff: List[Dict[str, Any]] = field(default_factory=list)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "plain_text": self.plain_text,
            "source_tagged_text": self.source_tagged_text,
            "target_tagged_text": self.target_tagged_text,
            "instructions": self.instructions,
            "tag_diff": self.tag_diff,
        }


@dataclass
class WaveGroup:
    """Multi-wave pitch/rate editing plan for one utterance."""

    plain_text: str
    tagged_texts: Dict[str, str]
    edit_pairs: List[WaveEditPair]
    boundaries: List[TextBoundary]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "plain_text": self.plain_text,
            "tagged_texts": self.tagged_texts,
            "edit_pairs": [pair.to_metadata_dict() for pair in self.edit_pairs],
            "boundaries": [
                {
                    "char_idx": boundary.char_idx,
                    "has_punctuation": boundary.has_punctuation,
                }
                for boundary in self.boundaries
            ],
            "stage1_info": self.metadata,
        }


_TAG_PATTERN = re.compile(
    r"<(pitch|rate)\s*,\s*"
    r"(semitones|factor)\s*=\s*"
    r"(-?[\d.]+)\s*>"
    r"(.*?)"
    r"</(pitch|rate)>",
    re.DOTALL,
)
_ANY_TAG_PATTERN = re.compile(r"</?[a-zA-Z_][^>]*>")


def parse_tags(tagged_text: str) -> List[TaggedSegment]:
    """Parse non-overlapping pitch/rate tags into ordered text segments."""
    segments: List[TaggedSegment] = []
    pos = 0

    for match in _TAG_PATTERN.finditer(tagged_text):
        start, end = match.span()
        if start < pos:
            raise ValueError("Overlapping pitch/rate tags are not supported")
        if start > pos:
            prefix = tagged_text[pos:start]
            _reject_residual_tags(prefix)
            segments.append(TaggedSegment(text=prefix))
        segment = _parse_simple_match(match)
        _reject_residual_tags(segment.text)
        segments.append(segment)
        pos = end

    if pos < len(tagged_text):
        tail = tagged_text[pos:]
        _reject_residual_tags(tail)
        segments.append(TaggedSegment(text=tail))

    if not segments and tagged_text:
        _reject_residual_tags(tagged_text)
        segments.append(TaggedSegment(text=tagged_text))

    return segments


def _reject_residual_tags(text: str) -> None:
    if _ANY_TAG_PATTERN.search(text):
        raise ValueError(
            "Nested, crossing, malformed, or overlapping pitch/rate tags are not supported"
        )


def _parse_simple_match(match: re.Match) -> TaggedSegment:
    open_tag = match.group(1)
    param_name = match.group(2)
    param_value_str = match.group(3)
    content = match.group(4)
    close_tag = match.group(5)

    if open_tag != close_tag:
        raise ValueError(f"Mismatched tags: <{open_tag}> ... </{close_tag}>")

    try:
        param_value = float(param_value_str)
    except ValueError:
        raise ValueError(f"Invalid parameter value: {param_value_str}") from None

    if open_tag == PITCH_TAG:
        if param_name != "semitones":
            raise ValueError(
                f"Pitch tag expects 'semitones' parameter, got '{param_name}'"
            )
        if not re.fullmatch(r"-?\d+", param_value_str):
            raise ValueError("Pitch semitones must be an integer")
        if not (PITCH_MIN_SEMITONES <= param_value <= PITCH_MAX_SEMITONES):
            raise ValueError(
                f"Semitones {param_value} out of range "
                f"[{PITCH_MIN_SEMITONES}, {PITCH_MAX_SEMITONES}]"
            )
        return TaggedSegment(text=content, tag_type=PITCH_TAG, semitones=int(param_value))

    if param_name != "factor":
        raise ValueError(f"Rate tag expects 'factor' parameter, got '{param_name}'")
    if not (RATE_MIN_FACTOR <= param_value <= RATE_MAX_FACTOR):
        raise ValueError(
            f"Factor {param_value} out of range [{RATE_MIN_FACTOR}, {RATE_MAX_FACTOR}]"
        )
    return TaggedSegment(text=content, tag_type=RATE_TAG, factor=param_value)


def render_tagged_text(segments: List[TaggedSegment]) -> str:
    parts: List[str] = []
    for seg in segments:
        if seg.tag_type == PITCH_TAG and seg.semitones is not None:
            parts.append(
                f"<pitch, semitones={_format_number(seg.semitones)}>{seg.text}</pitch>"
            )
        elif seg.tag_type == RATE_TAG and seg.factor is not None:
            parts.append(f"<rate, factor={_format_number(seg.factor)}>{seg.text}</rate>")
        else:
            parts.append(seg.text)
    return "".join(parts)


def strip_tags(tagged_text: str) -> str:
    """Remove pitch/rate tags from valid simple-tag text."""
    previous = None
    text = tagged_text
    while text != previous:
        previous = text
        text = _TAG_PATTERN.sub(r"\4", text)
    text = re.sub(r"</?(?:pitch|rate)[^>]*>", "", text)
    return text


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_tagged_text(
    tagged_text: str,
    source_plain_text: str,
) -> Tuple[bool, str]:
    """Validate syntax, plain-text preservation, and lexical edit boundaries."""
    try:
        segments = parse_tags(tagged_text)
    except ValueError as exc:
        return False, f"Parse error: {exc}"

    plain = "".join(seg.text for seg in segments)
    if _normalize_ws(plain) != _normalize_ws(source_plain_text):
        return False, (
            f"Plain text mismatch.\n"
            f"  Expected: {_normalize_ws(source_plain_text)!r}\n"
            f"  Got:      {_normalize_ws(plain)!r}"
        )
    if not any(seg.is_tagged for seg in segments):
        return False, "No pitch/rate tags found"
    return validate_boundaries_at_punctuation(tagged_text, plain)


def describe_edits(segments: List[TaggedSegment]) -> List[Dict[str, Any]]:
    edits: List[Dict[str, Any]] = []
    for seg in segments:
        if seg.tag_type == PITCH_TAG and seg.semitones is not None:
            edits.append(
                {
                    "word": seg.text.strip(),
                    "edit_type": "pitch",
                    "semitones": seg.semitones,
                }
            )
        elif seg.tag_type == RATE_TAG and seg.factor is not None:
            edits.append(
                {
                    "word": seg.text.strip(),
                    "edit_type": "rate",
                    "factor": seg.factor,
                }
            )
    return edits


def summarize_tagged_text(tagged_text: str) -> Dict[str, int]:
    """Return canonical edit counts for one tagged wave."""
    pitch_regions = 0
    rate_regions = 0
    for seg in parse_tags(tagged_text):
        if seg.tag_type == PITCH_TAG:
            pitch_regions += 1
        elif seg.tag_type == RATE_TAG:
            rate_regions += 1
    return {
        "num_regions": pitch_regions + rate_regions,
        "num_pitch_regions": pitch_regions,
        "num_rate_regions": rate_regions,
    }


def detect_boundaries(text: str, min_segment_words: int = 2) -> List[TextBoundary]:
    boundaries: List[TextBoundary] = []
    for idx, char in enumerate(text):
        if char not in ALL_PUNCTUATION:
            continue
        boundary_pos = idx + 1
        while boundary_pos < len(text) and text[boundary_pos].isspace():
            boundary_pos += 1
        boundaries.append(TextBoundary(char_idx=boundary_pos, has_punctuation=True))

    if not boundaries:
        word_spans = list(re.finditer(r"\S+", text))
        if len(word_spans) > min_segment_words * 2:
            for idx in range(min_segment_words, len(word_spans), min_segment_words):
                boundary_pos = word_spans[idx].start()
                if 0 < boundary_pos < len(text):
                    boundaries.append(
                        TextBoundary(char_idx=boundary_pos, has_punctuation=False)
                    )

    deduped: List[TextBoundary] = []
    seen = set()
    for boundary in boundaries:
        if boundary.char_idx <= 0 or boundary.char_idx >= len(text):
            continue
        if boundary.char_idx in seen:
            continue
        seen.add(boundary.char_idx)
        deduped.append(boundary)
    return deduped


def validate_boundaries_at_punctuation(
    tagged_text: str,
    plain_text: str,
) -> Tuple[bool, str]:
    segments = parse_tags(tagged_text)
    pos = 0
    for idx, seg in enumerate(segments[:-1]):
        pos += len(seg.text)
        next_seg = segments[idx + 1]
        if _segment_boundary_signature(seg) == _segment_boundary_signature(next_seg):
            continue
        if pos <= 0 or pos >= len(plain_text):
            continue
        char_before = plain_text[pos - 1]
        char_after = plain_text[pos]
        if (
            char_before not in ALL_PUNCTUATION
            and char_after not in ALL_PUNCTUATION
            and not char_before.isspace()
            and not char_after.isspace()
            and not (_is_han_character(char_before) or _is_han_character(char_after))
        ):
            return False, (
                f"Tag boundary at position {pos} is not at punctuation, whitespace, or CJK boundary. "
                f"chars around: ...{plain_text[max(0, pos - 3) : pos + 3]}..."
            )
    return True, ""


def describe_tag_diff(
    source_tagged_text: str,
    target_tagged_text: str,
) -> List[Dict[str, Any]]:
    source_records = _collect_tagged_segment_records(source_tagged_text)
    target_records = _collect_tagged_segment_records(target_tagged_text)
    source_map = {
        (record["start_char"], record["end_char"]): record for record in source_records
    }
    target_map = {
        (record["start_char"], record["end_char"]): record for record in target_records
    }

    diffs: List[Dict[str, Any]] = []
    for span in sorted(set(source_map) | set(target_map)):
        src_record = source_map.get(span)
        tgt_record = target_map.get(span)
        src_seg = src_record["segment"] if src_record is not None else None
        tgt_seg = tgt_record["segment"] if tgt_record is not None else None
        src_pitch = src_seg.semitones if src_seg is not None else None
        src_rate = src_seg.factor if src_seg is not None else None
        tgt_pitch = tgt_seg.semitones if tgt_seg is not None else None
        tgt_rate = tgt_seg.factor if tgt_seg is not None else None
        if src_pitch == tgt_pitch and src_rate == tgt_rate:
            continue

        record = tgt_record or src_record
        if record is None:
            continue
        diffs.append(
            {
                "word": record["word"],
                "occurrence": record["occurrence"],
                "span": {
                    "start_char": span[0],
                    "end_char": span[1],
                },
                "source": {
                    "edit_type": _segment_edit_type(src_seg),
                    "semitones": src_pitch,
                    "factor": src_rate,
                },
                "target": {
                    "edit_type": _segment_edit_type(tgt_seg),
                    "semitones": tgt_pitch,
                    "factor": tgt_rate,
                },
                "delta": {
                    "semitones": _delta_semitones(src_pitch, tgt_pitch),
                    "factor": _delta_factor(src_rate, tgt_rate),
                },
            }
        )
    return diffs


def _collect_tagged_segment_records(tagged_text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    pos = 0
    occurrences: Dict[str, int] = {}
    for seg in parse_tags(tagged_text):
        start_char = pos
        end_char = pos + len(seg.text)
        pos = end_char
        if not seg.is_tagged:
            continue
        word = seg.text.strip()
        if not word:
            continue
        occurrence = occurrences.get(word, 0) + 1
        occurrences[word] = occurrence
        records.append(
            {
                "word": word,
                "occurrence": occurrence,
                "start_char": start_char,
                "end_char": end_char,
                "segment": seg,
            }
        )
    return records


def _segment_edit_type(seg: Optional[TaggedSegment]) -> str:
    if seg is None or not seg.is_tagged:
        return "none"
    return seg.tag_type or "none"


def _segment_boundary_signature(seg: TaggedSegment) -> Tuple[Any, ...]:
    return (_segment_edit_type(seg), seg.semitones, seg.factor)


def _delta_semitones(
    source: Optional[float],
    target: Optional[float],
) -> Optional[float]:
    src = source or 0.0
    tgt = target or 0.0
    delta = tgt - src
    if abs(delta) < 1e-6:
        return None
    return round(delta, 2)


def _delta_factor(source: Optional[float], target: Optional[float]) -> Optional[float]:
    src = source or 1.0
    tgt = target or 1.0
    if abs(tgt - src) < 1e-6:
        return None
    return round(tgt / src, 3)


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_rate_inverse(value: float) -> str:
    return f"{1.0 / value:.2f}"


def _is_whole_sentence(seg: TaggedSegment, all_segments: List[TaggedSegment]) -> bool:
    tagged = [s for s in all_segments if s.is_tagged]
    if len(tagged) != 1:
        return False
    total_text = "".join(s.text for s in all_segments)
    return _normalize_ws(seg.text) == _normalize_ws(total_text)


def _is_sentence_transition(target_tagged_text: str, plain_text: str) -> bool:
    tagged = [seg for seg in parse_tags(target_tagged_text) if seg.is_tagged]
    return len(tagged) == 1 and _normalize_ws(tagged[0].text) == _normalize_ws(
        plain_text
    )


def generate_precise_instruction(segments: List[TaggedSegment]) -> str:
    parts: List[str] = []
    for seg in segments:
        if not seg.is_tagged:
            continue
        word = seg.text.strip()
        is_sentence = _is_whole_sentence(seg, segments)
        if seg.tag_type == PITCH_TAG and seg.semitones is not None:
            parts.append(_precise_pitch_part(word, seg.semitones, is_sentence))
        elif seg.tag_type == RATE_TAG and seg.factor is not None:
            parts.append(_precise_rate_part(word, seg.factor, is_sentence))
    return _join_instruction_parts(parts, empty="No edits")


def _precise_pitch_part(word: str, semitones: float, is_sentence: bool) -> str:
    amount = _format_number(abs(semitones))
    direction = "raise" if semitones > 0 else "lower"
    if is_sentence:
        return f"{direction} the pitch of the entire utterance by {amount} semitones"
    return f"{direction} the pitch of '{word}' by {amount} semitones"


def _precise_rate_part(word: str, factor: float, is_sentence: bool) -> str:
    factor_text = _format_number(factor)
    if factor > 1.0:
        if is_sentence:
            return f"speed up the entire utterance to {factor_text}x"
        return f"speed up '{word}' to {factor_text}x"
    if is_sentence:
        return f"slow down the entire utterance to {factor_text}x"
    return f"slow down '{word}' to {factor_text}x"


def generate_vague_instruction(segments: List[TaggedSegment]) -> str:
    parts: List[str] = []
    for seg in segments:
        if not seg.is_tagged:
            continue
        word = seg.text.strip()
        is_sentence = _is_whole_sentence(seg, segments)
        if seg.tag_type == PITCH_TAG and seg.semitones is not None:
            parts.append(_vague_pitch_part(word, seg.semitones, is_sentence))
        elif seg.tag_type == RATE_TAG and seg.factor is not None:
            parts.append(_vague_rate_part(word, seg.factor, is_sentence))
    return _join_instruction_parts(parts, empty="No edits")


def _vague_pitch_part(word: str, semitones: float, is_sentence: bool) -> str:
    if is_sentence:
        templates = (
            [
                "make the whole thing higher-pitched",
                "raise the overall tone",
                "make it sound brighter overall",
            ]
            if semitones > 0
            else [
                "make the whole thing sound deeper",
                "lower the overall tone",
                "deepen the voice throughout",
            ]
        )
        return random.choice(templates)
    templates = (
        [
            "make '{word}' sound higher-pitched",
            "raise the tone of '{word}'",
            "make '{word}' sound brighter",
        ]
        if semitones > 0
        else [
            "make '{word}' sound deeper",
            "lower the tone of '{word}'",
            "give '{word}' a deeper pitch",
        ]
    )
    return random.choice(templates).format(word=word)


def _vague_rate_part(word: str, factor: float, is_sentence: bool) -> str:
    if is_sentence:
        templates = (
            [
                "say the whole thing faster",
                "speed up the entire utterance",
                "pick up the pace throughout",
            ]
            if factor > 1.0
            else [
                "say the whole thing more slowly",
                "slow down the entire utterance",
                "make it more drawn out overall",
            ]
        )
        return random.choice(templates)
    templates = (
        [
            "make '{word}' faster",
            "speed up '{word}'",
            "say '{word}' more quickly",
        ]
        if factor > 1.0
        else [
            "make '{word}' slower",
            "slow down '{word}'",
            "draw out '{word}'",
        ]
    )
    return random.choice(templates).format(word=word)


def generate_precise_instruction_zh(segments: List[TaggedSegment]) -> str:
    parts: List[str] = []
    for seg in segments:
        if not seg.is_tagged:
            continue
        word = seg.text.strip()
        is_sentence = _is_whole_sentence(seg, segments)
        if seg.semitones is not None:
            parts.append(_zh_pitch_phrase(seg.semitones, word, is_sentence))
        elif seg.factor is not None:
            parts.append(_zh_rate_phrase(seg.factor, word, is_sentence))
    return "；".join(parts) if parts else "不做编辑"


def generate_vague_instruction_zh(segments: List[TaggedSegment]) -> str:
    parts: List[str] = []
    for seg in segments:
        if not seg.is_tagged:
            continue
        word = seg.text.strip()
        is_sentence = _is_whole_sentence(seg, segments)
        if seg.semitones is not None:
            direction = "更高" if seg.semitones > 0 else "更低"
            parts.append(
                f"让整句听起来音高{direction}"
                if is_sentence
                else f"让“{word}”听起来音高{direction}"
            )
        elif seg.factor is not None:
            direction = "更快" if seg.factor > 1.0 else "更慢"
            parts.append(
                f"让整句说得{direction}"
                if is_sentence
                else f"让“{word}”说得{direction}"
            )
    return "；".join(parts) if parts else "不做编辑"


def _zh_pitch_phrase(semitones: float, word: str, is_sentence: bool) -> str:
    direction = "升高" if semitones > 0 else "降低"
    amount = _format_number(abs(semitones))
    if is_sentence:
        return f"将整句音高{direction}{amount}个半音"
    return f"将“{word}”的音高{direction}{amount}个半音"


def _zh_rate_phrase(factor: float, word: str, is_sentence: bool) -> str:
    direction = "加快" if factor > 1.0 else "放慢"
    suffix = f"到{_format_number(factor)}倍速"
    if is_sentence:
        return f"将整句语速{direction}{suffix}"
    return f"将“{word}”的语速{direction}{suffix}"


def generate_precise_instruction_transition(
    source_tagged_text: str,
    target_tagged_text: str,
    plain_text: str,
) -> str:
    parts: List[str] = []
    is_sentence = _is_sentence_transition(target_tagged_text, plain_text)
    for diff in describe_tag_diff(source_tagged_text, target_tagged_text):
        word = diff["word"]
        delta_pitch = diff["delta"]["semitones"]
        delta_factor = diff["delta"]["factor"]
        if delta_pitch is not None:
            parts.append(_precise_pitch_part(word, delta_pitch, is_sentence))
        if delta_factor is not None:
            parts.append(_precise_rate_transition_part(word, delta_factor, is_sentence))
    return _join_instruction_parts(parts, empty="No edits")


def generate_vague_instruction_transition(
    source_tagged_text: str,
    target_tagged_text: str,
    plain_text: str,
) -> str:
    parts: List[str] = []
    is_sentence = _is_sentence_transition(target_tagged_text, plain_text)
    for diff in describe_tag_diff(source_tagged_text, target_tagged_text):
        word = diff["word"]
        delta_pitch = diff["delta"]["semitones"]
        delta_factor = diff["delta"]["factor"]
        if delta_pitch is not None:
            parts.append(_vague_pitch_part(word, delta_pitch, is_sentence))
        if delta_factor is not None:
            parts.append(_vague_rate_part(word, delta_factor, is_sentence))
    return _join_instruction_parts(parts, empty="No edits")


def generate_precise_instruction_transition_zh(
    source_tagged_text: str,
    target_tagged_text: str,
    plain_text: str,
) -> str:
    parts: List[str] = []
    is_sentence = _is_sentence_transition(target_tagged_text, plain_text)
    for diff in describe_tag_diff(source_tagged_text, target_tagged_text):
        word = diff["word"]
        delta_pitch = diff["delta"]["semitones"]
        delta_factor = diff["delta"]["factor"]
        if delta_pitch is not None:
            parts.append(_zh_pitch_phrase(delta_pitch, word, is_sentence))
        if delta_factor is not None:
            parts.append(_zh_rate_transition_phrase(delta_factor, word, is_sentence))
    return "；".join(parts) if parts else "不做编辑"


def generate_vague_instruction_transition_zh(
    source_tagged_text: str,
    target_tagged_text: str,
    plain_text: str,
) -> str:
    parts: List[str] = []
    is_sentence = _is_sentence_transition(target_tagged_text, plain_text)
    for diff in describe_tag_diff(source_tagged_text, target_tagged_text):
        word = diff["word"]
        delta_pitch = diff["delta"]["semitones"]
        delta_factor = diff["delta"]["factor"]
        if delta_pitch is not None:
            direction = "更高" if delta_pitch > 0 else "更低"
            parts.append(
                f"让整句听起来音高{direction}"
                if is_sentence
                else f"让“{word}”听起来音高{direction}"
            )
        if delta_factor is not None:
            direction = "更快" if delta_factor > 1.0 else "更慢"
            parts.append(
                f"让整句说得{direction}"
                if is_sentence
                else f"让“{word}”说得{direction}"
            )
    return "；".join(parts) if parts else "不做编辑"


def _precise_rate_transition_part(word: str, factor: float, is_sentence: bool) -> str:
    if factor > 1.0:
        factor_text = _format_number(factor)
        if is_sentence:
            return (
                f"speed up the entire utterance by {factor_text}x "
                "relative to the source rate"
            )
        return f"speed up '{word}' by {factor_text}x relative to the source rate"

    slower_factor = _format_number(1.0 / factor)
    if is_sentence:
        return (
            f"slow down the entire utterance by {slower_factor}x "
            "relative to the source rate"
        )
    return f"slow down '{word}' by {slower_factor}x relative to the source rate"


def _zh_rate_transition_phrase(factor: float, word: str, is_sentence: bool) -> str:
    if factor > 1.0:
        amount = _format_number(factor)
        if is_sentence:
            return f"将整句语速相对源语速加快{amount}倍"
        return f"将“{word}”的语速相对源语速加快{amount}倍"

    amount = _format_number(1.0 / factor)
    if is_sentence:
        return f"将整句语速相对源语速放慢{amount}倍"
    return f"将“{word}”的语速相对源语速放慢{amount}倍"


def _join_instruction_parts(parts: List[str], *, empty: str) -> str:
    if not parts:
        return empty
    parts[0] = parts[0][0].upper() + parts[0][1:]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def pair_key(source_id: str, target_id: str) -> str:
    return f"{source_id}_to_{target_id}"


def invert_structural_instruction(tagged_text: str) -> str:
    """Return the structural instruction for reversing one generated wave."""
    parts: List[str] = []
    for seg in parse_tags(tagged_text):
        if seg.tag_type == PITCH_TAG and seg.semitones is not None:
            parts.append(
                f"<pitch, semitones={_format_number(-seg.semitones)}>"
                f"{seg.text}</pitch>"
            )
        elif seg.tag_type == RATE_TAG and seg.factor is not None:
            parts.append(
                f"<rate, factor={_format_rate_inverse(seg.factor)}>"
                f"{seg.text}</rate>"
            )
        else:
            parts.append(seg.text)
    return "".join(parts)


def _rate_inverse_values(tagged_text: str) -> List[str]:
    return [
        _format_rate_inverse(seg.factor)
        for seg in parse_tags(tagged_text)
        if seg.tag_type == RATE_TAG and seg.factor is not None
    ]


def _fill_rate_placeholders(text: str, values: List[str]) -> str:
    expected = len(values)
    actual = text.count(RATE_VALUE_PLACEHOLDER)
    if actual != expected:
        raise ValueError(
            f"Expected {expected} {RATE_VALUE_PLACEHOLDER} placeholders, got {actual}"
        )
    for value in values:
        text = text.replace(RATE_VALUE_PLACEHOLDER, value, 1)
    return text


def _canonical_instructions(
    raw: Dict[str, Any],
    *,
    pair_id: str,
    canonical_st: str,
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{pair_id} instructions must be an object")
    nl = raw.get("nl")
    if not isinstance(nl, dict):
        raise ValueError(f"{pair_id} instructions.nl must be an object")
    precise = nl.get("precise")
    vague = nl.get("vague")
    if not isinstance(precise, str) or not precise.strip():
        raise ValueError(f"{pair_id} instructions.nl.precise must be a non-empty string")
    if not isinstance(vague, str) or not vague.strip():
        raise ValueError(f"{pair_id} instructions.nl.vague must be a non-empty string")
    return {
        "nl": {
            "precise": precise.strip(),
            "vague": vague.strip(),
        },
        "st": canonical_st,
    }


def _deterministic_pair_instructions(
    source_tagged_text: str,
    target_tagged_text: str,
    plain_text: str,
) -> Dict[str, Any]:
    return {
        "nl": {
            "precise": generate_precise_instruction_transition(
                source_tagged_text,
                target_tagged_text,
                plain_text,
            ),
            "vague": generate_vague_instruction_transition(
                source_tagged_text,
                target_tagged_text,
                plain_text,
            ),
        },
        "st": target_tagged_text,
    }


def normalize_pair_instructions_for_wave(
    plain_text: str,
    wave_id: str,
    tagged_text: str,
    raw_pair_instructions: Optional[Dict[str, Any]] = None,
    *,
    allow_filled_reverse: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Validate and canonicalize ordered pair instructions for one wave."""
    forward_key = pair_key(ORIGINAL_WAVE_ID, wave_id)
    reverse_key = pair_key(wave_id, ORIGINAL_WAVE_ID)
    expected_reverse_st = invert_structural_instruction(tagged_text)
    inverse_rate_values = _rate_inverse_values(tagged_text)

    if raw_pair_instructions is None:
        return {
            forward_key: _deterministic_pair_instructions(
                plain_text,
                tagged_text,
                plain_text,
            ),
            reverse_key: {
                **_deterministic_pair_instructions(
                    tagged_text,
                    plain_text,
                    plain_text,
                ),
                "st": expected_reverse_st,
            },
        }

    if not isinstance(raw_pair_instructions, dict):
        raise ValueError(f"Wave {wave_id} pair_instructions must be an object")

    if forward_key in raw_pair_instructions:
        forward = _canonical_instructions(
            raw_pair_instructions[forward_key],
            pair_id=forward_key,
            canonical_st=tagged_text,
        )
    else:
        forward = _deterministic_pair_instructions(plain_text, tagged_text, plain_text)

    if reverse_key in raw_pair_instructions:
        reverse = _canonical_instructions(
            raw_pair_instructions[reverse_key],
            pair_id=reverse_key,
            canonical_st=expected_reverse_st,
        )
    else:
        reverse = {
            **_deterministic_pair_instructions(tagged_text, plain_text, plain_text),
            "st": expected_reverse_st,
        }

    forward_payload = f"{forward['nl']['precise']} {forward['nl']['vague']}"
    if RATE_VALUE_PLACEHOLDER in forward_payload:
        raise ValueError(f"{forward_key} must not contain {RATE_VALUE_PLACEHOLDER}")

    if RATE_VALUE_PLACEHOLDER in reverse["nl"]["vague"]:
        raise ValueError(f"{reverse_key} vague instruction must not use placeholders")
    reverse_precise_has_placeholder = RATE_VALUE_PLACEHOLDER in reverse["nl"]["precise"]
    if inverse_rate_values and allow_filled_reverse and not reverse_precise_has_placeholder:
        pass
    else:
        reverse["nl"]["precise"] = _fill_rate_placeholders(
            reverse["nl"]["precise"],
            inverse_rate_values,
        )

    return {
        forward_key: forward,
        reverse_key: reverse,
    }


def build_wave_group(
    plain_text: str,
    tagged_texts: Dict[str, str],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> WaveGroup:
    """Build bidirectional original<->wave pairs for independent waves."""
    if not tagged_texts:
        raise ValueError("tagged_texts must not be empty")

    metadata_dict = dict(metadata or {})
    pair_instructions_by_wave = metadata_dict.get("pair_instructions", {})
    ordered_wave_ids = sorted(tagged_texts.keys(), key=lambda x: int(x))
    boundaries = detect_boundaries(plain_text)
    pairs: List[WaveEditPair] = []

    for wave_id in ordered_wave_ids:
        wave_pair_instructions = normalize_pair_instructions_for_wave(
            plain_text,
            wave_id,
            tagged_texts[wave_id],
            (
                pair_instructions_by_wave.get(wave_id)
                if isinstance(pair_instructions_by_wave, dict)
                else None
            ),
            allow_filled_reverse=True,
        )
        for source_id, target_id in (
            (ORIGINAL_WAVE_ID, wave_id),
            (wave_id, ORIGINAL_WAVE_ID),
        ):
            source_text = (
                plain_text if source_id == ORIGINAL_WAVE_ID else tagged_texts[source_id]
            )
            target_text = (
                plain_text if target_id == ORIGINAL_WAVE_ID else tagged_texts[target_id]
            )
            if target_id != ORIGINAL_WAVE_ID:
                is_valid, err = validate_tagged_text(target_text, plain_text)
                if not is_valid:
                    raise ValueError(f"Invalid tagged_text for wave {target_id}: {err}")
            current_pair_key = pair_key(source_id, target_id)
            pairs.append(
                WaveEditPair(
                    source_id=source_id,
                    target_id=target_id,
                    plain_text=plain_text,
                    source_tagged_text=source_text,
                    target_tagged_text=target_text,
                    instructions=wave_pair_instructions[current_pair_key],
                    tag_diff=describe_tag_diff(source_text, target_text),
                )
            )

    return WaveGroup(
        plain_text=plain_text,
        tagged_texts={wave_id: tagged_texts[wave_id] for wave_id in ordered_wave_ids},
        edit_pairs=pairs,
        boundaries=boundaries,
        metadata=metadata_dict,
    )
