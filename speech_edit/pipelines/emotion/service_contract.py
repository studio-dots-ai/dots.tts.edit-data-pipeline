"""Service-facing contract helpers for emotion edit requests.

The dataset pipeline stores fully tagged source/target variants. The service
accepts a lighter partial instruction that only marks target emotion spans, so
this module turns that request shape into the full tagged/assembly contract
used by Stage II and Stage III.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .tag_utils import (
    EMOTION_CONTROL_DESCRIPTION,
    NEUTRAL_TAG,
    TaggedSegment,
    TextBoundary,
    build_tagged_text,
    parse_tags,
    texts_equivalent_ignoring_safe_whitespace,
)


_ANY_TAG_PATTERN = re.compile(r"</?[a-zA-Z_][^>]*>")


@dataclass(frozen=True)
class EmotionEditServiceContract:
    """Normalized service request contract."""

    plain_text: str
    source_tagged_text: str
    target_tagged_text: str
    edit_tagged_text: str
    changed_segments: List[Dict[str, Any]]
    target_spans: List[Dict[str, Any]]
    synthesis_sub_type: str
    boundaries: List[TextBoundary]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _reject_residual_markup(segment: TaggedSegment) -> None:
    if _ANY_TAG_PATTERN.search(segment.text):
        raise ValueError(
            "Nested, malformed, or unsupported tags are not allowed in instruction_st"
        )


def _segment_to_tagged_tuple(
    segment: TaggedSegment,
) -> tuple[Optional[str], str, Optional[float], Optional[str], Optional[float], Optional[str]]:
    if segment.tag_name == NEUTRAL_TAG:
        return NEUTRAL_TAG, segment.text, None, None, segment.dur, None
    if segment.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
        return (
            None,
            segment.text,
            None,
            segment.emotion_desc,
            segment.dur,
            segment.emotion_type_ref,
        )
    return segment.evaluation_tag, segment.text, segment.effective_alpha, None, segment.dur, None


def _target_markup(text: str) -> str:
    return f"<target>{text}</target>"


def build_service_edit_contract(
    *,
    source_text: str,
    instruction_st: str,
) -> EmotionEditServiceContract:
    """Build the full emotion edit contract from a partial service instruction.

    ``instruction_st`` may leave unchanged spans untagged. Emotion-tagged spans
    mark the target delivery and become ``<target>`` spans for audio assembly.
    Unchanged spans must remain untagged in the external service request.
    ``<emo type="none">`` is an internal full-tagged representation and is
    rejected here.
    """
    parsed_segments = parse_tags(instruction_st)
    if not parsed_segments:
        raise ValueError("instruction_st must contain transcript content")
    for segment in parsed_segments:
        _reject_residual_markup(segment)

    plain_text = "".join(segment.text for segment in parsed_segments)
    if not texts_equivalent_ignoring_safe_whitespace(source_text, plain_text):
        raise ValueError(
            "instruction_st must preserve source_text after stripping tags. "
            f"expected={source_text!r}, got={plain_text!r}"
        )

    target_segments: List[
        tuple[Optional[str], str, Optional[float], Optional[str], Optional[float], Optional[str]]
    ] = []
    source_segments: List[
        tuple[Optional[str], str, Optional[float], Optional[str], Optional[float], Optional[str]]
    ] = []
    edit_parts: List[str] = []
    changed_segments: List[Dict[str, Any]] = []
    target_spans: List[Dict[str, Any]] = []
    boundaries: List[TextBoundary] = []
    char_pos = 0
    full_segment_idx = 0

    for segment in parsed_segments:
        if not segment.text:
            continue

        start_char = char_pos
        end_char = start_char + len(segment.text)
        char_pos = end_char

        source_segments.append((NEUTRAL_TAG, segment.text, None, None, None, None))

        if segment.tag_name is None:
            target_segments.append((NEUTRAL_TAG, segment.text, None, None, None, None))
            edit_parts.append(segment.text)
        elif segment.tag_name == NEUTRAL_TAG:
            raise ValueError(
                '<emo type="none"> is an internal tag and is not allowed in instruction_st; '
                "leave unchanged spans untagged"
            )
        elif segment.tag_type == "emotion":
            target_segments.append(_segment_to_tagged_tuple(segment))
            edit_parts.append(_target_markup(segment.text))
            changed_segments.append(
                {
                    "segment_idx": full_segment_idx,
                    "segment_text": segment.text.strip(),
                    "source_tag": NEUTRAL_TAG,
                    "target_tag": segment.evaluation_tag,
                    "source_alpha": None,
                    "target_alpha": segment.effective_alpha,
                    "target_emotion_control_mode": segment.emotion_control_mode,
                    "target_emotion_desc": segment.emotion_desc,
                    "target_emotion_type_ref": segment.emotion_type_ref,
                    "source_dur": None,
                    "target_dur": segment.dur,
                }
            )
            target_spans.append(
                {
                    "segment_idx": full_segment_idx,
                    "text": segment.text.strip(),
                    "start_char": start_char,
                    "end_char": end_char,
                    "tag": segment.evaluation_tag,
                    "alpha": segment.effective_alpha,
                    "emotion_control_mode": segment.emotion_control_mode,
                    "emotion_desc": segment.emotion_desc,
                    "emotion_type_ref": segment.emotion_type_ref,
                    "dur": segment.dur,
                }
            )
        else:
            raise ValueError(
                f"Unsupported tag {segment.tag_name!r}; emotion edit service only "
                "supports canonical <emo ...> tags; leave unchanged spans untagged"
            )

        if full_segment_idx > 0:
            boundaries.append(TextBoundary(char_idx=start_char))
        full_segment_idx += 1

    if not changed_segments:
        raise ValueError("instruction_st must mark at least one emotion edit span")

    target_tagged_text = build_tagged_text(target_segments)
    source_tagged_text = build_tagged_text(source_segments)
    edit_tagged_text = "".join(edit_parts)
    synthesis_sub_type = (
        "single"
        if len(changed_segments) == 1
        and _normalize_text(changed_segments[0]["segment_text"]) == _normalize_text(source_text)
        else "multi_emo_frequent"
    )
    return EmotionEditServiceContract(
        plain_text=plain_text,
        source_tagged_text=source_tagged_text,
        target_tagged_text=target_tagged_text,
        edit_tagged_text=edit_tagged_text,
        changed_segments=changed_segments,
        target_spans=target_spans,
        synthesis_sub_type=synthesis_sub_type,
        boundaries=boundaries,
    )
