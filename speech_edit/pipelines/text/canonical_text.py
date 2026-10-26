"""Canonical text projection for tagged text edit instructions."""

from __future__ import annotations

import bisect
import string
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from speech_edit.data.text_markup import (
    DEL_TAG,
    EQUAL_TAG,
    INS_TAG,
    SUB_TAG,
    TaggedEditSegment,
    TaggedInstructionError,
    edit_segments,
    parse_tagged_instruction,
    render_segment,
    render_source,
)


@dataclass(frozen=True)
class CanonicalText:
    text: str
    char_spans: List[Tuple[int, int]]
    original_text: str


ASCII_WORD_CHARS = set(string.ascii_letters + string.digits)


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _is_ascii_word(char: str) -> bool:
    return char in ASCII_WORD_CHARS


def _should_keep_space(prev_char: str | None, next_char: str | None) -> bool:
    return bool(prev_char and next_char and _is_ascii_word(prev_char) and _is_ascii_word(next_char))


def canonicalize_text(text: str) -> CanonicalText:
    """Normalize whitespace while keeping a char map back to ``text``."""
    chars: List[str] = []
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace():
            start = i
            while i < len(text) and text[i].isspace():
                i += 1
            prev_char = chars[-1] if chars else None
            next_char = text[i] if i < len(text) else None
            if _should_keep_space(prev_char, next_char):
                if not chars or chars[-1] != " ":
                    chars.append(" ")
                    spans.append((start, i))
            continue
        chars.append(char)
        spans.append((i, i + 1))
        i += 1
    return CanonicalText("".join(chars), spans, text)


def _canonical_span_for_raw_span(canonical: CanonicalText, start: int, end: int) -> Tuple[int, int]:
    if start == end:
        return _canonical_boundary_for_raw_offset(canonical, start), _canonical_boundary_for_raw_offset(canonical, end)
    indexes = [
        idx
        for idx, (span_start, span_end) in enumerate(canonical.char_spans)
        if span_start < end and span_end > start
    ]
    if not indexes:
        raise TaggedInstructionError("Tagged edit source span maps only to ignored whitespace")
    return indexes[0], indexes[-1] + 1


def _canonical_boundary_for_raw_offset(canonical: CanonicalText, offset: int) -> int:
    ends = [span_end for _, span_end in canonical.char_spans]
    return bisect.bisect_right(ends, offset)


def canonical_span_to_surface_span(
    canonical_surface: CanonicalText,
    start: int,
    end: int,
) -> Tuple[int, int]:
    if start < 0 or end < start or end > len(canonical_surface.text):
        raise TaggedInstructionError(
            f"Canonical span out of range: [{start}, {end}] for {len(canonical_surface.text)} chars"
        )
    if start == end:
        if not canonical_surface.char_spans:
            offset = len(canonical_surface.original_text)
        elif start == 0:
            offset = canonical_surface.char_spans[0][0]
        elif start >= len(canonical_surface.char_spans):
            offset = canonical_surface.char_spans[-1][1]
        else:
            offset = canonical_surface.char_spans[start][0]
        return offset, offset
    return canonical_surface.char_spans[start][0], canonical_surface.char_spans[end - 1][1]


def _project_edit_segments(
    canonical_st: str,
    surface_source: str,
) -> List[Tuple[TaggedEditSegment, int, int]]:
    canonical_surface = canonicalize_text(surface_source)
    llm_source = render_source(canonical_st)
    llm_canonical = canonicalize_text(llm_source)
    if llm_canonical.text != canonical_surface.text:
        raise TaggedInstructionError(
            "Canonical source render mismatch:\n"
            f"expected={canonical_surface.text!r}\nactual={llm_canonical.text!r}\nst={canonical_st!r}"
        )

    projected: List[Tuple[TaggedEditSegment, int, int]] = []
    for segment in edit_segments(canonical_st):
        canonical_start, canonical_end = _canonical_span_for_raw_span(
            llm_canonical,
            segment.source_start,
            segment.source_end,
        )
        surface_start, surface_end = canonical_span_to_surface_span(
            canonical_surface,
            canonical_start,
            canonical_end,
        )
        projected.append((segment, surface_start, surface_end))
    return projected


def project_tagged_instruction_to_surface(canonical_st: str, surface_source: str) -> str:
    """Render a strict surface-source tagged instruction from canonical LLM tags."""
    projected = _project_edit_segments(canonical_st, surface_source)
    projected.sort(key=lambda item: (item[1], item[2]))
    parts: List[str] = []
    cursor = 0
    for segment, start, end in projected:
        if start < cursor:
            raise TaggedInstructionError("Projected edit spans overlap or cross")
        if start > cursor:
            parts.append(render_segment(EQUAL_TAG, surface_source[cursor:start]))
        source = surface_source[start:end]
        if segment.kind == DEL_TAG:
            parts.append(render_segment(DEL_TAG, source))
        elif segment.kind == INS_TAG:
            parts.append(render_segment(INS_TAG, "", segment.target))
        elif segment.kind == SUB_TAG:
            parts.append(render_segment(SUB_TAG, source, segment.target))
        else:  # pragma: no cover - edit_segments filters this
            raise TaggedInstructionError(f"Unsupported edit kind: {segment.kind}")
        cursor = end
    if cursor < len(surface_source):
        parts.append(render_segment(EQUAL_TAG, surface_source[cursor:]))
    return "".join(parts)


def validate_canonical_sentence_edit(canonical_st: str, canonical_source: str) -> None:
    edits = edit_segments(canonical_st)
    if len(edits) != 1:
        raise TaggedInstructionError("Sentence generation must contain exactly one edit")
    edit = edits[0]
    if edit.kind != SUB_TAG:
        raise TaggedInstructionError("Sentence generation must use a single sub edit")
    source_canonical = canonicalize_text(render_source(canonical_st))
    start, end = _canonical_span_for_raw_span(
        source_canonical,
        edit.source_start,
        edit.source_end,
    )
    if start != 0 or end != len(canonical_source):
        raise TaggedInstructionError("Sentence generation must replace the full canonical source")


def validate_generation_op_count(canonical_st: str, *, expected_count: int) -> None:
    actual_count = len(edit_segments(canonical_st))
    if actual_count != expected_count:
        raise TaggedInstructionError(
            f"Generation has {actual_count} edits; expected {expected_count}"
        )


def has_target_double_ascii_space(st: str) -> bool:
    # Target may preserve original doubled spaces in untouched regions; this is only
    # exposed as a diagnostic helper for tests and summary metrics.
    from speech_edit.data.text_markup import render_target

    return "  " in render_target(st)

