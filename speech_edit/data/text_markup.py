"""Tagged-text contract for the text edit pipeline.

The structured instruction itself is the edit plan. ``render_source`` is a
strict projection of the current edit-direction input text. ``render_target``
renders the current edit-direction output text as a normalized surface string:
tag-internal spacing is not part of the contract, so target-side segments are
trimmed and re-spaced at word boundaries.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

ORIGINAL_WAVE_ID = "original"
DEL_TAG = "del"
INS_TAG = "ins"
SUB_TAG = "sub"
EQUAL_TAG = "equal"
VALID_EDIT_KINDS = {DEL_TAG, INS_TAG, SUB_TAG}

_TAG_OPEN_RE = re.compile(r"<(?:del|ins)>|<sub\s+targ=\"[^\"]*\">")
_ANY_TEXT_EDIT_TAG_RE = re.compile(r"</?(?:del|ins|sub)(?:\s+[^>]*)?>")
_QUERY_TAG_RE = re.compile(r"</?q>")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?，。？！；：])")
_APOSTROPHE_SUFFIX_RE = re.compile(r"(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)


class TaggedInstructionError(ValueError):
    """Raised when tagged text cannot satisfy the pipeline contract."""


@dataclass(frozen=True)
class TaggedEditSegment:
    """One ordered segment in a structured text edit instruction."""

    kind: str
    source: str
    target: str
    source_start: int
    source_end: int
    target_start: int
    target_end: int

    @property
    def is_edit(self) -> bool:
        return self.kind in VALID_EDIT_KINDS


def escape_text(text: str) -> str:
    return html.escape(text, quote=True)


def unescape_text(text: str) -> str:
    return html.unescape(text)


def _find_next_tag(text: str, start: int) -> int:
    match = _TAG_OPEN_RE.search(text, start)
    return len(text) if match is None else match.start()


def _reject_residual_edit_tags(text: str) -> None:
    if _ANY_TEXT_EDIT_TAG_RE.search(text):
        raise TaggedInstructionError(
            "Malformed, nested, crossing, or unsupported text edit tag"
        )


def _append_segment(
    segments: List[TaggedEditSegment],
    *,
    kind: str,
    source: str,
    target: str,
    source_pos: int,
    target_pos: int,
) -> Tuple[int, int]:
    segment = TaggedEditSegment(
        kind=kind,
        source=source,
        target=target,
        source_start=source_pos,
        source_end=source_pos + len(source),
        target_start=target_pos,
        target_end=target_pos + len(target),
    )
    segments.append(segment)
    return segment.source_end, segment.target_end


def parse_tagged_instruction(st: str) -> List[TaggedEditSegment]:
    """Parse `<del>`, `<ins>`, and `<sub targ="...">` into ordered segments."""
    segments: List[TaggedEditSegment] = []
    pos = 0
    source_pos = 0
    target_pos = 0

    while pos < len(st):
        if st.startswith("<del>", pos):
            end = st.find("</del>", pos + len("<del>"))
            if end < 0:
                raise TaggedInstructionError("Missing </del> closing tag")
            source = unescape_text(st[pos + len("<del>") : end])
            _reject_residual_edit_tags(source)
            source_pos, target_pos = _append_segment(
                segments,
                kind=DEL_TAG,
                source=source,
                target="",
                source_pos=source_pos,
                target_pos=target_pos,
            )
            pos = end + len("</del>")
            continue

        if st.startswith("<ins>", pos):
            end = st.find("</ins>", pos + len("<ins>"))
            if end < 0:
                raise TaggedInstructionError("Missing </ins> closing tag")
            target = unescape_text(st[pos + len("<ins>") : end])
            _reject_residual_edit_tags(target)
            source_pos, target_pos = _append_segment(
                segments,
                kind=INS_TAG,
                source="",
                target=target,
                source_pos=source_pos,
                target_pos=target_pos,
            )
            pos = end + len("</ins>")
            continue

        if st.startswith("<sub", pos):
            match = re.match(r"<sub\s+targ=\"([^\"]*)\">", st[pos:])
            if match is None:
                raise TaggedInstructionError("Malformed <sub targ=\"...\"> tag")
            open_end = pos + match.end()
            end = st.find("</sub>", open_end)
            if end < 0:
                raise TaggedInstructionError("Missing </sub> closing tag")
            target = unescape_text(match.group(1))
            source = unescape_text(st[open_end:end])
            _reject_residual_edit_tags(source)
            _reject_residual_edit_tags(target)
            source_pos, target_pos = _append_segment(
                segments,
                kind=SUB_TAG,
                source=source,
                target=target,
                source_pos=source_pos,
                target_pos=target_pos,
            )
            pos = end + len("</sub>")
            continue

        next_tag = _find_next_tag(st, pos)
        source = unescape_text(st[pos:next_tag])
        _reject_residual_edit_tags(source)
        if source:
            source_pos, target_pos = _append_segment(
                segments,
                kind=EQUAL_TAG,
                source=source,
                target=source,
                source_pos=source_pos,
                target_pos=target_pos,
            )
        pos = next_tag

    return segments


def render_source(st: str) -> str:
    return "".join(segment.source for segment in parse_tagged_instruction(st))


def _contains_ascii_word_char(text: str) -> bool:
    return bool(_ASCII_WORD_RE.search(text))


def _visible_edge(text: str, *, left: bool) -> str:
    stripped = _visible_text(text)
    if not stripped:
        return ""
    return stripped[0] if left else stripped[-1]


def _visible_text(text: str) -> str:
    return _QUERY_TAG_RE.sub("", text)


def _is_ascii_word_char(char: str) -> bool:
    return bool(char and _ASCII_WORD_RE.fullmatch(char))


def _is_word_internal_join(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if right[0] in {"'", "-", "’"} and len(right) > 1:
        return _is_ascii_word_char(left[-1]) and _is_ascii_word_char(right[1])
    if left[-1] in {"'", "-", "’"} and len(left) > 1:
        if not (_is_ascii_word_char(left[-2]) and _is_ascii_word_char(right[0])):
            return False
        if left[-1] in {"'", "’"}:
            return bool(_APOSTROPHE_SUFFIX_RE.match(right))
        return True
    return False


def _attach_connector_to_left(part_visible: str, left: str, right: str) -> bool:
    if not left or not right or not _is_ascii_word_char(left[-1]):
        return False
    if part_visible == "-":
        return _is_ascii_word_char(right[0])
    if part_visible in {"'", "’"}:
        return bool(_APOSTROPHE_SUFFIX_RE.match(right)) or left[-1] in {"s", "S"}
    return False


def normalize_target_parts(parts: Iterable[str]) -> str:
    """Normalize target-side render parts without trusting tag-boundary spaces."""
    clean_parts = [raw_part.strip() for raw_part in parts]
    rendered: List[str] = []
    for idx, part in enumerate(clean_parts):
        if not part:
            continue
        left_visible = _visible_text(rendered[-1]) if rendered else ""
        right_visible = _visible_text(part)
        next_visible = ""
        for next_part in clean_parts[idx + 1 :]:
            next_visible = _visible_text(next_part)
            if next_visible:
                break
        if rendered and _attach_connector_to_left(
            right_visible, left_visible, next_visible
        ):
            rendered[-1] += part
            continue
        left_edge = _visible_edge(left_visible, left=False)
        right_edge = _visible_edge(right_visible, left=True)
        needs_space = (
            rendered
            and not _is_word_internal_join(left_visible, right_visible)
            and (
                _contains_ascii_word_char(left_edge)
                or _contains_ascii_word_char(right_edge)
            )
        )
        if needs_space:
            rendered.append(" ")
        rendered.append(part)
    text = "".join(rendered)
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def render_target(st: str) -> str:
    return normalize_target_parts(
        segment.target for segment in parse_tagged_instruction(st)
    )


def render_segment(kind: str, source: str, target: str = "") -> str:
    if kind == EQUAL_TAG:
        return escape_text(source)
    if kind == DEL_TAG:
        return f"<del>{escape_text(source)}</del>"
    if kind == INS_TAG:
        return f"<ins>{escape_text(target)}</ins>"
    if kind == SUB_TAG:
        return f"<sub targ=\"{escape_text(target)}\">{escape_text(source)}</sub>"
    raise TaggedInstructionError(f"Unknown segment kind: {kind}")


def render_segments(segments: Iterable[TaggedEditSegment]) -> str:
    return "".join(
        render_segment(segment.kind, segment.source, segment.target)
        for segment in segments
    )


def _target_core_spans(
    segments: List[TaggedEditSegment],
) -> List[Optional[Tuple[int, int]]]:
    spans: List[Optional[Tuple[int, int]]] = []
    canonical_target = normalize_target_parts(segment.target for segment in segments)
    cursor = 0
    for segment in segments:
        core = _SPACE_BEFORE_PUNCT_RE.sub(
            r"\1", re.sub(r"\s+", " ", segment.target.strip())
        )
        if not core:
            spans.append(None)
            continue
        start = canonical_target.find(core, cursor)
        if start < 0:
            raise TaggedInstructionError("Unable to map target segment span")
        end = start + len(core)
        spans.append((start, end))
        cursor = end
    return spans


def render_target_with_segment_spans(
    st: str,
) -> Tuple[str, List[Optional[Tuple[int, int]]]]:
    """Render the canonical target and map each parsed segment into it."""
    segments = parse_tagged_instruction(st)
    return (
        normalize_target_parts(segment.target for segment in segments),
        _target_core_spans(segments),
    )


def reverse_tagged_instruction(st: str) -> str:
    forward_segments = parse_tagged_instruction(st)
    canonical_target = render_target(st)
    target_spans = _target_core_spans(forward_segments)
    reverse_segments: List[Dict[str, str]] = []
    cursor = 0
    previous_source_index: Optional[int] = None

    for segment in forward_segments:
        if segment.kind == EQUAL_TAG:
            reverse_segments.append(
                {"kind": EQUAL_TAG, "source": "", "target": ""}
            )
        elif segment.kind == DEL_TAG:
            reverse_segments.append(
                {"kind": INS_TAG, "source": "", "target": segment.source}
            )
        elif segment.kind == INS_TAG:
            reverse_segments.append(
                {"kind": DEL_TAG, "source": "", "target": ""}
            )
        elif segment.kind == SUB_TAG:
            reverse_segments.append(
                {"kind": SUB_TAG, "source": "", "target": segment.source}
            )
        else:  # pragma: no cover - dataclass invariant
            raise TaggedInstructionError(f"Unknown segment kind: {segment.kind}")

    for index, span in enumerate(target_spans):
        if span is None:
            continue
        start, end = span
        gap = canonical_target[cursor:start]
        if gap:
            if forward_segments[index].kind == EQUAL_TAG:
                reverse_segments[index]["source"] += gap
            elif previous_source_index is not None:
                reverse_segments[previous_source_index]["source"] += gap
            else:
                reverse_segments[index]["source"] += gap
        reverse_segments[index]["source"] += canonical_target[start:end]
        cursor = end
        previous_source_index = index

    if previous_source_index is not None:
        reverse_segments[previous_source_index]["source"] += canonical_target[cursor:]
    return "".join(
        render_segment(segment["kind"], segment["source"], segment["target"])
        for segment in reverse_segments
    )


def validate_tagged_instruction(st: str, source_text: str, target_text: str) -> None:
    source = render_source(st)
    target = render_target(st)
    if source != source_text:
        raise TaggedInstructionError(
            "Tagged instruction source render mismatch:\n"
            f"expected={source_text!r}\nactual={source!r}\nst={st!r}"
        )
    if target != target_text:
        raise TaggedInstructionError(
            "Tagged instruction target render mismatch:\n"
            f"expected={target_text!r}\nactual={target!r}\nst={st!r}"
        )


def edit_segments(st: str) -> List[TaggedEditSegment]:
    return [segment for segment in parse_tagged_instruction(st) if segment.is_edit]


def op_counts(st: str) -> Dict[str, int]:
    counts = {DEL_TAG: 0, INS_TAG: 0, SUB_TAG: 0}
    for segment in edit_segments(st):
        counts[segment.kind] += 1
    return counts


def tag_query(text: str, start: int, end: int) -> str:
    start = max(0, min(len(text), start))
    end = max(start, min(len(text), end))
    return f"{text[:start]}<q>{text[start:end]}</q>{text[end:]}"
