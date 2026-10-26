"""Word-level Dynamic Time Warping for speech-edit temporal fidelity.

The AST paper defines WDTW as DTW over forced-aligned `(word, duration)`
sequences, normalized by sequence duration. The local distance is not specified
in the paper, so this module uses a transparent duration-based cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import math
import re
import unicodedata
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from speech_edit.data.text_markup import (
    VALID_EDIT_KINDS,
    parse_tagged_instruction,
    render_source,
    render_target,
    render_target_with_segment_spans,
)


@dataclass(frozen=True)
class WordSegment:
    word: str
    start: float
    end: float
    raw_word: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass(frozen=True)
class WDTWResult:
    wdtw: Optional[float]
    distance: float
    normalizer: float
    mode: str
    status: str
    source_segments: List[WordSegment]
    target_segments: List[WordSegment]
    preserved_pairs: List[Tuple[int, int]]

    def to_dict(self, include_segments: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "wdtw": self.wdtw,
            "distance": self.distance,
            "normalizer": self.normalizer,
            "mode": self.mode,
            "status": self.status,
            "num_source_segments": len(self.source_segments),
            "num_target_segments": len(self.target_segments),
            "num_preserved_pairs": len(self.preserved_pairs),
            "preserved_pairs": [list(pair) for pair in self.preserved_pairs],
        }
        if include_segments:
            data["source_segments"] = [seg.to_dict() for seg in self.source_segments]
            data["target_segments"] = [seg.to_dict() for seg in self.target_segments]
        return data


@dataclass(frozen=True)
class WDTWDurationResult:
    wdtw_dur: Optional[float]
    distance: float
    normalizer: float
    mode: str
    status: str
    source_segments: List[WordSegment]
    target_segments: List[WordSegment]
    preserved_pairs: List[Tuple[int, int]]

    def to_dict(self, include_segments: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "wdtw_dur": self.wdtw_dur,
            "distance": self.distance,
            "normalizer": self.normalizer,
            "mode": self.mode,
            "status": self.status,
            "num_source_segments": len(self.source_segments),
            "num_target_segments": len(self.target_segments),
            "num_preserved_pairs": len(self.preserved_pairs),
            "preserved_pairs": [list(pair) for pair in self.preserved_pairs],
        }
        if include_segments:
            data["source_segments"] = [seg.to_dict() for seg in self.source_segments]
            data["target_segments"] = [seg.to_dict() for seg in self.target_segments]
        return data


@dataclass(frozen=True)
class WDTWF0Result:
    wdtw_f0: Optional[float]
    distance: float
    normalizer: float
    status: str
    eligible_word_count: int
    valid_word_count: int
    skipped_word_count: int
    word_results: List[Dict[str, Any]]
    error: Optional[str] = None
    unit: str = "semitones"

    def to_dict(self, include_words: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "wdtw_f0": self.wdtw_f0,
            "distance": self.distance,
            "normalizer": self.normalizer,
            "status": self.status,
            "unit": self.unit,
            "eligible_word_count": self.eligible_word_count,
            "valid_word_count": self.valid_word_count,
            "skipped_word_count": self.skipped_word_count,
        }
        if self.error:
            data["error"] = self.error
        if include_words:
            data["word_results"] = self.word_results
        return data


@dataclass(frozen=True)
class WDTWAggregateResult:
    wdtw: Optional[float]
    distance: float
    normalizer: float
    mode: str
    status: str
    count: int
    valid_count: int
    skipped_count: int
    skipped_no_segments: int
    skipped_missing: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WDTWDurationAggregateResult:
    wdtw_dur: Optional[float]
    distance: float
    normalizer: float
    mode: str
    status: str
    count: int
    valid_count: int
    skipped_count: int
    skipped_no_segments: int
    skipped_missing: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WDTWF0AggregateResult:
    wdtw_f0: Optional[float]
    distance: float
    normalizer: float
    status: str
    count: int
    valid_count: int
    skipped_count: int
    skipped_no_valid_f0: int
    skipped_no_preserved_words: int
    skipped_unavailable: int
    skipped_error: int
    skipped_other_invalid: int
    skipped_missing: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
    )


def normalize_word(text: Any) -> str:
    """Normalize a word/token for transcript and alignment matching."""
    value = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    kept: List[str] = []
    for ch in value:
        if ch.isalnum() or _is_cjk(ch):
            kept.append(ch)
    return "".join(kept)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into normalized word-like units.

    ASCII alphanumeric runs become one token. CJK characters are emitted as
    individual tokens because the available forced aligners may return
    character-like units for Chinese.
    """
    tokens: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            token = normalize_word("".join(current))
            if token:
                tokens.append(token)
            current.clear()

    for ch in unicodedata.normalize("NFKC", text or ""):
        if _is_cjk(ch):
            flush()
            tokens.append(ch)
        elif ch.isalnum():
            current.append(ch.lower())
        else:
            flush()
    flush()
    return tokens


def lcs_pairs(source: Sequence[str], target: Sequence[str]) -> List[Tuple[int, int]]:
    """Return duplicate-safe LCS index pairs in reading order."""
    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if source[i] == target[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    pairs: List[Tuple[int, int]] = []
    i = j = 0
    while i < m and j < n:
        if source[i] == target[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


@dataclass(frozen=True)
class EditSTInstruction:
    source_text: str
    target_text: str
    source_edited_mask: List[bool]
    target_edited_mask: List[bool]


_TAG_RE = re.compile(
    r"<(?P<closing>/)?(?P<name>[A-Za-z_][\w:-]*)(?P<body>[^<>]*?)(?P<selfclose>/)?>",
    re.DOTALL,
)
_SUB_TARGET_RE = re.compile(
    r"""(?:^|[\s,])targ\s*=\s*(?:"([^"]*)"|'([^']*)')""",
    re.DOTALL,
)
_TEXT_EDIT_TAG_NAMES = {"del", "ins", "sub"}


def _append_text_tokens(
    parts: List[str],
    mask: List[bool],
    text: str,
    *,
    edited: bool,
) -> None:
    parts.append(text)
    mask.extend([edited] * len(text))


def _reject_residual_tags(text: str) -> None:
    if _TAG_RE.search(text):
        raise ValueError("Nested, crossing, malformed, or unsupported edit_st_inst tag")


def _parse_sub_target(open_body: str) -> str:
    match = _SUB_TARGET_RE.search(open_body)
    if match is None:
        raise ValueError("Malformed <sub> tag: missing targ attribute")
    return html.unescape(match.group(1) if match.group(1) is not None else match.group(2))


def _parse_edit_st_inst_generic(
    edit_st_inst: str,
    *,
    unmasked_paired_tags: Collection[str] = (),
    mask_text_edits: bool = True,
) -> EditSTInstruction:
    """Parse structured edit markup into source/target text and edited masks.

    Text outside tags is preserved. Text edit tags follow the tagged text edit
    pipeline contract (`del`, `ins`, `sub targ="..."`). Any other paired tag is
    treated as a paralinguistic edit over its inner text. Self-closing tags are
    point edits and do not mask any word tokens.
    """
    unmasked_tags = {str(name).lower() for name in unmasked_paired_tags}
    source_parts: List[str] = []
    target_parts: List[str] = []
    source_mask: List[bool] = []
    target_mask: List[bool] = []
    pos = 0

    while pos < len(edit_st_inst):
        match = _TAG_RE.search(edit_st_inst, pos)
        if match is None:
            text = html.unescape(edit_st_inst[pos:])
            _append_text_tokens(source_parts, source_mask, text, edited=False)
            _append_text_tokens(target_parts, target_mask, text, edited=False)
            break

        prefix = html.unescape(edit_st_inst[pos : match.start()])
        _append_text_tokens(source_parts, source_mask, prefix, edited=False)
        _append_text_tokens(target_parts, target_mask, prefix, edited=False)

        if match.group("closing"):
            raise ValueError("Unexpected closing tag in edit_st_inst")

        name = match.group("name").lower()
        open_body = match.group("body") or ""
        if match.group("selfclose"):
            pos = match.end()
            continue

        close_re = re.compile(rf"</{re.escape(match.group('name'))}\s*>", re.DOTALL)
        close_match = close_re.search(edit_st_inst, match.end())
        if close_match is None:
            raise ValueError(f"Missing closing tag for <{match.group('name')}>")

        inner_raw = edit_st_inst[match.end() : close_match.start()]
        _reject_residual_tags(inner_raw)
        inner = html.unescape(inner_raw)

        if name == "del":
            _append_text_tokens(
                source_parts, source_mask, inner, edited=mask_text_edits
            )
        elif name == "ins":
            _append_text_tokens(
                target_parts, target_mask, inner, edited=mask_text_edits
            )
        elif name == "sub":
            target = _parse_sub_target(open_body)
            _reject_residual_tags(target)
            _append_text_tokens(
                source_parts, source_mask, inner, edited=mask_text_edits
            )
            _append_text_tokens(
                target_parts, target_mask, target, edited=mask_text_edits
            )
        else:
            edited = name not in unmasked_tags
            _append_text_tokens(source_parts, source_mask, inner, edited=edited)
            _append_text_tokens(target_parts, target_mask, inner, edited=edited)

        pos = close_match.end()

    source_text = "".join(source_parts)
    target_text = "".join(target_parts)
    return EditSTInstruction(
        source_text=source_text,
        target_text=target_text,
        source_edited_mask=_mask_tokens_from_char_flags(source_text, source_mask),
        target_edited_mask=_mask_tokens_from_char_flags(target_text, target_mask),
    )


def _strip_non_text_edit_tags(edit_st_inst: str) -> str:
    """Keep text-edit markup while making other edit tags surface-transparent."""
    parts: List[str] = []
    pos = 0
    for match in _TAG_RE.finditer(edit_st_inst):
        parts.append(edit_st_inst[pos : match.start()])
        if match.group("name").lower() in _TEXT_EDIT_TAG_NAMES:
            parts.append(match.group(0))
        pos = match.end()
    parts.append(edit_st_inst[pos:])
    return "".join(parts)


def _token_spans(text: str) -> List[Tuple[int, int]]:
    """Return spans matching ``tokenize_text`` for renderer-output masking."""
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, char in enumerate(text):
        if _is_cjk(char):
            if start is not None:
                spans.append((start, index))
                start = None
            spans.append((index, index + 1))
        elif char.isalnum():
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _mask_tokens_from_char_ranges(
    text: str,
    edited_ranges: Sequence[Tuple[int, int]],
) -> List[bool]:
    char_flags = [
        any(start <= index < end for start, end in edited_ranges)
        for index in range(len(text))
    ]
    return _mask_tokens_from_char_flags(text, char_flags)


def _mask_tokens_from_char_flags(
    text: str,
    char_flags: Sequence[bool],
) -> List[bool]:
    if len(text) != len(char_flags):
        raise ValueError("Character edit mask does not match rendered text")
    normalized_parts: List[str] = []
    normalized_edited: List[bool] = []
    for char, is_edited in zip(text, char_flags):
        normalized_char = unicodedata.normalize("NFKC", char)
        normalized_parts.append(normalized_char)
        normalized_edited.extend([is_edited] * len(normalized_char))
    normalized_text = "".join(normalized_parts)
    token_spans = _token_spans(normalized_text)
    span_tokens = [
        normalize_word(normalized_text[start:end]) for start, end in token_spans
    ]
    assert span_tokens == tokenize_text(text)
    token_mask = [
        any(normalized_edited[start:end]) for start, end in token_spans
    ]
    for edited_index, is_edited in enumerate(normalized_edited):
        if (
            not is_edited
            or normalized_text[edited_index] not in {"'", "’", "-"}
        ):
            continue
        for token_index, (start, end) in enumerate(token_spans):
            if end == edited_index or start == edited_index + 1:
                token_mask[token_index] = True
    return token_mask


def _text_edit_surfaces_and_masks(edit_st_inst: str) -> EditSTInstruction:
    """Render text edits with the pipeline contract and derive metric masks."""
    text_edit_st = _strip_non_text_edit_tags(edit_st_inst)
    segments = parse_tagged_instruction(text_edit_st)
    source_text = render_source(text_edit_st)
    target_text, target_segment_spans = render_target_with_segment_spans(text_edit_st)

    source_ranges = [
        (segment.source_start, segment.source_end)
        for segment in segments
        if segment.kind in VALID_EDIT_KINDS and segment.source_start < segment.source_end
    ]
    target_ranges = [
        span
        for segment, span in zip(segments, target_segment_spans)
        if segment.kind in VALID_EDIT_KINDS and span is not None
    ]

    return EditSTInstruction(
        source_text=source_text,
        target_text=target_text,
        source_edited_mask=_mask_tokens_from_char_ranges(source_text, source_ranges),
        target_edited_mask=_mask_tokens_from_char_ranges(target_text, target_ranges),
    )


def _merge_generic_masks(
    *,
    rendered: EditSTInstruction,
    generic: EditSTInstruction,
) -> EditSTInstruction:
    """Preserve non-text tag masks after renderer boundary normalization."""

    def merge_side(
        rendered_text: str,
        rendered_mask: Sequence[bool],
        generic_text: str,
        generic_mask: Sequence[bool],
    ) -> List[bool]:
        generic_tokens = tokenize_text(generic_text)
        rendered_tokens = tokenize_text(rendered_text)
        merged = list(rendered_mask)
        if len(generic_tokens) != len(generic_mask):
            return merged
        if "".join(generic_tokens) == "".join(rendered_tokens):
            rendered_offsets: List[Tuple[int, int]] = []
            cursor = 0
            for token in rendered_tokens:
                rendered_offsets.append((cursor, cursor + len(token)))
                cursor += len(token)
            cursor = 0
            for generic_token, is_edited in zip(generic_tokens, generic_mask):
                generic_start = cursor
                generic_end = cursor + len(generic_token)
                cursor = generic_end
                if not is_edited:
                    continue
                for rendered_index, (rendered_start, rendered_end) in enumerate(
                    rendered_offsets
                ):
                    if rendered_start < generic_end and generic_start < rendered_end:
                        merged[rendered_index] = True
            return merged
        for generic_index, rendered_index in lcs_pairs(generic_tokens, rendered_tokens):
            merged[rendered_index] = merged[rendered_index] or generic_mask[generic_index]
        return merged

    return EditSTInstruction(
        source_text=rendered.source_text,
        target_text=rendered.target_text,
        source_edited_mask=merge_side(
            rendered.source_text,
            rendered.source_edited_mask,
            generic.source_text,
            generic.source_edited_mask,
        ),
        target_edited_mask=merge_side(
            rendered.target_text,
            rendered.target_edited_mask,
            generic.target_text,
            generic.target_edited_mask,
        ),
    )


def parse_edit_st_inst(
    edit_st_inst: str,
    *,
    unmasked_paired_tags: Collection[str] = (),
) -> EditSTInstruction:
    """Parse edit markup, delegating text-edit rendering to the pipeline."""
    tag_names = {match.group("name").lower() for match in _TAG_RE.finditer(edit_st_inst)}
    rendered = None
    if tag_names.intersection(_TEXT_EDIT_TAG_NAMES):
        # Fail on the shared text-edit contract before considering the generic
        # tag path; malformed text edits must never silently use old semantics.
        rendered = _text_edit_surfaces_and_masks(edit_st_inst)
    generic = _parse_edit_st_inst_generic(
        edit_st_inst,
        unmasked_paired_tags=unmasked_paired_tags,
        mask_text_edits=rendered is None,
    )
    if rendered is None:
        return generic
    return _merge_generic_masks(rendered=rendered, generic=generic)


def expected_preserved_pair_count(
    source_transcript: str,
    target_transcript: str,
    edit_st_inst: str,
    *,
    unmasked_paired_tags: Collection[str] = (),
) -> int:
    """Return the instruction-defined number of preserved transcript pairs."""

    instruction = parse_edit_st_inst(
        edit_st_inst,
        unmasked_paired_tags=unmasked_paired_tags,
    )
    source_tokens = tokenize_text(source_transcript)
    target_tokens = tokenize_text(target_transcript)
    if source_tokens != tokenize_text(instruction.source_text):
        raise ValueError("source transcript does not match frozen instruction")
    if target_tokens != tokenize_text(instruction.target_text):
        raise ValueError("target transcript does not match frozen instruction")
    return sum(
        not instruction.source_edited_mask[source_index]
        and not instruction.target_edited_mask[target_index]
        for source_index, target_index in lcs_pairs(source_tokens, target_tokens)
    )


def _item_text(item: Mapping[str, Any]) -> str:
    for key in ("word", "char", "text", "token"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def alignment_to_segments(alignments: Iterable[Mapping[str, Any]]) -> List[WordSegment]:
    """Convert aligner response entries to normalized word-duration segments."""
    segments: List[WordSegment] = []
    for item in alignments or []:
        if item.get("is_punctuation") is True:
            continue
        raw_word = _item_text(item)
        tokens = tokenize_text(raw_word)
        if not tokens:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Alignment item missing numeric start/end: {item}") from exc
        if end < start:
            raise ValueError(f"Alignment item has end before start: {item}")

        if len(tokens) == 1:
            segments.append(
                WordSegment(word=tokens[0], start=start, end=end, raw_word=raw_word)
            )
            continue

        # If a backend returns a multi-word chunk, split duration evenly so the
        # caller can still match transcript tokens in order.
        step = (end - start) / len(tokens) if tokens else 0.0
        for idx, token in enumerate(tokens):
            seg_start = start + idx * step
            seg_end = start + (idx + 1) * step
            segments.append(
                WordSegment(word=token, start=seg_start, end=seg_end, raw_word=raw_word)
            )
    return segments


def _map_text_tokens_to_segments(
    text_tokens: Sequence[str], segments: Sequence[WordSegment]
) -> Dict[int, int]:
    segment_tokens = [seg.word for seg in segments]
    return {text_idx: seg_idx for text_idx, seg_idx in lcs_pairs(text_tokens, segment_tokens)}


def _local_cost(
    source: WordSegment,
    target: WordSegment,
    *,
    word_mismatch_penalty: float,
) -> float:
    duration_cost = abs(source.duration - target.duration)
    if source.word == target.word:
        return duration_cost
    return source.duration + target.duration + word_mismatch_penalty


def dtw_distance(
    source: Sequence[WordSegment],
    target: Sequence[WordSegment],
    *,
    word_mismatch_penalty: float = 1.0,
) -> float:
    """Compute DTW distance over word-duration segments."""
    if not source and not target:
        return 0.0
    if not source:
        return sum(seg.duration + word_mismatch_penalty for seg in target)
    if not target:
        return sum(seg.duration + word_mismatch_penalty for seg in source)

    rows = len(source) + 1
    cols = len(target) + 1
    dp = [[math.inf] * cols for _ in range(rows)]
    dp[0][0] = 0.0
    for i in range(1, rows):
        for j in range(1, cols):
            cost = _local_cost(
                source[i - 1],
                target[j - 1],
                word_mismatch_penalty=word_mismatch_penalty,
            )
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return float(dp[-1][-1])


def _select_segments(
    source_transcript: str,
    target_transcript: str,
    source_segments_all: Sequence[WordSegment],
    target_segments_all: Sequence[WordSegment],
    *,
    mode: str,
    edit_st_inst: Optional[str] = None,
    unmasked_paired_tags: Collection[str] = (),
) -> Tuple[List[WordSegment], List[WordSegment], List[Tuple[int, int]]]:
    if mode == "all":
        return list(source_segments_all), list(target_segments_all), []
    if mode != "preserved":
        raise ValueError(f"Unsupported WDTW mode: {mode}")

    source_tokens = tokenize_text(source_transcript)
    target_tokens = tokenize_text(target_transcript)
    source_edited_mask = [False] * len(source_tokens)
    target_edited_mask = [False] * len(target_tokens)
    if edit_st_inst:
        instruction = parse_edit_st_inst(
            edit_st_inst,
            unmasked_paired_tags=unmasked_paired_tags,
        )
        inst_source_tokens = tokenize_text(instruction.source_text)
        inst_target_tokens = tokenize_text(instruction.target_text)
        if inst_source_tokens != source_tokens:
            raise ValueError(
                "edit_st_inst source text does not match source_transcript tokens"
            )
        if inst_target_tokens != target_tokens:
            raise ValueError(
                "edit_st_inst target text does not match target_transcript tokens"
            )
        source_edited_mask = instruction.source_edited_mask
        target_edited_mask = instruction.target_edited_mask

    transcript_pairs = lcs_pairs(source_tokens, target_tokens)
    source_map = _map_text_tokens_to_segments(source_tokens, source_segments_all)
    target_map = _map_text_tokens_to_segments(target_tokens, target_segments_all)

    source_selected: List[WordSegment] = []
    target_selected: List[WordSegment] = []
    kept_pairs: List[Tuple[int, int]] = []
    for source_text_idx, target_text_idx in transcript_pairs:
        if source_edited_mask[source_text_idx] or target_edited_mask[target_text_idx]:
            continue
        source_seg_idx = source_map.get(source_text_idx)
        target_seg_idx = target_map.get(target_text_idx)
        if source_seg_idx is None or target_seg_idx is None:
            continue
        source_seg = source_segments_all[source_seg_idx]
        target_seg = target_segments_all[target_seg_idx]
        if source_seg.word != target_seg.word:
            continue
        source_selected.append(source_seg)
        target_selected.append(target_seg)
        kept_pairs.append((source_text_idx, target_text_idx))
    return source_selected, target_selected, kept_pairs


def compute_wdtw_from_alignments(
    *,
    source_alignments: Optional[Sequence[Mapping[str, Any]]] = None,
    target_alignments: Optional[Sequence[Mapping[str, Any]]] = None,
    source_transcript: Optional[str] = None,
    target_transcript: Optional[str] = None,
    edit_st_inst: Optional[str] = None,
    unmasked_paired_tags: Collection[str] = (),
    mode: str = "preserved",
    word_mismatch_penalty: float = 1.0,
) -> WDTWResult:
    """Compute normalized WDTW from precomputed forced alignments."""
    if source_alignments is None or target_alignments is None:
        raise ValueError("source_alignments and target_alignments are required")

    if source_transcript is None or target_transcript is None:
        if not edit_st_inst:
            raise ValueError(
                "source_transcript and target_transcript are required unless edit_st_inst is provided"
            )
        instruction = parse_edit_st_inst(edit_st_inst)
        source_transcript = instruction.source_text
        target_transcript = instruction.target_text

    source_segments_all = alignment_to_segments(source_alignments)
    target_segments_all = alignment_to_segments(target_alignments)
    return compute_wdtw_from_segments(
        source_segments_all=source_segments_all,
        target_segments_all=target_segments_all,
        source_transcript=source_transcript,
        target_transcript=target_transcript,
        edit_st_inst=edit_st_inst,
        unmasked_paired_tags=unmasked_paired_tags,
        mode=mode,
        word_mismatch_penalty=word_mismatch_penalty,
    )


def compute_wdtw_from_segments(
    *,
    source_segments_all: Sequence[WordSegment],
    target_segments_all: Sequence[WordSegment],
    source_transcript: str,
    target_transcript: str,
    edit_st_inst: Optional[str] = None,
    unmasked_paired_tags: Collection[str] = (),
    mode: str = "preserved",
    word_mismatch_penalty: float = 1.0,
) -> WDTWResult:
    """Compute WDTW from already normalized alignment segments."""
    source_segments, target_segments, preserved_pairs = _select_segments(
        source_transcript,
        target_transcript,
        source_segments_all,
        target_segments_all,
        mode=mode,
        edit_st_inst=edit_st_inst,
        unmasked_paired_tags=unmasked_paired_tags,
    )

    normalizer = sum(seg.duration for seg in source_segments) + sum(
        seg.duration for seg in target_segments
    )
    if normalizer <= 0.0:
        return WDTWResult(
            wdtw=None,
            distance=0.0,
            normalizer=0.0,
            mode=mode,
            status="no_segments",
            source_segments=source_segments,
            target_segments=target_segments,
            preserved_pairs=preserved_pairs,
        )

    distance = dtw_distance(
        source_segments,
        target_segments,
        word_mismatch_penalty=word_mismatch_penalty,
    )
    return WDTWResult(
        wdtw=distance / normalizer,
        distance=distance,
        normalizer=normalizer,
        mode=mode,
        status="ok",
        source_segments=source_segments,
        target_segments=target_segments,
        preserved_pairs=preserved_pairs,
    )


def compute_wdtw_dur_from_alignments(
    *,
    source_alignments: Optional[Sequence[Mapping[str, Any]]] = None,
    target_alignments: Optional[Sequence[Mapping[str, Any]]] = None,
    source_transcript: Optional[str] = None,
    target_transcript: Optional[str] = None,
    preservation_st_inst: Optional[str] = None,
    word_mismatch_penalty: float = 1.0,
) -> WDTWDurationResult:
    """Compute WDTW-Dur while retaining words inside pitch edit tags."""
    result = compute_wdtw_from_alignments(
        source_alignments=source_alignments,
        target_alignments=target_alignments,
        source_transcript=source_transcript,
        target_transcript=target_transcript,
        edit_st_inst=preservation_st_inst,
        unmasked_paired_tags=("pitch",),
        mode="preserved",
        word_mismatch_penalty=word_mismatch_penalty,
    )
    return WDTWDurationResult(
        wdtw_dur=result.wdtw,
        distance=result.distance,
        normalizer=result.normalizer,
        mode=result.mode,
        status=result.status,
        source_segments=result.source_segments,
        target_segments=result.target_segments,
        preserved_pairs=result.preserved_pairs,
    )


def compute_wdtw_dur_from_segments(
    *,
    source_segments_all: Sequence[WordSegment],
    target_segments_all: Sequence[WordSegment],
    source_transcript: str,
    target_transcript: str,
    preservation_st_inst: Optional[str] = None,
    word_mismatch_penalty: float = 1.0,
) -> WDTWDurationResult:
    """Compute preserved-mode WDTW-Dur from shared alignment segments."""
    result = compute_wdtw_from_segments(
        source_segments_all=source_segments_all,
        target_segments_all=target_segments_all,
        source_transcript=source_transcript,
        target_transcript=target_transcript,
        edit_st_inst=preservation_st_inst,
        unmasked_paired_tags=("pitch",),
        mode="preserved",
        word_mismatch_penalty=word_mismatch_penalty,
    )
    return WDTWDurationResult(
        wdtw_dur=result.wdtw,
        distance=result.distance,
        normalizer=result.normalizer,
        mode=result.mode,
        status=result.status,
        source_segments=result.source_segments,
        target_segments=result.target_segments,
        preserved_pairs=result.preserved_pairs,
    )


def compute_wdtw_f0_from_spans(
    *,
    source_segments: Sequence[WordSegment],
    target_segments: Sequence[WordSegment],
    source_f0_spans: Sequence[Mapping[str, Any]],
    target_f0_spans: Sequence[Mapping[str, Any]],
) -> WDTWF0Result:
    """Compute equal-word-weighted F0 preservation error in semitones."""
    if len(source_segments) != len(target_segments):
        raise ValueError(
            "Source and target preserved segment counts must match: "
            f"source={len(source_segments)}, target={len(target_segments)}"
        )
    eligible_count = len(source_segments)
    if eligible_count == 0:
        return WDTWF0Result(
            wdtw_f0=None,
            distance=0.0,
            normalizer=0.0,
            status="no_preserved_words",
            eligible_word_count=0,
            valid_word_count=0,
            skipped_word_count=0,
            word_results=[],
        )
    if len(source_f0_spans) != eligible_count or len(target_f0_spans) != eligible_count:
        raise ValueError(
            "F0 span count must match preserved word count: "
            f"eligible={eligible_count}, source={len(source_f0_spans)}, "
            f"target={len(target_f0_spans)}"
        )

    word_results: List[Dict[str, Any]] = []
    total_error = 0.0
    valid_count = 0
    statistic_keys = ("min_f0_hz", "max_f0_hz", "mean_f0_hz")
    for index in range(eligible_count):
        source_f0 = source_f0_spans[index]
        target_f0 = target_f0_spans[index]
        errors: Dict[str, float] = {}
        valid = True
        for key in statistic_keys:
            try:
                source_value = float(source_f0[key])
                target_value = float(target_f0[key])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if (
                not math.isfinite(source_value)
                or not math.isfinite(target_value)
                or source_value <= 0.0
                or target_value <= 0.0
            ):
                valid = False
                break
            errors[key.removesuffix("_f0_hz") + "_error_semitones"] = abs(
                12.0 * math.log2(target_value / source_value)
            )

        word_result: Dict[str, Any] = {
            "index": index,
            "word": source_segments[index].word,
            "source_span": source_segments[index].to_dict(),
            "target_span": target_segments[index].to_dict(),
            "source_f0": dict(source_f0),
            "target_f0": dict(target_f0),
            "status": "ok" if valid else "invalid_f0",
        }
        if valid:
            word_error = sum(errors.values()) / len(errors)
            word_result.update(errors)
            word_result["word_error_semitones"] = word_error
            total_error += word_error
            valid_count += 1
        word_results.append(word_result)

    skipped_count = eligible_count - valid_count
    return WDTWF0Result(
        wdtw_f0=(total_error / valid_count) if valid_count else None,
        distance=total_error,
        normalizer=float(valid_count),
        status="ok" if valid_count else "no_valid_f0",
        eligible_word_count=eligible_count,
        valid_word_count=valid_count,
        skipped_word_count=skipped_count,
        word_results=word_results,
    )


def unavailable_wdtw_f0_result(
    status: str,
    error: str,
    *,
    eligible_word_count: int = 0,
) -> WDTWF0Result:
    return WDTWF0Result(
        wdtw_f0=None,
        distance=0.0,
        normalizer=0.0,
        status=status,
        eligible_word_count=eligible_word_count,
        valid_word_count=0,
        skipped_word_count=eligible_word_count,
        word_results=[],
        error=error,
    )


def _wdtw_result_to_dict(result: object) -> Dict[str, Any]:
    if isinstance(result, WDTWResult):
        return result.to_dict(include_segments=False)
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict(include_segments=False)
    raise TypeError(f"Unsupported WDTW result type: {type(result).__name__}")


def aggregate_wdtw_results(results: Sequence[object]) -> WDTWAggregateResult:
    """Aggregate WDTW results by total DTW distance over total normalizer."""
    total_distance = 0.0
    total_normalizer = 0.0
    valid_count = 0
    skipped_no_segments = 0
    skipped_missing = 0
    modes: set[str] = set()

    for result in results:
        try:
            data = _wdtw_result_to_dict(result)
            distance = float(data["distance"])
            normalizer = float(data["normalizer"])
        except (KeyError, TypeError, ValueError):
            skipped_missing += 1
            continue
        if data.get("status") == "no_segments" or normalizer <= 0.0:
            skipped_no_segments += 1
            continue
        total_distance += distance
        total_normalizer += normalizer
        valid_count += 1
        mode = data.get("mode")
        if isinstance(mode, str) and mode:
            modes.add(mode)

    if total_normalizer > 0.0:
        wdtw: Optional[float] = total_distance / total_normalizer
        status = "ok"
    else:
        wdtw = None
        status = "no_segments"
    if not modes:
        mode = "unknown"
    elif len(modes) == 1:
        mode = next(iter(modes))
    else:
        mode = "mixed"
    skipped_count = skipped_no_segments + skipped_missing
    return WDTWAggregateResult(
        wdtw=wdtw,
        distance=total_distance,
        normalizer=total_normalizer,
        mode=mode,
        status=status,
        count=len(results),
        valid_count=valid_count,
        skipped_count=skipped_count,
        skipped_no_segments=skipped_no_segments,
        skipped_missing=skipped_missing,
    )


def aggregate_wdtw_dur_results(
    results: Sequence[object],
) -> WDTWDurationAggregateResult:
    """Aggregate WDTW-Dur without aliasing the legacy WDTW metric."""
    normalized: List[Dict[str, Any]] = []
    for result in results:
        if isinstance(result, WDTWDurationResult):
            normalized.append(result.to_dict(include_segments=False))
        elif isinstance(result, dict):
            normalized.append(result)
        elif hasattr(result, "to_dict"):
            normalized.append(result.to_dict(include_segments=False))
        else:
            normalized.append({})
    legacy_shaped = [
        {**data, "wdtw": data.get("wdtw_dur")}
        for data in normalized
    ]
    aggregate = aggregate_wdtw_results(legacy_shaped)
    return WDTWDurationAggregateResult(
        wdtw_dur=aggregate.wdtw,
        distance=aggregate.distance,
        normalizer=aggregate.normalizer,
        mode=aggregate.mode,
        status=aggregate.status,
        count=aggregate.count,
        valid_count=aggregate.valid_count,
        skipped_count=aggregate.skipped_count,
        skipped_no_segments=aggregate.skipped_no_segments,
        skipped_missing=aggregate.skipped_missing,
    )


def aggregate_wdtw_f0_results(results: Sequence[object]) -> WDTWF0AggregateResult:
    """Aggregate WDTW-F0 by total word error over total valid word count."""
    total_distance = 0.0
    total_normalizer = 0.0
    valid_count = 0
    skipped_no_valid_f0 = 0
    skipped_no_preserved_words = 0
    skipped_unavailable = 0
    skipped_error = 0
    skipped_other_invalid = 0
    skipped_missing = 0
    for result in results:
        if isinstance(result, WDTWF0Result):
            data = result.to_dict(include_words=False)
        elif isinstance(result, dict):
            data = result
        elif hasattr(result, "to_dict"):
            data = result.to_dict(include_words=False)
        else:
            skipped_missing += 1
            continue
        try:
            distance = float(data["distance"])
            normalizer = float(data["normalizer"])
        except (KeyError, TypeError, ValueError):
            skipped_missing += 1
            continue
        if normalizer <= 0.0:
            status = str(data.get("status") or "")
            if status == "no_valid_f0":
                skipped_no_valid_f0 += 1
            elif status == "no_preserved_words":
                skipped_no_preserved_words += 1
            elif status == "unavailable":
                skipped_unavailable += 1
            elif status == "error":
                skipped_error += 1
            else:
                skipped_other_invalid += 1
            continue
        total_distance += distance
        total_normalizer += normalizer
        valid_count += 1
    value = total_distance / total_normalizer if total_normalizer > 0.0 else None
    return WDTWF0AggregateResult(
        wdtw_f0=value,
        distance=total_distance,
        normalizer=total_normalizer,
        status="ok" if value is not None else "no_valid_f0",
        count=len(results),
        valid_count=valid_count,
        skipped_count=(
            skipped_no_valid_f0
            + skipped_no_preserved_words
            + skipped_unavailable
            + skipped_error
            + skipped_other_invalid
            + skipped_missing
        ),
        skipped_no_valid_f0=skipped_no_valid_f0,
        skipped_no_preserved_words=skipped_no_preserved_words,
        skipped_unavailable=skipped_unavailable,
        skipped_error=skipped_error,
        skipped_other_invalid=skipped_other_invalid,
        skipped_missing=skipped_missing,
    )
