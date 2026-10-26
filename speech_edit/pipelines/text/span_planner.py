"""Span and duration planning for tagged text edit synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from speech_edit.data.text_markup import (
    DEL_TAG,
    INS_TAG,
    SUB_TAG,
    TaggedEditSegment,
    edit_segments,
    render_source,
    tag_query,
)

LETTER_TO_CHAR_RATIO = 2.7
MIN_EDIT_SEC = 0.01


@dataclass(frozen=True)
class AlignmentQuery:
    """One query sent to the aligner for an edit segment."""

    segment_index: int
    text: str
    char_start: int
    char_end: int
    query: str
    anchor_mode: str


@dataclass(frozen=True)
class PlannedSpan:
    """A text edit segment with source and target time spans."""

    segment: TaggedEditSegment
    source_span: List[float]
    target_span: List[float]
    fix_duration: float
    query: Optional[AlignmentQuery] = None
    alignment_answer: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "kind": self.segment.kind,
            "source_text": self.segment.source.strip(),
            "target_text": self.segment.target.strip(),
            "source_char_span": [self.segment.source_start, self.segment.source_end],
            "source_span": self.source_span,
            "target_span": self.target_span,
            "fix_duration": self.fix_duration,
        }


@dataclass(frozen=True)
class SpanPlan:
    """Complete source/target time plan for one tagged edit."""

    source_spans: List[List[float]]
    target_spans: List[List[float]]
    planned_spans: List[PlannedSpan]
    target_duration: float
    duration_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def letter_equivalent_count(text: str) -> float:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    english_letters = sum(
        1 for char in text if char.isalpha() and not ("\u4e00" <= char <= "\u9fff")
    )
    return english_letters + chinese_chars * LETTER_TO_CHAR_RATIO


def estimate_speech_rate(text: str, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 10.0
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    english_letters = sum(
        1 for char in text if char.isalpha() and not ("\u4e00" <= char <= "\u9fff")
    )
    chinese_equiv = chinese_chars * LETTER_TO_CHAR_RATIO
    if chinese_equiv > english_letters:
        return chinese_equiv / duration_sec if chinese_chars > 0 else 10.0
    return english_letters / duration_sec if english_letters > 0 else 10.0


def estimate_text_duration(text: str, speech_rate: Optional[float]) -> float:
    if not text or speech_rate is None or speech_rate <= 0:
        return 0.0
    return letter_equivalent_count(text) / speech_rate


def build_alignment_queries_from_st(st: str) -> List[AlignmentQuery]:
    """Build current-source-side aligner queries from one tagged instruction."""
    source_text = render_source(st)
    queries: List[AlignmentQuery] = []
    for index, segment in enumerate(edit_segments(st)):
        start = segment.source_start
        end = segment.source_end
        query = tag_query(source_text, start, end)
        queries.append(
            AlignmentQuery(
                segment_index=index,
                text=source_text[start:end],
                char_start=start,
                char_end=end,
                query=query,
                anchor_mode="point" if segment.kind == INS_TAG else "span",
            )
        )
    return queries


def _answer_to_span(
    answer: Dict[str, Any],
    *,
    anchor_mode: str,
    audio_duration: float,
    min_edit_sec: float = MIN_EDIT_SEC,
) -> List[float]:
    start = max(0.0, float(answer.get("start", 0.0)))
    end = max(start, float(answer.get("end", start)))
    audio_duration = max(0.0, float(audio_duration))
    if anchor_mode == "span":
        if end <= start:
            end = min(audio_duration, start + min_edit_sec)
        return [round(start, 5), round(min(end, audio_duration), 5)]

    seam = end if anchor_mode == "left" else start
    seam = min(max(0.0, seam), audio_duration)
    return [round(seam, 5), round(seam, 5)]


def _fix_duration_for_segment(
    segment: TaggedEditSegment,
    source_span: Sequence[float],
    *,
    speech_rate: float,
    generated_duration: Optional[float],
    duration_mode: str,
) -> float:
    source_duration = max(0.0, float(source_span[1]) - float(source_span[0]))
    if segment.kind == DEL_TAG:
        return 0.0
    if duration_mode == "generation" and generated_duration is not None:
        return max(0.0, float(generated_duration))
    target_duration = estimate_text_duration(segment.target.strip(), speech_rate)
    if segment.kind == INS_TAG:
        return target_duration + source_duration
    if segment.kind == SUB_TAG:
        return target_duration
    return source_duration


def _expand_spans(
    spans: Sequence[Sequence[float]],
    durations: Sequence[float],
    *,
    radius: float,
    audio_duration: float,
) -> Tuple[List[List[float]], List[float]]:
    if radius <= 0:
        return [list(span) for span in spans], [float(duration) for duration in durations]
    expanded: List[List[float]] = []
    expanded_durations: List[float] = []
    for span, duration in zip(spans, durations):
        start, end = float(span[0]), float(span[1])
        new_start = max(0.0, start - radius)
        new_end = min(float(audio_duration), end + radius)
        expansion = (start - new_start) + (new_end - end)
        expanded.append([round(new_start, 5), round(new_end, 5)])
        expanded_durations.append(max(0.0, float(duration) + expansion))
    return expanded, expanded_durations


def _merge_spans(
    spans: Sequence[Sequence[float]],
    durations: Sequence[float],
    *,
    merge_gap: float,
) -> Tuple[List[List[float]], List[float]]:
    if not spans:
        return [], []
    items = sorted(zip(spans, durations), key=lambda item: (item[0][0], item[0][1]))
    merged_spans: List[List[float]] = []
    merged_durations: List[float] = []
    cur_start, cur_end = float(items[0][0][0]), float(items[0][0][1])
    cur_duration = float(items[0][1])
    for span, duration in items[1:]:
        start, end = float(span[0]), float(span[1])
        gap = start - cur_end
        if gap < merge_gap:
            cur_duration += float(duration) + max(0.0, gap)
            cur_end = max(cur_end, end)
        else:
            merged_spans.append([round(cur_start, 5), round(cur_end, 5)])
            merged_durations.append(max(0.0, cur_duration))
            cur_start, cur_end = start, end
            cur_duration = float(duration)
    merged_spans.append([round(cur_start, 5), round(cur_end, 5)])
    merged_durations.append(max(0.0, cur_duration))
    return merged_spans, merged_durations


def _target_spans_from_source_spans(
    source_spans: Sequence[Sequence[float]],
    fix_durations: Sequence[float],
    *,
    source_duration: float,
) -> Tuple[List[List[float]], float]:
    target_spans: List[List[float]] = []
    source_cursor = 0.0
    target_cursor = 0.0
    for span, fix_duration in sorted(
        zip(source_spans, fix_durations), key=lambda item: (item[0][0], item[0][1])
    ):
        start, end = float(span[0]), float(span[1])
        keep_duration = max(0.0, start - source_cursor)
        target_cursor += keep_duration
        target_start = target_cursor
        target_end = target_start + max(0.0, float(fix_duration))
        target_spans.append([round(target_start, 5), round(target_end, 5)])
        target_cursor = target_end
        source_cursor = max(source_cursor, end)
    target_cursor += max(0.0, float(source_duration) - source_cursor)
    return target_spans, round(target_cursor, 5)


def plan_spans_from_answers(
    *,
    source_text: str,
    target_text: str,
    edit_segments: Sequence[TaggedEditSegment],
    queries: Sequence[AlignmentQuery],
    answers: Sequence[Dict[str, Any]],
    source_duration: float,
    speech_duration: Optional[float] = None,
    duration_mode: str = "estimate",
    generated_durations: Optional[Sequence[Optional[float]]] = None,
    expand_radius: float = 0.4,
    merge_gap: float = 0.1,
) -> SpanPlan:
    """Build source and planned target spans from tagged text and aligner answers."""
    if duration_mode not in {"estimate", "generation", "synthesis"}:
        raise ValueError("duration_mode must be 'estimate', 'generation', or 'synthesis'")
    if len(edit_segments) != len(queries) or len(queries) != len(answers):
        raise ValueError("edit_segments, queries, and answers must have equal length")

    generated_durations = generated_durations or [None] * len(edit_segments)
    speech_duration = source_duration if speech_duration is None else speech_duration
    speech_rate = estimate_speech_rate(source_text, speech_duration)
    raw_spans: List[List[float]] = []
    raw_durations: List[float] = []
    raw_planned: List[Tuple[TaggedEditSegment, AlignmentQuery, Dict[str, Any]]] = []
    for segment, query, answer, generated_duration in zip(
        edit_segments, queries, answers, generated_durations
    ):
        source_span = _answer_to_span(
            answer,
            anchor_mode=query.anchor_mode,
            audio_duration=source_duration,
        )
        fix_duration = _fix_duration_for_segment(
            segment,
            source_span,
            speech_rate=speech_rate,
            generated_duration=generated_duration,
            duration_mode="generation" if duration_mode == "synthesis" else duration_mode,
        )
        raw_spans.append(source_span)
        raw_durations.append(max(0.0, fix_duration))
        raw_planned.append((segment, query, answer))

    expanded_spans, expanded_durations = _expand_spans(
        raw_spans,
        raw_durations,
        radius=expand_radius,
        audio_duration=source_duration,
    )
    merged_spans, merged_durations = _merge_spans(
        expanded_spans,
        expanded_durations,
        merge_gap=merge_gap,
    )
    target_spans, target_duration = _target_spans_from_source_spans(
        merged_spans,
        merged_durations,
        source_duration=source_duration,
    )
    expanded_target_spans, _ = _target_spans_from_source_spans(
        expanded_spans,
        expanded_durations,
        source_duration=source_duration,
    )

    planned_spans: List[PlannedSpan] = []
    for idx, (segment, query, answer) in enumerate(raw_planned):
        source_span = expanded_spans[idx]
        planned_spans.append(
            PlannedSpan(
                segment=segment,
                source_span=source_span,
                target_span=expanded_target_spans[idx],
                fix_duration=round(expanded_durations[idx], 5),
                query=query,
                alignment_answer=answer,
            )
        )

    return SpanPlan(
        source_spans=merged_spans,
        target_spans=target_spans,
        planned_spans=planned_spans,
        target_duration=target_duration,
        duration_mode=duration_mode,
        metadata={
            "source_text": source_text,
            "target_text": target_text,
            "speech_rate": speech_rate,
            "speech_duration": speech_duration,
            "source_duration": source_duration,
            "expand_radius": expand_radius,
            "merge_gap": merge_gap,
            "raw_source_spans": raw_spans,
            "raw_fix_durations": raw_durations,
            "expanded_source_spans": expanded_spans,
            "expanded_fix_durations": expanded_durations,
        },
    )


def alignment_answers_to_spans(
    *,
    audio_duration: float,
    queries: Sequence[AlignmentQuery],
    answers: Sequence[Dict[str, Any]],
) -> List[List[float]]:
    if len(queries) != len(answers):
        raise ValueError("queries and answers must have equal length")
    spans = [
        _answer_to_span(
            answer,
            anchor_mode=query.anchor_mode,
            audio_duration=audio_duration,
        )
        for query, answer in zip(queries, answers)
    ]
    return _merge_only(spans)


def _merge_only(spans: Sequence[Sequence[float]]) -> List[List[float]]:
    if not spans:
        return []
    merged: List[List[float]] = []
    for span in sorted(spans, key=lambda item: (item[0], item[1])):
        start, end = float(span[0]), float(span[1])
        if not merged or start > merged[-1][1]:
            merged.append([round(start, 5), round(end, 5)])
        else:
            merged[-1][1] = round(max(merged[-1][1], end), 5)
    return merged
