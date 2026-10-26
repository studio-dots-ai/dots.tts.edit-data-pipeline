from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

QUERY_START_TAG = "<q>"
QUERY_END_TAG = "</q>"


class QueryValidationError(ValueError):
    """Raised when tagged queries are malformed or inconsistent."""


@dataclass(frozen=True)
class ParsedQuery:
    query: str
    start_pos: int
    end_pos: int


@dataclass(frozen=True)
class QueryModePlan:
    plain_text: str
    parsed_queries: List[ParsedQuery]
    boundary_positions: List[int]
    segments: List[str]
    segment_units: List[List[str]]
    full_units: List[str]
    query_mode_units: List[str]
    tokenization_matches_default: bool


def parse_tagged_query(query: str) -> Tuple[int, int, str]:
    start_count = query.count(QUERY_START_TAG)
    end_count = query.count(QUERY_END_TAG)
    if start_count != 1 or end_count != 1:
        raise QueryValidationError(
            "Each query must contain exactly one '<q>' tag and one '</q>' tag. "
            f"Got start_tags={start_count}, end_tags={end_count}."
        )

    start_idx = query.find(QUERY_START_TAG)
    end_idx = query.find(QUERY_END_TAG)
    if start_idx < 0 or end_idx < 0 or start_idx > end_idx:
        raise QueryValidationError("Query tags are malformed or out of order.")

    plain_text = query.replace(QUERY_START_TAG, "").replace(QUERY_END_TAG, "")
    return start_idx, end_idx - len(QUERY_START_TAG), plain_text


def build_query_mode_plan(
    text: str,
    queries: Sequence[str],
    *,
    tokenize_text,
) -> QueryModePlan:
    if not queries:
        raise QueryValidationError("queries must contain at least one query string.")

    parsed_queries: List[ParsedQuery] = []
    plain_text: str | None = None
    boundary_positions = {0, len(text)}

    for query in queries:
        start_pos, end_pos, query_plain_text = parse_tagged_query(query)
        if plain_text is None:
            plain_text = query_plain_text
        elif query_plain_text != plain_text:
            raise QueryValidationError(
                "All queries must reduce to the same plain text after removing tags."
            )
        parsed_queries.append(
            ParsedQuery(query=query, start_pos=start_pos, end_pos=end_pos)
        )
        boundary_positions.add(start_pos)
        boundary_positions.add(end_pos)

    assert plain_text is not None
    if plain_text != text:
        raise QueryValidationError(
            "Removing query tags must recover the exact base text used for alignment."
        )

    sorted_boundaries = sorted(boundary_positions)
    segments = [
        text[sorted_boundaries[idx] : sorted_boundaries[idx + 1]]
        for idx in range(len(sorted_boundaries) - 1)
    ]
    segment_units = [list(tokenize_text(segment)) for segment in segments]
    query_mode_units = [unit for units in segment_units for unit in units]
    full_units = list(tokenize_text(text))

    return QueryModePlan(
        plain_text=text,
        parsed_queries=parsed_queries,
        boundary_positions=sorted_boundaries,
        segments=segments,
        segment_units=segment_units,
        full_units=full_units,
        query_mode_units=query_mode_units,
        tokenization_matches_default=(query_mode_units == full_units),
    )


def _boundary_unit_counts(plan: QueryModePlan) -> Dict[int, int]:
    counts: Dict[int, int] = {plan.boundary_positions[0]: 0}
    total = 0
    for idx, segment in enumerate(plan.segments):
        total += len(plan.segment_units[idx])
        counts[plan.boundary_positions[idx + 1]] = total
    return counts


def _boundary_time(
    alignments: Sequence[Dict[str, float]], boundary_index: int
) -> float:
    if not alignments:
        raise QueryValidationError(
            "Cannot resolve query boundaries against empty alignments."
        )
    if boundary_index <= 0:
        return round(float(alignments[0]["start"]), 6)
    if boundary_index >= len(alignments):
        return round(float(alignments[-1]["end"]), 6)
    return round(float(alignments[boundary_index - 1]["end"]), 6)


def resolve_query_answers(
    plan: QueryModePlan,
    alignments: Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    if len(alignments) != len(plan.query_mode_units):
        raise QueryValidationError(
            "Alignment count does not match query-mode tokenization: "
            f"alignments={len(alignments)}, units={len(plan.query_mode_units)}"
        )

    boundary_counts = _boundary_unit_counts(plan)
    answers: List[Dict[str, float]] = []
    for parsed in plan.parsed_queries:
        start_boundary = boundary_counts[parsed.start_pos]
        end_boundary = boundary_counts[parsed.end_pos]

        if start_boundary == end_boundary:
            point = _boundary_time(alignments, start_boundary)
            answers.append({"start": point, "end": point})
            continue

        answers.append(
            {
                "start": round(float(alignments[start_boundary]["start"]), 6),
                "end": round(float(alignments[end_boundary - 1]["end"]), 6),
            }
        )

    return answers
