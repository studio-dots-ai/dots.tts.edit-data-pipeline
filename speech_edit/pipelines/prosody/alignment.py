"""Query-alignment validation helpers for prosody edit regions."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _validate_query_alignment_response(
    response: Dict[str, Any],
    *,
    query_count: int,
    query_texts: Optional[Sequence[str]] = None,
) -> None:
    answers = response.get("answers")
    if not isinstance(answers, list) or len(answers) != query_count:
        raise ValueError(
            "Aligner did not return one answer per query. "
            f"queries={query_count}, answers={answers!r}"
        )

    for idx, answer in enumerate(answers):
        if not isinstance(answer, dict) or "start" not in answer or "end" not in answer:
            raise ValueError(
                f"Invalid answer for query {idx}: expected start/end, got {answer!r}"
            )
        start_sec = float(answer["start"])
        end_sec = float(answer["end"])
        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            raise ValueError(
                f"Invalid answer for query {idx}: non-finite start/end "
                f"({start_sec}, {end_sec})"
            )
        if end_sec < start_sec:
            raise ValueError(
                f"Invalid answer for query {idx}: start={start_sec}, end={end_sec}"
            )


def align_queries(
    aligner_client: Any,
    *,
    audio_bytes: bytes,
    plain_text: str,
    backend: str,
    language: Optional[str],
    queries: Sequence[str],
    query_texts: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve query spans through the configured Qwen3 aligner."""
    if backend != "qwen3":
        raise ValueError("aligner backend must be qwen3")
    response = aligner_client.align(
        audio_bytes,
        plain_text,
        backend=backend,
        language=language,
        queries=list(queries),
        return_response=True,
    )
    _validate_query_alignment_response(
        response, query_count=len(queries), query_texts=query_texts,
    )
    if response.get("backend", backend) != backend:
        raise ValueError("Aligner response backend differs from the requested backend")
    return response, {"backend": backend}
