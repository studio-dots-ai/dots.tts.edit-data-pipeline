"""Audio assembly core logic for `<target>...</target>` edit transcripts."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import requests
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 24000
TARGET_OPEN = "<target>"
TARGET_CLOSE = "</target>"


@dataclass(frozen=True)
class EditPiece:
    from_target: bool
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class SplicePoint:
    position_sec: float


@dataclass
class AssemblyResult:
    audio: np.ndarray
    sample_rate: int
    splice_points: List[SplicePoint]
    source_answers: List[Dict[str, float]]
    target_answers: List[Dict[str, float]]
    target_spans: List[Dict[str, Any]]
    details: Dict[str, Any]


def decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode audio bytes to float32 mono numpy array and sample rate."""
    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio, sr


def encode_wav(audio: np.ndarray, sr: int) -> bytes:
    """Encode float32 numpy array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio on CPU."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def _validate_plain_chunk(chunk: str) -> None:
    if "<" in chunk or ">" in chunk:
        raise ValueError(
            "Only <target>...</target> markup is supported in edit_tagged_text."
        )


def _append_piece(pieces: List[EditPiece], *, from_target: bool, text: str) -> None:
    if not text:
        return
    start_char = pieces[-1].end_char if pieces else 0
    pieces.append(
        EditPiece(
            from_target=from_target,
            text=text,
            start_char=start_char,
            end_char=start_char + len(text),
        )
    )


def _merge_adjacent_pieces(pieces: List[EditPiece]) -> List[EditPiece]:
    merged: List[EditPiece] = []
    for piece in pieces:
        if not merged:
            merged.append(piece)
            continue
        prev = merged[-1]
        if prev.from_target == piece.from_target and prev.end_char == piece.start_char:
            merged[-1] = EditPiece(
                from_target=prev.from_target,
                text=prev.text + piece.text,
                start_char=prev.start_char,
                end_char=piece.end_char,
            )
            continue
        merged.append(piece)
    return merged


def parse_edit_tagged_text(edit_tagged_text: str) -> Tuple[str, List[EditPiece]]:
    """Parse `<target>...</target>` markup into ordered source/target pieces."""
    if not isinstance(edit_tagged_text, str) or not edit_tagged_text:
        raise ValueError("edit_tagged_text must be a non-empty string.")

    pieces: List[EditPiece] = []
    cursor = 0

    while cursor < len(edit_tagged_text):
        open_idx = edit_tagged_text.find(TARGET_OPEN, cursor)
        if open_idx < 0:
            tail = edit_tagged_text[cursor:]
            _validate_plain_chunk(tail)
            _append_piece(pieces, from_target=False, text=tail)
            break

        plain_chunk = edit_tagged_text[cursor:open_idx]
        _validate_plain_chunk(plain_chunk)
        _append_piece(pieces, from_target=False, text=plain_chunk)

        inner_start = open_idx + len(TARGET_OPEN)
        close_idx = edit_tagged_text.find(TARGET_CLOSE, inner_start)
        if close_idx < 0:
            raise ValueError("Missing closing </target> tag.")

        inner_text = edit_tagged_text[inner_start:close_idx]
        if "<" in inner_text or ">" in inner_text:
            raise ValueError("Nested or malformed <target> tags are not supported.")
        _append_piece(pieces, from_target=True, text=inner_text)
        cursor = close_idx + len(TARGET_CLOSE)

    if not pieces:
        raise ValueError("edit_tagged_text must contain transcript content.")

    pieces = _merge_adjacent_pieces(pieces)
    base_text = "".join(piece.text for piece in pieces)
    if not base_text:
        raise ValueError("edit_tagged_text must contain transcript content.")
    return base_text, pieces


def _build_query(base_text: str, start_char: int, end_char: int) -> str:
    return (
        base_text[:start_char]
        + "<q>"
        + base_text[start_char:end_char]
        + "</q>"
        + base_text[end_char:]
    )


def build_queries(
    base_text: str, pieces: List[EditPiece]
) -> Tuple[List[str], Dict[int, int]]:
    """Build full-text queries for every target piece.

    Returns `(queries, piece_to_query_index)`.
    """
    queries: List[str] = []
    piece_to_query_index: Dict[int, int] = {}
    for piece_index, piece in enumerate(pieces):
        if not piece.from_target:
            continue
        piece_to_query_index[piece_index] = len(queries)
        queries.append(_build_query(base_text, piece.start_char, piece.end_char))
    return queries, piece_to_query_index


def _has_abstract_adjacent_targets(edit_tagged_text: str) -> bool:
    return (
        re.search(
            rf"{re.escape(TARGET_CLOSE)}\s*{re.escape(TARGET_OPEN)}", edit_tagged_text
        )
        is not None
    )


def _validate_time_spans(
    spans: List[Tuple[float, float]], *, duration_sec: float, label: str
) -> List[Dict[str, float]]:
    validated: List[Dict[str, float]] = []
    prev_end: Optional[float] = None
    for idx, span in enumerate(spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"{label}[{idx}] must be a 2-item time span.")

        start_sec = float(span[0])
        end_sec = float(span[1])
        if not np.isfinite(start_sec) or not np.isfinite(end_sec):
            raise ValueError(f"{label}[{idx}] must contain finite values.")
        if start_sec < 0.0 or end_sec > duration_sec:
            raise ValueError(
                f"{label}[{idx}] must stay within audio duration {duration_sec:.6f}s."
            )
        if end_sec <= start_sec:
            raise ValueError(f"{label}[{idx}] must satisfy start < end.")
        if prev_end is not None and start_sec < prev_end:
            raise ValueError(f"{label} must be monotonic and non-overlapping.")

        validated.append({"start": start_sec, "end": end_sec})
        prev_end = end_sec
    return validated


def _prepare_manual_time_spans(
    *,
    edit_tagged_text: str,
    pieces: List[EditPiece],
    source_duration_sec: float,
    target_duration_sec: float,
    time_spans_from_source: Optional[List[Tuple[float, float]]],
    time_spans_from_target: Optional[List[Tuple[float, float]]],
) -> Optional[
    Tuple[
        Dict[int, Dict[str, float]],
        Dict[int, Dict[str, float]],
        List[Dict[str, float]],
        List[Dict[str, float]],
    ]
]:
    if time_spans_from_source is None and time_spans_from_target is None:
        return None
    if time_spans_from_source is None or time_spans_from_target is None:
        raise ValueError("Both source and target manual spans are required")

    if _has_abstract_adjacent_targets(edit_tagged_text):
        raise ValueError("Manual spans cannot represent abstract-adjacent targets")

    target_piece_count = edit_tagged_text.count(TARGET_OPEN)
    source_piece_count = sum(1 for piece in pieces if not piece.from_target)
    if len(time_spans_from_target) != target_piece_count:
        raise ValueError(
            "time_spans_from_target count does not match the number of <target> spans."
        )
    if len(time_spans_from_source) != source_piece_count:
        raise ValueError(
            "time_spans_from_source count does not match the expected source segment count."
        )

    validated_source_spans = _validate_time_spans(
        time_spans_from_source,
        duration_sec=source_duration_sec,
        label="time_spans_from_source",
    )
    validated_target_spans = _validate_time_spans(
        time_spans_from_target,
        duration_sec=target_duration_sec,
        label="time_spans_from_target",
    )

    source_span_by_piece_index: Dict[int, Dict[str, float]] = {}
    target_span_by_piece_index: Dict[int, Dict[str, float]] = {}
    source_cursor = 0
    target_cursor = 0
    for piece_index, piece in enumerate(pieces):
        if piece.from_target:
            target_span_by_piece_index[piece_index] = validated_target_spans[
                target_cursor
            ]
            target_cursor += 1
        else:
            source_span_by_piece_index[piece_index] = validated_source_spans[
                source_cursor
            ]
            source_cursor += 1

    source_answers: List[Dict[str, float]] = []
    target_answers: List[Dict[str, float]] = []
    for piece_index, piece in enumerate(pieces):
        if not piece.from_target:
            continue

        prev_source_span = source_span_by_piece_index.get(piece_index - 1)
        next_source_span = source_span_by_piece_index.get(piece_index + 1)
        start_sec = 0.0 if prev_source_span is None else prev_source_span["end"]
        end_sec = (
            source_duration_sec
            if next_source_span is None
            else next_source_span["start"]
        )
        if end_sec <= start_sec:
            raise ValueError(
                "Derived source-side target span is invalid for manual assembly."
            )

        source_answers.append({"start": start_sec, "end": end_sec})
        target_answers.append(target_span_by_piece_index[piece_index])

    return (
        source_span_by_piece_index,
        target_span_by_piece_index,
        source_answers,
        target_answers,
    )

def _slice_audio(
    audio: np.ndarray, sr: int, start_sec: float, end_sec: float
) -> np.ndarray:
    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(audio), int(end_sec * sr))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    return audio[start_idx:end_idx]


def _previous_target_index(
    piece_index: int, piece_to_query_index: Dict[int, int]
) -> Optional[int]:
    prev = [idx for idx in piece_to_query_index if idx < piece_index]
    if not prev:
        return None
    return max(prev)


def _next_target_index(
    piece_index: int, piece_to_query_index: Dict[int, int]
) -> Optional[int]:
    nxt = [idx for idx in piece_to_query_index if idx > piece_index]
    if not nxt:
        return None
    return min(nxt)


def _build_target_spans(
    pieces: List[EditPiece],
    piece_to_query_index: Dict[int, int],
    queries: List[str],
    source_answers: List[Dict[str, float]],
    target_answers: List[Dict[str, float]],
) -> List[Dict[str, Any]]:
    target_spans: List[Dict[str, Any]] = []
    for piece_index, query_index in sorted(
        piece_to_query_index.items(), key=lambda item: item[1]
    ):
        piece = pieces[piece_index]
        target_spans.append(
            {
                "text": piece.text,
                "start_char": piece.start_char,
                "end_char": piece.end_char,
                "query": queries[query_index],
                "source_answer": source_answers[query_index],
                "target_answer": target_answers[query_index],
            }
        )
    return target_spans


def _finalize_assembled_audio(
    assembled_parts: List[Tuple[np.ndarray, EditPiece]], *, base_text: str, sr: int
) -> Tuple[np.ndarray, List[SplicePoint]]:
    if not assembled_parts:
        return np.array([], dtype=np.float32), []

    assembled = np.concatenate([chunk for chunk, _ in assembled_parts])
    splice_points: List[SplicePoint] = []
    running_samples = 0
    prev_piece: Optional[EditPiece] = None
    for chunk, piece in assembled_parts:
        if prev_piece is not None and prev_piece.from_target != piece.from_target:
            splice_points.append(
                SplicePoint(position_sec=running_samples / sr)
            )
        running_samples += len(chunk)
        prev_piece = piece
    return assembled, splice_points


def _build_passthrough_target_spans(
    pieces: List[EditPiece], base_text: str, duration_sec: float
) -> List[Dict[str, Any]]:
    target_spans: List[Dict[str, Any]] = []
    for piece in pieces:
        if not piece.from_target:
            continue
        target_spans.append(
            {
                "text": piece.text,
                "start_char": piece.start_char,
                "end_char": piece.end_char,
                "query": _build_query(base_text, piece.start_char, piece.end_char),
                "source_answer": None,
                "target_answer": {"start": 0.0, "end": duration_sec},
            }
        )
    return target_spans


def _call_aligner(
    wav_bytes: bytes,
    *,
    base_text: str,
    queries: List[str],
    aligner_url: str,
    aligner_backend: str,
    language: Optional[str],
    label: str,
) -> List[Dict[str, float]]:
    payload: Dict[str, Any] = {
        "backend": aligner_backend,
        "audio": {
            "data": base64.b64encode(wav_bytes).decode("utf-8"),
            "format": "wav",
        },
        "text": base_text,
        "word_level": False,
        "queries": queries,
    }
    if language is not None:
        payload["language"] = language

    response = requests.post(f"{aligner_url}/align", json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Aligner gateway failed for {label}: {response.status_code} - {response.text}"
        )

    data = response.json()
    answers = data.get("answers")
    if not isinstance(answers, list) or len(answers) != len(queries):
        raise RuntimeError(
            f"Aligner gateway returned invalid answers for {label}: expected {len(queries)}, got {answers!r}"
        )
    return answers


def _assemble_mixed_audio(
    *,
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    sr: int,
    base_text: str,
    pieces: List[EditPiece],
    piece_to_query_index: Dict[int, int],
    source_answers: List[Dict[str, float]],
    target_answers: List[Dict[str, float]],
    preserve_outer_boundaries: bool,
) -> Tuple[np.ndarray, List[SplicePoint]]:
    source_dur = len(source_audio) / sr
    target_dur = len(target_audio) / sr
    assembled_parts: List[Tuple[np.ndarray, EditPiece]] = []

    for piece_index, piece in enumerate(pieces):
        if piece.from_target:
            query_index = piece_to_query_index[piece_index]
            answer = target_answers[query_index]
            start_sec = (
                0.0
                if preserve_outer_boundaries and piece_index == 0
                else answer["start"]
            )
            end_sec = (
                target_dur
                if preserve_outer_boundaries and piece_index == len(pieces) - 1
                else answer["end"]
            )
            chunk = _slice_audio(target_audio, sr, start_sec, end_sec)
        else:
            prev_target = _previous_target_index(piece_index, piece_to_query_index)
            next_target = _next_target_index(piece_index, piece_to_query_index)
            start_sec = (
                0.0
                if preserve_outer_boundaries and piece_index == 0
                else (
                    0.0
                    if prev_target is None
                    else source_answers[piece_to_query_index[prev_target]]["end"]
                )
            )
            end_sec = (
                source_dur
                if preserve_outer_boundaries and piece_index == len(pieces) - 1
                else (
                    source_dur
                    if next_target is None
                    else source_answers[piece_to_query_index[next_target]]["start"]
                )
            )
            chunk = _slice_audio(source_audio, sr, start_sec, end_sec)

        if len(chunk) == 0:
            continue
        assembled_parts.append((chunk, piece))

    return _finalize_assembled_audio(assembled_parts, base_text=base_text, sr=sr)


def _assemble_manual_spans_audio(
    *,
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    sr: int,
    base_text: str,
    pieces: List[EditPiece],
    source_span_by_piece_index: Dict[int, Dict[str, float]],
    target_span_by_piece_index: Dict[int, Dict[str, float]],
    preserve_outer_boundaries: bool,
) -> Tuple[np.ndarray, List[SplicePoint]]:
    source_dur = len(source_audio) / sr
    target_dur = len(target_audio) / sr
    assembled_parts: List[Tuple[np.ndarray, EditPiece]] = []

    for piece_index, piece in enumerate(pieces):
        if piece.from_target:
            span = target_span_by_piece_index[piece_index]
            start_sec = (
                0.0 if preserve_outer_boundaries and piece_index == 0 else span["start"]
            )
            end_sec = (
                target_dur
                if preserve_outer_boundaries and piece_index == len(pieces) - 1
                else span["end"]
            )
            chunk = _slice_audio(target_audio, sr, start_sec, end_sec)
        else:
            span = source_span_by_piece_index[piece_index]
            start_sec = (
                0.0 if preserve_outer_boundaries and piece_index == 0 else span["start"]
            )
            end_sec = (
                source_dur
                if preserve_outer_boundaries and piece_index == len(pieces) - 1
                else span["end"]
            )
            chunk = _slice_audio(source_audio, sr, start_sec, end_sec)

        if len(chunk) == 0:
            continue
        assembled_parts.append((chunk, piece))

    return _finalize_assembled_audio(assembled_parts, base_text=base_text, sr=sr)


def _smooth_splice_points(
    *,
    audio: np.ndarray,
    sr: int,
    base_text: str,
    splice_points: List[SplicePoint],
    smoother_url: str,
    regen_margin_sec: Optional[float],
    f5_steps: int,
    f5_cfg_strength: float,
    f5_sway_sampling_coef: float,
    f5_seed: Optional[int],
    debug: bool,
    debug_dir: Optional[str],
    trace_id: Optional[str],
) -> np.ndarray:
    if not splice_points:
        return audio

    assembled_wav = encode_wav(audio, sr)
    smooth_points = []
    for point in splice_points:
        smooth_point = {
            "mode": "smooth_only",
            "position_sec": point.position_sec,
        }
        if regen_margin_sec is not None:
            smooth_point["regen_margin_sec"] = regen_margin_sec
        smooth_points.append(smooth_point)

    payload: Dict[str, Any] = {
        "audio": {
            "data": base64.b64encode(assembled_wav).decode("utf-8"),
            "format": "wav",
        },
        "text": base_text,
        "points": smooth_points,
        "f5_steps": f5_steps,
        "f5_cfg_strength": f5_cfg_strength,
        "f5_sway_sampling_coef": f5_sway_sampling_coef,
        "output_format": "wav",
        "debug": debug,
    }
    if f5_seed is not None:
        payload["f5_seed"] = f5_seed
    if debug and debug_dir:
        payload["debug_dir"] = os.path.join(debug_dir, "smoothing")
    if trace_id is not None:
        payload["trace_id"] = trace_id

    response = requests.post(f"{smoother_url}/smooth", json=payload, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(
            f"Audio smoothing failed: {response.status_code} - {response.text}"
        )

    smoothed_audio, smoothed_sr = decode_audio(response.content)
    return resample_audio(smoothed_audio, smoothed_sr, sr)


def run_assembly(
    source_bytes: bytes,
    target_bytes: bytes,
    edit_tagged_text: str,
    *,
    aligner_url: str = "http://localhost:8029",
    smoother_url: str = "http://localhost:8018",
    language: Optional[str] = None,
    aligner_backend: str = "auto",
    regen_margin_sec: Optional[float] = None,
    smooth_continuation_boundaries: bool = True,
    f5_steps: int = 32,
    f5_cfg_strength: float = 2.0,
    f5_sway_sampling_coef: float = -1.0,
    f5_seed: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    trace_id: Optional[str] = None,
    time_spans_from_source: Optional[List[Tuple[float, float]]] = None,
    time_spans_from_target: Optional[List[Tuple[float, float]]] = None,
    preserve_outer_boundaries: bool = True,
) -> AssemblyResult:
    """Run the full assembly pipeline from `<target>...</target>` markup."""
    base_text, pieces = parse_edit_tagged_text(edit_tagged_text)

    source_audio, source_sr = decode_audio(source_bytes)
    target_audio, target_sr = decode_audio(target_bytes)
    source_audio = resample_audio(source_audio, source_sr, TARGET_SR)
    target_audio = resample_audio(target_audio, target_sr, TARGET_SR)

    has_source = any(not piece.from_target for piece in pieces)
    has_target = any(piece.from_target for piece in pieces)

    target_spans: List[Dict[str, Any]] = [
        {
            "text": piece.text,
            "start_char": piece.start_char,
            "end_char": piece.end_char,
        }
        for piece in pieces
        if piece.from_target
    ]
    source_answers: List[Dict[str, float]] = []
    target_answers: List[Dict[str, float]] = []
    splice_points: List[SplicePoint] = []

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        sf.write(os.path.join(debug_dir, "source.wav"), source_audio, TARGET_SR)
        sf.write(os.path.join(debug_dir, "target.wav"), target_audio, TARGET_SR)

    if not has_target:
        assembled = source_audio
        assembly_mode = "passthrough_source"
    elif not has_source:
        assembled = target_audio
        target_spans = _build_passthrough_target_spans(
            pieces, base_text, len(target_audio) / TARGET_SR
        )
        assembly_mode = "passthrough_target"
    else:
        queries, piece_to_query_index = build_queries(base_text, pieces)
        manual_spans = _prepare_manual_time_spans(
            edit_tagged_text=edit_tagged_text,
            pieces=pieces,
            source_duration_sec=len(source_audio) / TARGET_SR,
            target_duration_sec=len(target_audio) / TARGET_SR,
            time_spans_from_source=time_spans_from_source,
            time_spans_from_target=time_spans_from_target,
        )
        if manual_spans is not None:
            (
                source_span_by_piece_index,
                target_span_by_piece_index,
                source_answers,
                target_answers,
            ) = manual_spans
            target_spans = _build_target_spans(
                pieces,
                piece_to_query_index,
                queries,
                source_answers,
                target_answers,
            )
            assembled, splice_points = _assemble_manual_spans_audio(
                source_audio=source_audio,
                target_audio=target_audio,
                sr=TARGET_SR,
                base_text=base_text,
                pieces=pieces,
                source_span_by_piece_index=source_span_by_piece_index,
                target_span_by_piece_index=target_span_by_piece_index,
                preserve_outer_boundaries=preserve_outer_boundaries,
            )
        else:
            source_wav = encode_wav(source_audio, TARGET_SR)
            target_wav = encode_wav(target_audio, TARGET_SR)
            source_answers = _call_aligner(
                source_wav,
                base_text=base_text,
                queries=queries,
                aligner_url=aligner_url,
                aligner_backend=aligner_backend,
                language=language,
                label="source",
            )
            target_answers = _call_aligner(
                target_wav,
                base_text=base_text,
                queries=queries,
                aligner_url=aligner_url,
                aligner_backend=aligner_backend,
                language=language,
                label="target",
            )
            target_spans = _build_target_spans(
                pieces,
                piece_to_query_index,
                queries,
                source_answers,
                target_answers,
            )
            assembled, splice_points = _assemble_mixed_audio(
                source_audio=source_audio,
                target_audio=target_audio,
                sr=TARGET_SR,
                base_text=base_text,
                pieces=pieces,
                piece_to_query_index=piece_to_query_index,
                source_answers=source_answers,
                target_answers=target_answers,
                preserve_outer_boundaries=preserve_outer_boundaries,
            )
        assembly_mode = "splice_and_smooth"

    if debug and debug_dir:
        sf.write(os.path.join(debug_dir, "00_pre_smoothing.wav"), assembled, TARGET_SR)

    if smooth_continuation_boundaries and len(splice_points) > 0:
        assembled = _smooth_splice_points(
            audio=assembled,
            sr=TARGET_SR,
            base_text=base_text,
            splice_points=splice_points,
            smoother_url=smoother_url,
            regen_margin_sec=regen_margin_sec,
            f5_steps=f5_steps,
            f5_cfg_strength=f5_cfg_strength,
            f5_sway_sampling_coef=f5_sway_sampling_coef,
            f5_seed=f5_seed,
            debug=debug,
            debug_dir=debug_dir,
            trace_id=trace_id,
        )

    if debug and debug_dir:
        sf.write(os.path.join(debug_dir, "01_assembled.wav"), assembled, TARGET_SR)
        with open(
            os.path.join(debug_dir, "debug_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "edit_tagged_text": edit_tagged_text,
                    "base_text": base_text,
                    "language": language,
                    "aligner_backend": aligner_backend,
                    "target_spans": target_spans,
                    "source_answers": source_answers,
                    "target_answers": target_answers,
                    "splice_points": [
                        {"position_sec": point.position_sec}
                        for point in splice_points
                    ],
                    "source_duration": len(source_audio) / TARGET_SR,
                    "target_duration": len(target_audio) / TARGET_SR,
                    "assembled_duration": len(assembled) / TARGET_SR,
                    "assembly_mode": assembly_mode,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return AssemblyResult(
        audio=assembled,
        sample_rate=TARGET_SR,
        splice_points=splice_points,
        source_answers=source_answers,
        target_answers=target_answers,
        target_spans=target_spans,
        details={
            "assembly_mode": assembly_mode,
            "source_duration": len(source_audio) / TARGET_SR,
            "target_duration": len(target_audio) / TARGET_SR,
            "assembled_duration": len(assembled) / TARGET_SR,
            "n_splice_points": len(splice_points),
            "n_target_spans": len(target_spans),
            "base_text_length": len(base_text),
        },
    )
