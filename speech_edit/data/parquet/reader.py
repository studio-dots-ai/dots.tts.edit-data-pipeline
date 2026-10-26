"""Lightweight parquet segment reader for distributed-pipeline source workers."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import soundfile as sf

from speech_edit.data.language import classify_text_language, normalize_language_code


@dataclass
class Segment:
    clip_id: str
    text: str
    audio_bytes: bytes
    duration: float
    sample_rate: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_segments(row: pd.Series) -> List[dict]:
    if "segments" not in row or _is_missing(row["segments"]):
        return []
    seg_data = row["segments"]
    if isinstance(seg_data, str):
        seg_data = json.loads(seg_data)
    if isinstance(seg_data, list):
        return seg_data
    if isinstance(seg_data, dict):
        return [
            segment
            for key in ("src", "tgt")
            if isinstance((segment := seg_data.get(key)), dict)
        ]
    return []


def _parse_audios(row: pd.Series) -> Optional[Dict[str, Any]]:
    if "audios" not in row or _is_missing(row["audios"]):
        return None
    aud_data = row["audios"]
    if isinstance(aud_data, str):
        return json.loads(aud_data)
    if isinstance(aud_data, dict):
        return aud_data
    return None


def _decode_audio_payload(audio_payload: Any) -> Optional[bytes]:
    if _is_missing(audio_payload):
        return None
    if isinstance(audio_payload, memoryview):
        return audio_payload.tobytes()
    if isinstance(audio_payload, bytes):
        return audio_payload
    if isinstance(audio_payload, bytearray):
        return bytes(audio_payload)
    if isinstance(audio_payload, str):
        return base64.b64decode(audio_payload, validate=True)
    return None


def _filter_metadata_fields(row: pd.Series, excluded_keys: set[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded_keys or _is_missing(value):
            continue
        metadata[key] = value.tobytes() if isinstance(value, memoryview) else value
    return metadata


def _row_to_segment_nested(
    row: pd.Series,
    row_idx: int,
    parquet_path: Path,
    *,
    source_file_index: int,
    min_single_duration: float,
    max_single_duration: float,
) -> Optional[Segment]:
    segments = _parse_segments(row)
    audios = _parse_audios(row)
    if not segments or not audios:
        return None
    seg = segments[0]
    chunk_id = seg.get("chunk_id", "000000")
    if chunk_id not in audios:
        return None
    duration = seg.get("duration") or seg.get("audio_length")
    if duration is None:
        start, end = seg.get("start"), seg.get("end")
        if start is not None and end is not None:
            duration = float(end) - float(start)
    if duration is None:
        return None
    duration = float(duration)
    if duration < min_single_duration or duration > max_single_duration:
        return None
    audio_bytes = _decode_audio_payload(audios[chunk_id])
    if audio_bytes is None:
        return None
    audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))
    from_dataset = row.get("from_dataset", parquet_path.stem)
    audio_id = row.get("audio_id", f"row_{row_idx}")
    metadata: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "start": seg.get("start"),
        "end": seg.get("end"),
        "source_parquet_path": str(parquet_path.expanduser().resolve()),
        "source_row_idx": int(row_idx),
        "source_file_index": source_file_index,
        "source_schema": "nested",
    }
    if seg.get("label"):
        metadata["label"] = seg["label"]
    if seg.get("gemini_captions"):
        metadata["gemini_captions"] = seg["gemini_captions"]
    return Segment(
        clip_id=f"{from_dataset}_{audio_id}_{chunk_id}",
        text=str(seg.get("text") or seg.get("label") or ""),
        audio_bytes=audio_bytes,
        duration=len(audio_np) / sample_rate,
        sample_rate=int(sample_rate),
        metadata=metadata,
    )


def _row_to_segment_flat(
    row: pd.Series,
    row_idx: int,
    parquet_path: Path,
    *,
    source_file_index: int,
    min_single_duration: float,
    max_single_duration: float,
) -> Optional[Segment]:
    audio_bytes = _decode_audio_payload(row.get("audio"))
    if audio_bytes is None:
        return None
    duration = row.get("duration")
    if duration is None:
        return None
    duration = float(duration)
    if duration < min_single_duration or duration > max_single_duration:
        return None
    audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))
    metadata = _filter_metadata_fields(
        row,
        excluded_keys={"audio", "text", "label", "duration", "sample_rate", "clip_id"},
    )
    metadata.update(
        {
            "source_parquet_path": str(parquet_path.expanduser().resolve()),
            "source_row_idx": int(row_idx),
            "source_file_index": source_file_index,
            "source_schema": "flat",
        }
    )
    clip_id = row.get("clip_id") or row.get("id") or f"row_{row_idx}"
    return Segment(
        clip_id=str(clip_id),
        text=str(row.get("text") or row.get("label") or ""),
        audio_bytes=audio_bytes,
        duration=len(audio_np) / sample_rate,
        sample_rate=int(sample_rate),
        metadata=metadata,
    )


def row_to_segment(
    row: pd.Series,
    row_idx: int,
    parquet_path: Path,
    *,
    source_file_index: int,
    min_single_duration: float,
    max_single_duration: float,
) -> Optional[Segment]:
    kwargs = {
        "source_file_index": source_file_index,
        "min_single_duration": min_single_duration,
        "max_single_duration": max_single_duration,
    }
    if "segments" in row and "audios" in row:
        return _row_to_segment_nested(row, row_idx, parquet_path, **kwargs)
    if "audio" in row:
        return _row_to_segment_flat(row, row_idx, parquet_path, **kwargs)
    return None


def _requested_language(cfg: Mapping[str, Any]) -> Optional[str]:
    requested = cfg.get("sample_language", cfg.get("source_language"))
    if requested is None or str(requested).strip() == "":
        return None
    code = normalize_language_code(str(requested))
    return "zh" if code == "zho" else "en" if code == "eng" else str(requested).lower()


def _segment_language(segment: Segment) -> str:
    raw = segment.metadata.get("language") or segment.metadata.get("sample_language")
    if raw:
        code = normalize_language_code(str(raw), text=segment.text)
        return "zh" if code == "zho" else "en" if code == "eng" else str(raw).lower()
    return classify_text_language(segment.text)


def _duration_bucket_policy(cfg: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    policy = cfg.get("duration_bucket_policy")
    if policy is None:
        return None
    if not isinstance(policy, Mapping):
        raise ValueError("duration_bucket_policy must be an object")
    raw_buckets = policy.get("buckets")
    if raw_buckets is None:
        return None
    if not isinstance(raw_buckets, Mapping) or not raw_buckets:
        raise ValueError("duration_bucket_policy.buckets must be a non-empty object")
    buckets: List[Dict[str, Any]] = []
    for name, raw_bucket in raw_buckets.items():
        if not isinstance(raw_bucket, Mapping):
            raise ValueError(f"duration_bucket_policy.buckets.{name} must be an object")
        try:
            min_sec = float(raw_bucket["min"])
            max_sec = float(raw_bucket["max"])
            weight = float(raw_bucket["weight"])
        except KeyError as exc:
            raise ValueError(
                f"duration_bucket_policy.buckets.{name} requires min, max, and weight"
            ) from exc
        if max_sec <= min_sec:
            raise ValueError(f"duration_bucket_policy.buckets.{name}.max must be greater than min")
        if weight < 0:
            raise ValueError(f"duration_bucket_policy.buckets.{name}.weight must be non-negative")
        buckets.append({"name": str(name), "min": min_sec, "max": max_sec, "weight": weight})
    if not any(bucket["weight"] > 0 for bucket in buckets):
        raise ValueError("duration_bucket_policy.buckets must contain at least one positive weight")
    return buckets


def _duration_bucket_for_value(
    duration: float,
    buckets: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for index, bucket in enumerate(buckets):
        is_last = index == len(buckets) - 1
        if duration >= bucket["min"] and (duration < bucket["max"] or (is_last and duration <= bucket["max"])):
            return bucket
    return None


def _duration_bucket_accepts(
    *,
    duration: float,
    buckets: Optional[List[Dict[str, Any]]],
    seed: int,
    source_file_index: int,
    row_idx: int,
) -> tuple[bool, Optional[str]]:
    if buckets is None:
        return True, None
    bucket = _duration_bucket_for_value(duration, buckets)
    if bucket is None:
        return False, None
    max_weight = max(float(item["weight"]) for item in buckets)
    weight = float(bucket["weight"])
    if weight <= 0:
        return False, str(bucket["name"])
    if weight >= max_weight:
        return True, str(bucket["name"])
    digest = hashlib.sha256(f"{seed}:{source_file_index}:{row_idx}".encode("utf-8")).hexdigest()
    draw = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return draw < (weight / max_weight), str(bucket["name"])


def read_segments_from_parquet_payload(
    payload: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> List[Segment]:
    parquet_path = Path(str(payload["parquet_path"]))
    source_file_index = int(payload.get("source_file_index", 0))
    min_single_duration = float(cfg.get("min_duration", cfg.get("min_single_duration", 2.0)))
    max_single_duration = float(cfg.get("max_duration", cfg.get("max_single_duration", 30.0)))
    requested_language = _requested_language(cfg)
    duration_buckets = _duration_bucket_policy(cfg)
    seed = int(cfg.get("seed", 0) or 0)
    df = pd.read_parquet(parquet_path)
    row_start = max(0, int(payload.get("row_start", 0)))
    row_stop = min(len(df), int(payload.get("row_stop", len(df))))
    row_indices = list(range(row_start, row_stop))
    if bool(cfg.get("shuffle_source_rows", True)):
        seed = int(cfg.get("seed", 0) or 0)
        rng = random.Random(seed * 131 + source_file_index * 997 + row_start)
        rng.shuffle(row_indices)

    segments: List[Segment] = []
    for row_idx in row_indices:
        segment = row_to_segment(
            df.iloc[row_idx],
            row_idx,
            parquet_path,
            source_file_index=source_file_index,
            min_single_duration=min_single_duration,
            max_single_duration=max_single_duration,
        )
        if segment is not None:
            if requested_language and _segment_language(segment) != requested_language:
                continue
            if requested_language:
                segment.metadata.setdefault("sample_language", requested_language)
                segment.metadata.setdefault("language", requested_language)
            accepted, bucket_name = _duration_bucket_accepts(
                duration=segment.duration,
                buckets=duration_buckets,
                seed=seed,
                source_file_index=source_file_index,
                row_idx=row_idx,
            )
            if not accepted:
                continue
            if bucket_name:
                segment.metadata.setdefault("duration_bucket", bucket_name)
            segments.append(segment)
    return segments
