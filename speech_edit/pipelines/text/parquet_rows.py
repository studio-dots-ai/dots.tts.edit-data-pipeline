"""Direct parquet row construction for tagged text edit outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from speech_edit.data.parquet.rows.audio import (
    encode_audio_flac_base64,
    json_dumps,
)
from speech_edit.data.parquet.rows.base import ParquetRowProcessor

ORIGINAL_WAVE_ID = "original"


def _edited_generation_id(source_id: str, target_id: str) -> str:
    if source_id == ORIGINAL_WAVE_ID:
        return target_id
    if target_id == ORIGINAL_WAVE_ID:
        return source_id
    return ""


def _count_tag_diff(tag_diff: Any) -> Dict[str, int]:
    counts = {"sub": 0, "del": 0, "ins": 0}
    if not isinstance(tag_diff, list):
        return counts
    for item in tag_diff:
        if isinstance(item, dict) and item.get("kind") in counts:
            counts[str(item["kind"])] += 1
    return counts


def text_edit_rows(
    *,
    sample_id: str,
    original_audio_bytes: bytes,
    generated_audio_bytes: Dict[str, bytes],
    pair_results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    dataset_name: str,
    sample_rate: int,
) -> List[Dict[str, Any]]:
    """Build rows for tagged text editing."""
    audio_by_id = {ORIGINAL_WAVE_ID: original_audio_bytes, **generated_audio_bytes}
    rows: List[Dict[str, Any]] = []
    for pair in pair_results:
        source_id = str(pair.get("source_id") or "")
        target_id = str(pair.get("target_id") or "")
        if not source_id or not target_id:
            continue
        if source_id not in audio_by_id or target_id not in audio_by_id:
            continue
        segments = pair.get("segments")
        if not isinstance(segments, dict):
            continue
        instructions = pair.get("instructions")
        if not isinstance(instructions, dict) or "st" not in instructions:
            continue

        src_audio, src_duration = encode_audio_flac_base64(audio_by_id[source_id], sample_rate)
        tgt_audio, tgt_duration = encode_audio_flac_base64(audio_by_id[target_id], sample_rate)

        segments = json.loads(json.dumps(segments, ensure_ascii=False))
        if "src" not in segments or "tgt" not in segments:
            continue
        segments["src"].setdefault("edit_time_spans", [])
        segments["tgt"].setdefault("edit_time_spans", [])
        segments["src"]["end"] = src_duration
        segments["tgt"]["end"] = tgt_duration

        pair_dir_name = str(pair.get("pair_dir_name") or f"{source_id}_to_{target_id}")
        pair_metadata = {
            "sample_id": sample_id,
            "src_id": source_id,
            "tgt_id": target_id,
            "edited_generation_id": _edited_generation_id(source_id, target_id),
            "original_side": "src" if source_id == ORIGINAL_WAVE_ID else "tgt",
            "direction": pair.get("direction", ""),
            "generation_id": pair.get("generation_id", ""),
            "source_audio_file": pair.get("source_audio_file", "source.wav"),
            "target_audio_file": pair.get("target_audio_file", "target.wav"),
        }
        source_metadata = metadata.get("source_metadata", {})
        if isinstance(source_metadata, dict) and source_metadata.get("language"):
            pair_metadata["language"] = source_metadata["language"]

        config = metadata.get("config", {})
        edit_distance = pair.get("target_edit_distance")
        if not isinstance(edit_distance, dict):
            edit_distance = {}
        edit_metadata = {
            "edit_kind": "text",
            "op_counts": _count_tag_diff(pair.get("tag_diff", [])),
            "num_regions": len(pair.get("tag_diff", []))
            if isinstance(pair.get("tag_diff"), list)
            else 0,
            "duration_mode": config.get("duration_mode") if isinstance(config, dict) else None,
            "requested_span_level": pair.get("requested_span_level"),
            "requested_op_count": pair.get("requested_op_count"),
            "target_edit_distance_pct": edit_distance.get("max_pct"),
            "target_edit_distance_source_pct": edit_distance.get("source_pct"),
            "target_edit_distance_target_pct": edit_distance.get("target_pct"),
            "target_edit_distance_metric": edit_distance.get("metric"),
            "target_edit_distance_denominator": "source_or_target",
        }

        rows.append(
            {
                "from_dataset": dataset_name,
                "audio_id": f"{sample_id}--{pair_dir_name}",
                "sample_rate": sample_rate,
                "audios": json_dumps({"src": src_audio, "tgt": tgt_audio}),
                "duration": json_dumps({"src": src_duration, "tgt": tgt_duration}),
                "segments": json_dumps(segments),
                "instructions": json_dumps(instructions),
                "tag_diff": json_dumps(pair.get("tag_diff", [])),
                "eval_results": json_dumps(pair.get("eval_results", {})),
                "pair_metadata": json_dumps(pair_metadata),
                "edit_metadata": json_dumps(edit_metadata),
            }
        )
    return rows


class TextEditParquetRowProcessor(ParquetRowProcessor):
    def rows_from_payload(
        self,
        payload: Any,
        *,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        result = getattr(payload, "result", {})
        segment = getattr(payload, "segment")
        return text_edit_rows(
            sample_id=str(getattr(payload, "sample_id")),
            original_audio_bytes=segment.audio_bytes,
            generated_audio_bytes=dict(result.get("generated_audio_bytes", {})),
            pair_results=list(result.get("pair_results", [])),
            metadata=dict(result.get("metadata", {})),
            dataset_name=dataset_name,
            sample_rate=int(segment.sample_rate),
        )
