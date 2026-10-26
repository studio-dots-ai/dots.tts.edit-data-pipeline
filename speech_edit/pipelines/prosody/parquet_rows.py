"""Direct parquet row construction for prosody edit outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def prosody_rows(
    *,
    stage3_state: Dict[str, Any],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    from speech_edit.pipelines.prosody.tag_utils import (
        describe_tag_diff,
        strip_tags,
    )

    def edited_wave_id(source_id: str, target_id: str) -> Optional[str]:
        if source_id == ORIGINAL_WAVE_ID and target_id != ORIGINAL_WAVE_ID:
            return target_id
        if target_id == ORIGINAL_WAVE_ID and source_id != ORIGINAL_WAVE_ID:
            return source_id
        return None

    def transcript_for_label(label: str, plain_text: str) -> str:
        stripped = strip_tags(label)
        return stripped if stripped.strip() else plain_text

    def spans_from_regions(regions: Any) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        if not isinstance(regions, list):
            return spans
        for region in regions:
            if not isinstance(region, dict):
                continue
            try:
                start = float(region["start_sec"])
                end = float(region["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            spans.append(
                {
                    "text": str(region.get("text") or region.get("words") or "").strip(),
                    "span": [start, end],
                }
            )
        return spans

    def spans_from_edits(edits: Any, span_key: str) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        if not isinstance(edits, list):
            return spans
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            span = edit.get(span_key)
            if not isinstance(span, list) or len(span) != 2:
                continue
            try:
                start = float(span[0])
                end = float(span[1])
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            spans.append({"text": str(edit.get("words") or "").strip(), "span": [start, end]})
        return spans

    def collect_edit_params(tag_diff: List[Dict[str, Any]]) -> Dict[str, Any]:
        pitch_semitones: List[float] = []
        rate_factors: List[float] = []
        edited_words = 0
        for diff_entry in tag_diff:
            has_edit = False
            for side in ("source", "target"):
                side_info = diff_entry.get(side, {})
                if side_info.get("semitones") is not None:
                    pitch_semitones.append(float(side_info["semitones"]))
                    has_edit = True
                if side_info.get("factor") is not None:
                    rate_factors.append(float(side_info["factor"]))
                    has_edit = True
                if side_info.get("edit_type", "none") != "none":
                    has_edit = True
            if has_edit:
                edited_words += 1
        return {
            "pitch_semitones": pitch_semitones,
            "rate_factors": rate_factors,
            "num_edited_words": edited_words,
        }

    pipeline_input = stage3_state["pipeline_input"]
    metadata = stage3_state["metadata"]
    plain_text = metadata.get("plain_text", "")
    language = metadata.get("language") or metadata.get("stage1_info", {}).get("language")
    tagged_texts = metadata.get("tagged_texts", {})
    waves = metadata.get("waves", {})
    stage1_info = metadata.get("stage1_info", {})
    wave_specs = metadata.get("wave_specs") or stage1_info.get("wave_specs") or {}
    sample_rate = int(metadata.get("sample_rate") or pipeline_input.sample_rate)
    sample_id = str(metadata.get("sample_id") or stage3_state.get("sample_id"))

    rows: List[Dict[str, Any]] = []
    for pair in stage3_state.get("pair_results", []):
        source_id = str(pair.get("source_id", ""))
        target_id = str(pair.get("target_id", ""))
        wave_id = edited_wave_id(source_id, target_id)
        if wave_id is None:
            continue
        wave_spec = wave_specs.get(str(wave_id))
        if not isinstance(wave_spec, dict):
            continue
        src_encoded, src_duration = encode_audio_flac_base64(
            pair["source_audio_bytes"],
            sample_rate,
        )
        tgt_encoded, tgt_duration = encode_audio_flac_base64(
            pair["target_audio_bytes"],
            sample_rate,
        )
        src_tagged_text = plain_text if source_id == ORIGINAL_WAVE_ID else tagged_texts.get(source_id, "")
        tgt_tagged_text = plain_text if target_id == ORIGINAL_WAVE_ID else tagged_texts.get(target_id, "")
        tag_diff_data = describe_tag_diff(src_tagged_text, tgt_tagged_text)
        if not tag_diff_data and isinstance(pair.get("tag_diff"), list):
            tag_diff_data = pair["tag_diff"]
        params = collect_edit_params(tag_diff_data)
        total_words = len(plain_text.split()) if plain_text else 1
        edit_metadata = {
            "edit_kind": str(wave_spec.get("edit_kind", "unknown")),
            "span_granularity": str(wave_spec.get("span_granularity", "unknown")),
            "num_regions": int(wave_spec.get("num_regions", 0)),
            "num_pitch_regions": int(wave_spec.get("num_pitch_regions", 0)),
            "num_rate_regions": int(wave_spec.get("num_rate_regions", 0)),
            "pitch_semitones": params["pitch_semitones"],
            "rate_factors": params["rate_factors"],
            "num_edited_words": int(params["num_edited_words"]),
            "num_total_words": total_words,
        }
        segments: Dict[str, Dict[str, Any]] = {}
        wave_meta = waves.get(str(wave_id), {}) if wave_id else {}
        result = wave_meta.get("result") if isinstance(wave_meta, dict) else {}
        result = result if isinstance(result, dict) else {}
        alignment = result.get("alignment") if isinstance(result.get("alignment"), dict) else {}
        edits = result.get("edits_applied", [])
        for side, side_wave_id, label, duration in (
            ("src", source_id, src_tagged_text, src_duration),
            ("tgt", target_id, tgt_tagged_text, tgt_duration),
        ):
            if side_wave_id == ORIGINAL_WAVE_ID:
                edit_time_spans = spans_from_regions(alignment.get("edit_regions"))
            else:
                edit_time_spans = spans_from_edits(edits, "target_span")
            segment = {
                "chunk_id": side,
                "wave_id": side_wave_id,
                "is_original": side_wave_id == ORIGINAL_WAVE_ID,
                "start": 0.0,
                "end": duration,
                "label": label,
                "transcript": transcript_for_label(label, plain_text),
                "edit_time_spans": edit_time_spans,
                "alignment": alignment,
            }
            if language:
                segment["language"] = language
            segments[side] = segment
        pair_dir = str(pair.get("pair_dir_name") or f"{source_id}_to_{target_id}")
        pair_metadata = {
            "sample_id": sample_id,
            "src_id": source_id,
            "tgt_id": target_id,
            "edited_generation_id": _edited_generation_id(source_id, target_id),
            "original_side": "src" if source_id == ORIGINAL_WAVE_ID else "tgt",
            "direction": f"{source_id}_to_{target_id}",
            "generation_id": str(wave_id or ""),
            "source_audio_file": pair.get("source_audio_file", "source.wav"),
            "target_audio_file": pair.get("target_audio_file", "target.wav"),
            "language": language,
            "wave_id": str(wave_id or ""),
            "clip_id": metadata.get("clip_id") or pipeline_input.clip_id,
            "input_id": metadata.get("input_id")
            or pipeline_input.metadata.get("input_id"),
            "pair_dir_name": pair_dir,
            "plain_text": plain_text,
            "normed_text": metadata.get("normed_text") or pipeline_input.normed_text,
        }
        pair_metadata = {
            key: value for key, value in pair_metadata.items() if value is not None
        }
        rows.append(
            {
                "from_dataset": dataset_name,
                "audio_id": f"{sample_id}--{pair_dir}",
                "sample_rate": sample_rate,
                "audios": json_dumps({"src": src_encoded, "tgt": tgt_encoded}),
                "duration": json_dumps({"src": src_duration, "tgt": tgt_duration}),
                "segments": json_dumps(segments),
                "instructions": json_dumps(pair.get("instructions", {})),
                "tag_diff": json_dumps(tag_diff_data),
                "eval_results": json_dumps(
                    pair.get("eval_results", {})
                ),
                "pair_metadata": json_dumps(pair_metadata),
                "edit_metadata": json_dumps(edit_metadata),
            }
        )
    return rows


class ProsodyParquetRowProcessor(ParquetRowProcessor):
    def rows_from_payload(
        self,
        payload: Any,
        *,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        return prosody_rows(
            stage3_state=payload,
            dataset_name=dataset_name,
        )
