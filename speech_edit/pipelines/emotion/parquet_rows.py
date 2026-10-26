"""Direct parquet row construction for emotion/style edit outputs."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from speech_edit.data.parquet.rows.audio import (
    encode_audio_flac_base64,
    json_dumps,
)
from speech_edit.data.parquet.rows.base import ParquetRowProcessor

from .emotion_acoustic_effect import (
    METRIC_KEY as EMOTION_ACOUSTIC_EFFECT_KEY,
)
from .emotion_acoustic_effect import METRIC_VERSION

EVAL_RESULTS_VERSION = "v3"


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _extract_instructions(record: Dict[str, Any]) -> Dict[str, str]:
    instructions = record.get("instructions")
    if isinstance(instructions, dict):
        return {
            "nl": str(instructions.get("nl", instructions.get("instruction", ""))),
            "st": str(instructions.get("st", "")),
        }
    return {
        "nl": str(record.get("nl_instruction", record.get("instruction", ""))),
        "st": str(record.get("st_instruction", "")),
    }


def _normalize_edit_time_spans(eval_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    normalized: Dict[str, List[Dict[str, Any]]] = {"src": [], "tgt": []}
    for side in ("src", "tgt"):
        side_payload = eval_results.get(side)
        side_spans = side_payload.get("edit_time_spans", []) if isinstance(side_payload, dict) else []
        if not isinstance(side_spans, list):
            continue
        for item in side_spans:
            if not isinstance(item, dict):
                continue
            span = item.get("span")
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            try:
                start_sec = float(span[0])
                end_sec = float(span[1])
            except (TypeError, ValueError):
                continue
            if end_sec <= start_sec:
                continue
            normalized[side].append(
                {
                    "text": str(item.get("text", "")).strip(),
                    "span": [start_sec, end_sec],
                }
            )
    return normalized


def _compact_eval_results(eval_results: Any) -> Dict[str, Any]:
    if not isinstance(eval_results, dict):
        return {
            "version": EVAL_RESULTS_VERSION,
            "src": {},
            "tgt": {},
            EMOTION_ACOUSTIC_EFFECT_KEY: {
                "version": METRIC_VERSION,
                "status": "error",
                "error": "missing eval_results",
                "regions": [],
            },
            "error": "missing eval_results",
        }

    payload: Dict[str, Any] = {"version": EVAL_RESULTS_VERSION}
    for side in ("src", "tgt"):
        side_payload = eval_results.get(side)
        normalized_side = dict(side_payload) if isinstance(side_payload, dict) else {}
        normalized_side.pop("emotion", None)
        payload[side] = normalized_side
    for key in (
        EMOTION_ACOUSTIC_EFFECT_KEY,
        "spk_sim",
        "eval_time_sec",
        "wdtw",
        "wdtw_result",
        "error",
    ):
        if key in eval_results:
            payload[key] = eval_results[key]
    payload.setdefault(
        EMOTION_ACOUSTIC_EFFECT_KEY,
        {
            "version": METRIC_VERSION,
            "status": "error",
            "error": "missing emotion acoustic-effect evaluation",
            "regions": [],
        },
    )
    if "edit_time_spans_error" in eval_results:
        payload["edit_time_spans_error"] = eval_results["edit_time_spans_error"]
    return payload


def _build_pair_metadata(group: Any, *, sample_id: str) -> Dict[str, Any]:
    group_metadata = getattr(group, "metadata", {}) or {}
    synthesis_profile = group_metadata.get("synthesis_profile")
    if not isinstance(synthesis_profile, dict):
        synthesis_profile = {}
    emotion_edit_plan = group_metadata.get("emotion_edit_plan")
    if not isinstance(emotion_edit_plan, dict):
        emotion_edit_plan = {}
    prompted_edit_mode = group_metadata.get("prompted_edit_mode") or emotion_edit_plan.get(
        "edit_mode"
    )
    selected_window = synthesis_profile.get("selected_text_window")
    if not isinstance(selected_window, dict):
        selected_window = {}
    estimated_duration = synthesis_profile.get("estimated_duration_sec")
    if estimated_duration is None:
        estimated_duration = selected_window.get("estimated_duration_sec")
    return _json_compatible({
        "sample_id": sample_id,
        "language": group_metadata.get("language"),
        "prompted_edit_mode": prompted_edit_mode,
        "synthesis_profile": synthesis_profile,
        "synthesis_backend": group_metadata.get("synthesis_backend")
        or synthesis_profile.get("synthesis_backend"),
        "synthesis_mode": group_metadata.get("synthesis_mode")
        or synthesis_profile.get("synthesis_mode"),
        "emotion_condition_mode": group_metadata.get("emotion_condition_mode")
        or synthesis_profile.get("emotion_condition_mode"),
        "estimated_duration_sec": estimated_duration,
        "ted_estimated_duration_soft_limit_sec": group_metadata.get(
            "ted_estimated_duration_soft_limit_sec"
        )
        or synthesis_profile.get("ted_estimated_duration_soft_limit_sec"),
        "ted_max_generated_duration_sec": group_metadata.get(
            "ted_max_generated_duration_sec"
        )
        or synthesis_profile.get("ted_max_generated_duration_sec"),
        "ted_max_mel_tokens": group_metadata.get("ted_max_mel_tokens")
        or synthesis_profile.get("ted_max_mel_tokens"),
        "selected_text_window": selected_window,
        "pair_change_summary": group_metadata.get("pair_change_summary"),
        "speaker_prompt_metadata": group_metadata.get("speaker_prompt_metadata"),
    })


def emotion_style_rows(
    *,
    group: Any,
    synth_result: Any,
    eval_result: Dict[str, Any],
    sample_id: str,
    dataset_name: str,
    sample_rate: int = 24000,
) -> List[Dict[str, Any]]:
    from speech_edit.pipelines.emotion.tag_utils import describe_tag_diff

    eval_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
    for pair_eval in eval_result.get("pair_evaluations", []):
        if isinstance(pair_eval, dict):
            eval_by_pair[
                (str(pair_eval.get("source_id", "")), str(pair_eval.get("target_id", "")))
            ] = pair_eval

    edit_meta_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
    for edit_pair in getattr(group, "edit_pairs", []) or []:
        edit_meta_by_pair[(edit_pair.source_id, edit_pair.target_id)] = {
            "source_id": edit_pair.source_id,
            "target_id": edit_pair.target_id,
            "instructions": {
                "nl": edit_pair.nl_instruction,
                "st": edit_pair.st_instruction,
            },
        }

    speaker_prompt_encoded = ""
    speaker_prompt_audio = getattr(group, "speaker_prompt_audio", b"")
    if speaker_prompt_audio:
        speaker_prompt_encoded, _ = encode_audio_flac_base64(speaker_prompt_audio, sample_rate)

    pair_metadata = _build_pair_metadata(group, sample_id=sample_id)
    rows: List[Dict[str, Any]] = []
    for pair in getattr(synth_result, "edit_pairs", []) or []:
        src_audio, src_duration = encode_audio_flac_base64(pair.source_audio, sample_rate)
        tgt_audio, tgt_duration = encode_audio_flac_base64(pair.target_audio, sample_rate)
        src_id = str(pair.source_id)
        tgt_id = str(pair.target_id)
        pair_eval = eval_by_pair.get((src_id, tgt_id), {})
        eval_payload = _compact_eval_results(pair_eval.get("eval_results"))
        src_tagged_text = getattr(group, "tagged_texts", {}).get(src_id, "")
        tgt_tagged_text = getattr(group, "tagged_texts", {}).get(tgt_id, "")
        edit_time_spans = _normalize_edit_time_spans(eval_payload)
        segments = [
            {
                "chunk_id": "src",
                "start": 0.0,
                "end": src_duration,
                "label": src_tagged_text,
                "edit_time_spans": edit_time_spans["src"],
                "alignment": getattr(pair, "source_alignment", {}),
            },
            {
                "chunk_id": "tgt",
                "start": 0.0,
                "end": tgt_duration,
                "label": tgt_tagged_text,
                "edit_time_spans": edit_time_spans["tgt"],
                "alignment": getattr(pair, "target_alignment", {}),
            },
        ]
        meta = edit_meta_by_pair.get((src_id, tgt_id), {})
        instructions = _extract_instructions(meta) if meta else {
            "nl": str(getattr(pair, "nl_instruction", "")),
            "st": str(getattr(pair, "st_instruction", "")),
        }
        rows.append(
            {
                "from_dataset": dataset_name,
                "audio_id": f"{sample_id}--{src_id}_to_{tgt_id}",
                "audio_path": "dist_pipeline",
                "audios": json_dumps({"src": src_audio, "tgt": tgt_audio}),
                "duration": json_dumps({"src": src_duration, "tgt": tgt_duration}),
                "segments": json_dumps(segments),
                "tagged_texts": json_dumps({"src": src_tagged_text, "tgt": tgt_tagged_text}),
                "instructions": json_dumps(instructions),
                "tag_diff": json_dumps(describe_tag_diff(src_tagged_text, tgt_tagged_text)),
                "sub_type": getattr(group, "sub_type", ""),
                "plain_text": getattr(group, "plain_text", ""),
                "speaker_prompt_text": getattr(group, "speaker_prompt_text", ""),
                "pair_metadata": json_dumps(pair_metadata),
                "speaker_prompt": json_dumps(
                    {
                        "audio": speaker_prompt_encoded,
                        "text": getattr(group, "speaker_prompt_text", ""),
                    }
                ),
                "eval_results": json_dumps(eval_payload),
            }
        )
    return rows


class EmotionStyleParquetRowProcessor(ParquetRowProcessor):
    def rows_from_payload(
        self,
        payload: Any,
        *,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        return emotion_style_rows(
            group=payload["group"],
            synth_result=payload["synth_result"],
            eval_result=payload["eval_result"],
            sample_id=str(payload["sample_id"]),
            dataset_name=dataset_name,
        )
