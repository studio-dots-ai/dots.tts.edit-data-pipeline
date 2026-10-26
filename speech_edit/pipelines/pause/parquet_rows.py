"""Direct parquet row construction for pause edit outputs."""

from __future__ import annotations

from typing import Any, Dict, List

from speech_edit.data.parquet.rows.audio import (
    encode_audio_flac_base64,
    json_dumps,
)
from speech_edit.data.parquet.rows.base import ParquetRowProcessor
from speech_edit.pipelines.pause.evaluator import serialize_eval_results
from speech_edit.pipelines.pause.tag_utils import (
    build_pause_nl_instructions,
    parse_instruction_pause_tags,
    reverse_operation,
)


def _build_edit_metadata(*, instruction_st: str, source_text: str) -> Dict[str, Any]:
    tags = parse_instruction_pause_tags(instruction_st, source_text)
    return {
        "edit_kind": "pause",
        "operations": [
            {
                "act": tag.act,
                "level": int(tag.level),
                "char_idx": int(tag.char_idx),
                "boundary_type": "punct" if getattr(tag, "punctuation", None) else "nonpunct",
                "left_context": str(getattr(tag, "left_context", "")),
                "right_context": str(getattr(tag, "right_context", "")),
            }
            for tag in tags
        ],
    }


def _segments_payload(
    *,
    src_duration: float,
    tgt_duration: float,
    src_label: str,
    tgt_label: str,
) -> Dict[str, Any]:
    return {
        "src": {
            "chunk_id": "src",
            "start": 0.0,
            "end": src_duration,
            "label": src_label,
            "edit_time_spans": [],
        },
        "tgt": {
            "chunk_id": "tgt",
            "start": 0.0,
            "end": tgt_duration,
            "label": tgt_label,
            "edit_time_spans": [],
        },
    }


def _row_for_direction(
    *,
    dataset_name: str,
    sample_id: str,
    profile: Any,
    sample_rate: int,
    language: Any,
    direction: str,
    src_id: str,
    tgt_id: str,
    src_audio: str,
    tgt_audio: str,
    src_duration: float,
    tgt_duration: float,
    src_label: str,
    tgt_label: str,
    src_tagged_text: str,
    tgt_tagged_text: str,
    instruction_st: str,
    eval_result: Any,
    pause_edit_plan: Dict[str, Any],
) -> Dict[str, Any]:
    pair_metadata = {
        "sample_id": sample_id,
        "src_id": src_id,
        "tgt_id": tgt_id,
        "original_side": "src" if src_id == "original" else "tgt",
        "direction": direction,
        "profile_id": profile.profile_id,
        "language": language,
    }
    return {
        "from_dataset": dataset_name,
        "audio_id": f"{sample_id}--profile_{profile.profile_id}--{direction}",
        "sample_rate": sample_rate,
        "audios": json_dumps({"src": src_audio, "tgt": tgt_audio}),
        "duration": json_dumps({"src": src_duration, "tgt": tgt_duration}),
        "segments": json_dumps(
            _segments_payload(
                src_duration=src_duration,
                tgt_duration=tgt_duration,
                src_label=src_label,
                tgt_label=tgt_label,
            )
        ),
        "tagged_texts": json_dumps({"src": src_tagged_text, "tgt": tgt_tagged_text}),
        "instructions": json_dumps(
            {
                "nl": build_pause_nl_instructions(instruction_st),
                "st": instruction_st,
            }
        ),
        "eval_results": json_dumps(
            serialize_eval_results(eval_result, direction=direction)
        ),
        "pair_metadata": json_dumps(pair_metadata),
        "edit_metadata": json_dumps(
            _build_edit_metadata(instruction_st=instruction_st, source_text=src_label)
        ),
        "pause_edit_plan": json_dumps(pause_edit_plan),
    }


def pause_edit_rows(*, payload: Any, dataset_name: str) -> List[Dict[str, Any]]:
    work_item = getattr(payload, "work_item")
    synth_pair = getattr(payload, "synth_pair")
    eval_result = getattr(payload, "eval_result", None)

    profile = work_item.profile
    stage1_info = dict(work_item.metadata or {})
    normed_text = str(work_item.normed_text)
    language = stage1_info.get("language")
    sample_rate = int(stage1_info.get("sample_rate") or 24000)

    src_encoded, src_duration = encode_audio_flac_base64(
        synth_pair.original_audio_bytes,
        sample_rate,
    )
    tgt_encoded, tgt_duration = encode_audio_flac_base64(
        synth_pair.generated_audio_bytes,
        sample_rate,
    )

    pause_edit_plan = dict(getattr(synth_pair, "pause_edit_plan", {}) or {})
    generated_text = str(getattr(synth_pair, "generated_text", normed_text))
    reverse_st = reverse_operation(profile.operation)
    profile_id = f"profile_{profile.profile_id}"
    return [
        _row_for_direction(
            dataset_name=dataset_name,
            sample_id=work_item.sample_id,
            profile=profile,
            sample_rate=sample_rate,
            language=language,
            direction="original_to_generated",
            src_id="original",
            tgt_id=profile_id,
            src_audio=src_encoded,
            tgt_audio=tgt_encoded,
            src_duration=src_duration,
            tgt_duration=tgt_duration,
            src_label=normed_text,
            tgt_label=generated_text,
            src_tagged_text=normed_text,
            tgt_tagged_text=profile.operation,
            instruction_st=profile.operation,
            eval_result=eval_result,
            pause_edit_plan=pause_edit_plan,
        ),
        _row_for_direction(
            dataset_name=dataset_name,
            sample_id=work_item.sample_id,
            profile=profile,
            sample_rate=sample_rate,
            language=language,
            direction="generated_to_original",
            src_id=profile_id,
            tgt_id="original",
            src_audio=tgt_encoded,
            tgt_audio=src_encoded,
            src_duration=tgt_duration,
            tgt_duration=src_duration,
            src_label=generated_text,
            tgt_label=normed_text,
            src_tagged_text=profile.operation,
            tgt_tagged_text=reverse_st,
            instruction_st=reverse_st,
            eval_result=eval_result,
            pause_edit_plan=pause_edit_plan,
        ),
    ]


class PauseEditParquetRowProcessor(ParquetRowProcessor):
    def rows_from_payload(
        self,
        payload: Any,
        *,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        return pause_edit_rows(
            payload=payload,
            dataset_name=dataset_name,
        )
