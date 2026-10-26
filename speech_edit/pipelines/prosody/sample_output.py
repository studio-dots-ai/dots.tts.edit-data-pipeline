"""Sample output layout helpers for the prosody pipeline."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .editor import EditResult
from speech_edit.evaluation.audio import PairEvalResult
from .input_generator import PipelineInput
from .tag_utils import ORIGINAL_WAVE_ID, WaveEditPair


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_bytes(path: str, payload: bytes) -> None:
    with open(path, "wb") as f:
        f.write(payload)


def pair_dir_name(pair: WaveEditPair) -> str:
    return f"{pair.source_id}_to_{pair.target_id}"


def serialize_edit_result(edit_result: EditResult) -> Dict[str, Any]:
    return {
        "edits_applied": edit_result.edits_applied,
        "edit_time_sec": edit_result.edit_time_sec,
        "boundary_times": edit_result.boundary_times,
        "smoothing_applied": edit_result.smoothing_applied,
        "alignment": edit_result.alignment_metadata,
    }


def serialize_pair_eval(eval_result: Optional[PairEvalResult]) -> Dict[str, Any]:
    if eval_result is None:
        return {}
    return eval_result.to_dict()


def build_sample_metadata(
    sample_id: str,
    pipeline_input: PipelineInput,
    wave_results: Dict[str, Dict[str, Any]],
    pair_results: List[Dict[str, Any]],
    *,
    pipeline_version: str,
    pipeline_type: str,
) -> Dict[str, Any]:
    stage1_info = {
        **pipeline_input.wave_group.metadata,
        **pipeline_input.metadata,
    }
    stage0_info = dict(pipeline_input.metadata.get("stage0_info", {}))
    return {
        "version": 1,
        "pipeline_version": pipeline_version,
        "pipeline_type": pipeline_type,
        "sample_id": sample_id,
        "timestamp": datetime.now().isoformat(),
        "clip_id": pipeline_input.clip_id,
        "input_id": pipeline_input.metadata.get("input_id"),
        "language": pipeline_input.metadata.get("language"),
        "transcript": pipeline_input.transcript,
        "plain_text": pipeline_input.wave_group.plain_text,
        "normed_text": pipeline_input.normed_text,
        "original_audio_file": "audios/original.wav",
        "sample_rate": pipeline_input.sample_rate,
        "original_duration": pipeline_input.duration,
        "tagged_texts": pipeline_input.wave_group.tagged_texts,
        "wave_specs": stage1_info.get("wave_specs", {}),
        "parameter_values": stage1_info.get("parameter_values", {}),
        "parameter_policy": stage1_info.get("parameter_policy", {}),
        "audios": {
            "original": "audios/original.wav",
            "generated": {
                wave_id: f"audios/generated_{wave_id}.wav"
                for wave_id in wave_results
            },
        },
        "pair_count": len(pair_results),
        "pairs_with_eval": sum(1 for pair in pair_results if pair["eval_results"]),
        "boundaries": [
            {
                "char_idx": boundary.char_idx,
                "has_punctuation": boundary.has_punctuation,
            }
            for boundary in pipeline_input.wave_group.boundaries
        ],
        "stage0_info": stage0_info,
        "stage1_info": stage1_info,
        "waves": {
            wave_id: {
                **wave_meta,
                "audio_file": f"audios/generated_{wave_id}.wav",
            }
            for wave_id, wave_meta in wave_results.items()
        },
        "edit_pairs": [
            {
                key: value
                for key, value in pair_result.items()
                if key not in {"source_audio_bytes", "target_audio_bytes"}
            }
            for pair_result in pair_results
        ],
    }


def _replace_relative_symlink(target_path: str, link_path: str) -> None:
    if os.path.lexists(link_path):
        os.unlink(link_path)
    rel_target = os.path.relpath(target_path, os.path.dirname(link_path))
    os.symlink(rel_target, link_path)


def _audio_path_for_id(sample_dir: str, wave_id: str) -> str:
    if wave_id == ORIGINAL_WAVE_ID:
        return os.path.join(sample_dir, "audios", "original.wav")
    return os.path.join(sample_dir, "audios", f"generated_{wave_id}.wav")


def save_sample_outputs(
    sample_dir: str,
    *,
    original_audio_bytes: bytes,
    wave_audio_bytes: Dict[str, bytes],
    pair_results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    """Write interactive sample artifacts and record their location in metadata."""
    os.makedirs(sample_dir, exist_ok=True)
    audios_dir = os.path.join(sample_dir, "audios")
    os.makedirs(audios_dir, exist_ok=True)
    pairs_root = os.path.join(sample_dir, "edit_pairs")
    os.makedirs(pairs_root, exist_ok=True)

    save_bytes(os.path.join(audios_dir, "original.wav"), original_audio_bytes)
    for wave_id, audio_bytes in wave_audio_bytes.items():
        save_bytes(os.path.join(audios_dir, f"generated_{wave_id}.wav"), audio_bytes)

    for pair_meta in pair_results:
        pair_dir = os.path.join(pairs_root, pair_meta["pair_dir_name"])
        os.makedirs(pair_dir, exist_ok=True)
        _replace_relative_symlink(
            _audio_path_for_id(sample_dir, pair_meta["source_id"]),
            os.path.join(pair_dir, "source.wav"),
        )
        _replace_relative_symlink(
            _audio_path_for_id(sample_dir, pair_meta["target_id"]),
            os.path.join(pair_dir, "target.wav"),
        )
        pair_payload = {
            key: value
            for key, value in pair_meta.items()
            if key not in {"source_audio_bytes", "target_audio_bytes"}
        }
        pair_payload["source_audio_file"] = "source.wav"
        pair_payload["target_audio_file"] = "target.wav"
        save_json(os.path.join(pair_dir, "metadata.json"), pair_payload)

    metadata["sample_dir"] = sample_dir
    save_json(os.path.join(sample_dir, "metadata.json"), metadata)
