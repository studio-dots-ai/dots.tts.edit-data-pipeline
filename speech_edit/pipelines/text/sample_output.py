"""Parquet-friendly output layout for tagged text edit samples."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from speech_edit.data.text_markup import ORIGINAL_WAVE_ID


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_bytes(path: str, payload: bytes) -> None:
    with open(path, "wb") as f:
        f.write(payload)


def pair_dir_name(source_id: str, target_id: str) -> str:
    return f"{source_id}_to_{target_id}"


def _replace_relative_symlink(target_path: str, link_path: str) -> None:
    if os.path.lexists(link_path):
        os.unlink(link_path)
    rel_target = os.path.relpath(target_path, os.path.dirname(link_path))
    os.symlink(rel_target, link_path)


def _audio_path_for_id(sample_dir: str, generation_id: str) -> str:
    if generation_id == ORIGINAL_WAVE_ID:
        return os.path.join(sample_dir, "audios", "original.wav")
    return os.path.join(sample_dir, "audios", f"generated_{generation_id}.wav")


def build_segments_metadata(
    *,
    source_label: str,
    target_label: str,
    source_duration: float,
    target_duration: float,
    source_edit_spans: List[List[float]],
    target_edit_spans: List[List[float]],
) -> Dict[str, Dict[str, Any]]:
    return {
        "src": {
            "chunk_id": "src",
            "start": 0.0,
            "end": source_duration,
            "label": source_label,
            "edit_time_spans": source_edit_spans,
        },
        "tgt": {
            "chunk_id": "tgt",
            "start": 0.0,
            "end": target_duration,
            "label": target_label,
            "edit_time_spans": target_edit_spans,
        },
    }


def build_sample_metadata(
    *,
    sample_id: str,
    clip_id: str,
    source_text: str,
    original_duration: float,
    sample_rate: int,
    generations: Dict[str, Dict[str, Any]],
    pair_results: List[Dict[str, Any]],
    pipeline_version: str,
    pipeline_type: str,
    config: Dict[str, Any],
    source_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "pipeline_version": pipeline_version,
        "pipeline_type": pipeline_type,
        "sample_id": sample_id,
        "clip_id": clip_id,
        "timestamp": datetime.now().isoformat(),
        "source_text": source_text,
        "original_audio_file": "audios/original.wav",
        "sample_rate": sample_rate,
        "original_duration": original_duration,
        "audios": {
            "original": "audios/original.wav",
            "generated": {
                generation_id: f"audios/generated_{generation_id}.wav"
                for generation_id in generations
            },
        },
        "generations": {
            generation_id: {
                **generation_meta,
                "audio_file": f"audios/generated_{generation_id}.wav",
            }
            for generation_id, generation_meta in generations.items()
        },
        "pair_count": len(pair_results),
        "pairs_with_eval": sum(1 for pair in pair_results if pair.get("eval_results")),
        "edit_pairs": [
            {
                key: value
                for key, value in pair.items()
                if key not in {"source_audio_bytes", "target_audio_bytes"}
            }
            for pair in pair_results
        ],
        "config": config,
        "source_metadata": source_metadata or {},
    }


def save_sample_outputs(
    sample_dir: str,
    *,
    original_audio_bytes: bytes,
    generated_audio_bytes: Dict[str, bytes],
    pair_results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(sample_dir, exist_ok=True)
    audios_dir = os.path.join(sample_dir, "audios")
    os.makedirs(audios_dir, exist_ok=True)
    pairs_root = os.path.join(sample_dir, "edit_pairs")
    os.makedirs(pairs_root, exist_ok=True)

    save_bytes(os.path.join(audios_dir, "original.wav"), original_audio_bytes)
    for generation_id, audio_bytes in generated_audio_bytes.items():
        save_bytes(os.path.join(audios_dir, f"generated_{generation_id}.wav"), audio_bytes)

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

    save_json(os.path.join(sample_dir, "metadata.json"), metadata)
