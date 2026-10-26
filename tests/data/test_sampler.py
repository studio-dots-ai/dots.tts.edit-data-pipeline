"""Speaker prompt pool sampling with real nested and flat parquet inputs."""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from speech_edit.data.sampler import ParquetListSampler


def _dataset_list_path(dataset_dir: Path, name: str = "dataset.list") -> Path:
    return _write_list(
        dataset_dir / name,
        [str(path) for path in sorted(dataset_dir.glob("*.parquet"))],
    )


# ---------------------------------------------------------------------------
# Fixture helpers — build tiny parquet files with real WAV audio
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_sec: float, sr: int = 16000) -> bytes:
    """Generate a short sine-wave WAV as bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _make_parquet(path: Path, rows: list[dict]) -> None:
    """Write a single parquet file from a list of row dicts.

    Each row dict should contain:
      audio_id, chunk_id, text, duration_sec, from_dataset (optional)
    """
    records = []
    for row in rows:
        chunk_id = row.get("chunk_id", "000000")
        dur = row["duration_sec"]
        wav = _make_wav_bytes(dur)

        seg = {
            "chunk_id": chunk_id,
            "duration": dur,
            "text": row.get("text", "hello world"),
        }
        if "label" in row:
            seg["label"] = row["label"]
        audios = {chunk_id: base64.b64encode(wav).decode()}

        records.append(
            {
                "audio_id": row.get("audio_id", f"utt_{len(records)}"),
                "from_dataset": row.get("from_dataset", "test_ds"),
                "segments": json.dumps([seg]),
                "audios": json.dumps(audios),
            }
        )

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


def _make_flat_parquet(path: Path, rows: list[dict]) -> None:
    """Write a flat-schema parquet file with top-level audio bytes."""
    records = []
    for idx, row in enumerate(rows):
        dur = row["duration_sec"]
        wav = _make_wav_bytes(dur)
        records.append(
            {
                "clip_id": row.get("clip_id", f"clip_{idx}"),
                "speaker_id": row.get("speaker_id", f"speaker_{idx}"),
                "text": row.get("text", f"flat text {idx}"),
                "duration": dur,
                "sample_rate": row.get("sample_rate", 16000),
                "source_dataset": row.get("source_dataset", "flat_ds"),
                "source_row_idx": row.get("source_row_idx", idx),
                "audio": wav,
                "score": row.get("score", float(idx) / 10.0),
            }
        )

    pd.DataFrame(records).to_parquet(path, index=False)


def _write_list(path: Path, entries: list[str]) -> Path:
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> Path:
    """Create a tiny dataset with 2 parquet files, 5 rows each (10 total).

    All segments have duration ~3s so they pass the default 2-15s filter.
    """
    dataset_dir = tmp_path / "sample_dataset"
    dataset_dir.mkdir()
    for file_idx in range(2):
        rows = [
            {
                "audio_id": f"f{file_idx}_r{i}",
                "chunk_id": f"{file_idx:03d}{i:03d}",
                "text": f"file {file_idx} row {i}",
                "duration_sec": 3.0,
            }
            for i in range(5)
        ]
        _make_parquet(dataset_dir / f"part_{file_idx}.parquet", rows)
    return dataset_dir


@pytest.fixture()
def varied_duration_dataset(tmp_path: Path) -> Path:
    """Dataset with mixed durations: 1s (too short), 3s, 5s, 20s (too long)."""
    dataset_dir = tmp_path / "varied_duration_dataset"
    dataset_dir.mkdir()
    rows = [
        {"audio_id": "short", "duration_sec": 1.0, "text": "short"},
        {"audio_id": "ok3", "duration_sec": 3.0, "text": "ok three"},
        {"audio_id": "ok5", "duration_sec": 5.0, "text": "ok five"},
        {"audio_id": "long", "duration_sec": 20.0, "text": "long"},
    ]
    _make_parquet(dataset_dir / "varied.parquet", rows)
    return dataset_dir


@pytest.fixture()
def flat_schema_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "flat_schema_dataset"
    dataset_dir.mkdir()
    rows = [
        {
            "clip_id": "flat_clip_0",
            "speaker_id": "speaker_a",
            "text": "flat zero",
            "duration_sec": 3.0,
            "score": 0.9,
        },
        {
            "clip_id": "flat_clip_1",
            "speaker_id": "speaker_b",
            "text": "flat one",
            "duration_sec": 4.0,
            "score": 0.8,
        },
        {
            "clip_id": "flat_clip_2",
            "speaker_id": "speaker_c",
            "text": "flat two",
            "duration_sec": 5.0,
            "score": 0.7,
        },
    ]
    _make_flat_parquet(dataset_dir / "flat.parquet", rows)
    return dataset_dir


# ---------------------------------------------------------------------------
# Tests: basic iteration
# ---------------------------------------------------------------------------


def _sampler(root, seed=42, min_duration=2.0, max_duration=15.0):
    return ParquetListSampler(
        parquet_sources=[_dataset_list_path(root)],
        min_single_duration=min_duration,
        max_single_duration=max_duration,
        seed=seed,
    )


def test_nested_segment_fields_and_provenance(sample_dataset):
    sampler = _sampler(sample_dataset)
    segments = [sampler.sample() for _ in range(10)]
    assert len({s.clip_id for s in segments}) == 10
    for segment in segments:
        assert segment.text
        assert segment.audio_bytes
        assert segment.sample_rate == 16000
        assert segment.duration == pytest.approx(3.0)
        assert segment.metadata['source_file_index'] in (0, 1)
        assert Path(segment.metadata['source_parquet_path']).is_file()


def test_flat_audio_and_metadata(flat_schema_dataset):
    sampler = _sampler(flat_schema_dataset)
    for _ in range(3):
        segment = sampler.sample()
        assert segment.clip_id.startswith('flat_clip_')
        assert segment.text.startswith('flat')
        assert segment.metadata['speaker_id'].startswith('speaker_')
        assert 'audio' not in segment.metadata
        assert sf.info(io.BytesIO(segment.audio_bytes)).samplerate == 16000
        json.dumps(segment.metadata)


def test_duration_filtering_and_cycling(varied_duration_dataset):
    sampler = _sampler(varied_duration_dataset)
    segments = [sampler.sample() for _ in range(6)]
    assert all(2 <= s.duration <= 15 for s in segments)
    assert {s.duration for s in segments} == {3.0, 5.0}
    assert sampler._epoch == 2


def test_seeded_order_across_epochs(sample_dataset):
    first, second, different = (_sampler(sample_dataset, seed) for seed in (42, 42, 43))
    a = [first.sample().clip_id for _ in range(20)]
    b = [second.sample().clip_id for _ in range(20)]
    c = [different.sample().clip_id for _ in range(20)]
    assert a == b
    assert a != c
    assert set(a[:10]) == set(a[10:])
    assert a[:10] != a[10:]


def test_pool_without_eligible_audio_fails(varied_duration_dataset):
    sampler = _sampler(varied_duration_dataset, min_duration=40, max_duration=50)
    with pytest.raises(RuntimeError, match='no segments'):
        sampler.sample()


def test_empty_pool_fails(tmp_path):
    with pytest.raises(ValueError, match="Parquet list file is empty"):
        _sampler(tmp_path)
