import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from speech_edit.data.parquet.reader import (
    read_segments_from_parquet_payload,
)


def _encoded_wav(duration_sec: float, sample_rate: int = 1000) -> str:
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.zeros(int(duration_sec * sample_rate), dtype=np.float32),
        sample_rate,
        format="WAV",
    )
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_source_segments_accepts_dict_segments_schema(tmp_path: Path):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(2400, dtype=np.float32), 24000, format="WAV")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    parquet_path = tmp_path / "prosody.parquet"
    pd.DataFrame(
        [
            {
                "from_dataset": "unit",
                "audio_id": "sample--original_to_1",
                "audios": json.dumps({"src": encoded, "tgt": encoded}),
                "segments": json.dumps(
                    {
                        "src": {
                            "chunk_id": "src",
                            "transcript": "hello world",
                            "label": "hello world",
                            "start": 0.0,
                            "end": 0.1,
                        },
                        "tgt": {
                            "chunk_id": "tgt",
                            "transcript": "hello world",
                            "label": "hello world",
                            "start": 0.0,
                            "end": 0.1,
                        },
                    }
                ),
            }
        ]
    ).to_parquet(parquet_path, index=False)

    segments = read_segments_from_parquet_payload(
        {"parquet_path": str(parquet_path), "row_start": 0, "row_stop": 1},
        {"min_duration": 0.0, "max_duration": 1.0},
    )

    assert len(segments) == 1
    assert segments[0].metadata["chunk_id"] == "src"
    assert segments[0].text == "hello world"


def test_duration_bucket_policy_filters_and_labels_buckets(tmp_path: Path):
    parquet_path = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "clip_id": "short",
                "text": "short sample",
                "duration": 4.0,
                "audio": _encoded_wav(4.0),
            },
            {
                "clip_id": "mid",
                "text": "medium sample",
                "duration": 12.0,
                "audio": _encoded_wav(12.0),
            },
            {
                "clip_id": "long",
                "text": "long sample",
                "duration": 30.0,
                "audio": _encoded_wav(30.0),
            },
        ]
    ).to_parquet(parquet_path, index=False)

    cfg = {
        "seed": 123,
        "min_duration": 3.0,
        "max_duration": 30.0,
        "duration_bucket_policy": {
            "buckets": {
                "short": {"min": 3.0, "max": 10.0, "weight": 0.0},
                "mid": {"min": 10.0, "max": 20.0, "weight": 1.0},
                "long": {"min": 20.0, "max": 30.0, "weight": 1.0},
            }
        },
    }

    first = read_segments_from_parquet_payload(
        {
            "parquet_path": str(parquet_path),
            "source_file_index": 2,
            "row_start": 0,
            "row_stop": 3,
        },
        cfg,
    )
    second = read_segments_from_parquet_payload(
        {
            "parquet_path": str(parquet_path),
            "source_file_index": 2,
            "row_start": 0,
            "row_stop": 3,
        },
        cfg,
    )

    assert [segment.clip_id for segment in first] == ["mid", "long"]
    assert [segment.clip_id for segment in second] == ["mid", "long"]
    assert [segment.metadata["duration_bucket"] for segment in first] == ["mid", "long"]


def test_minimal_documented_input_schemas(tmp_path):
    encoded = _encoded_wav(3.2, 24000)
    rows = [
        {"audio": base64.b64decode(encoded), "text": "Hello world.", "duration": 3.2},
        {"segments": json.dumps([{"chunk_id": "000000", "text": "Hello world.", "duration": 3.2}]),
         "audios": json.dumps({"000000": encoded})},
    ]
    for index, row in enumerate(rows):
        path = tmp_path / f"source-{index}.parquet"
        pd.DataFrame([row]).to_parquet(path, index=False)
        segments = read_segments_from_parquet_payload(
            {"parquet_path": str(path), "row_start": 0, "row_stop": 1},
            {"min_duration": 2.0, "max_duration": 30.0},
        )
        assert len(segments) == 1
        assert segments[0].text == "Hello world."
        assert segments[0].sample_rate == 24000
        assert segments[0].duration == 3.2
        assert segments[0].audio_bytes == base64.b64decode(encoded)
