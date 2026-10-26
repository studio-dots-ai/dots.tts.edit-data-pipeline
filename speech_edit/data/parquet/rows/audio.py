"""Small helpers shared by pipeline-owned parquet row builders."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from pydub import AudioSegment


def audio_segment_from_bytes(audio_bytes: bytes) -> AudioSegment:
    return AudioSegment.from_file(io.BytesIO(audio_bytes))


def encode_audio_flac_base64(audio_bytes: bytes, sample_rate: int = 24000) -> tuple[str, float]:
    audio = audio_segment_from_bytes(audio_bytes)
    flac_io = io.BytesIO()
    audio.export(
        flac_io,
        format="flac",
        parameters=["-ar", str(sample_rate), "-ac", "1", "-sample_fmt", "s16"],
    )
    return base64.b64encode(flac_io.getvalue()).decode("utf-8"), len(audio) / 1000.0


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)
