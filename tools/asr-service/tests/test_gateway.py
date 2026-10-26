"""Unit tests for the multi-backend ASR gateway."""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException


ASR_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASR_SERVICE_ROOT))

import asr_service


@dataclass
class _Result:
    text: str
    language: str | None = None


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.backend_name = name
        self.inference_name = "fake"
        self.model_path = f"http://{name}"
        self.is_loaded = True
        self.batches: list[list[str | None]] = []

    def transcribe_batch(self, audios, languages):
        self.batches.append(list(languages))
        return [_Result(text=f"{self.backend_name}:{index}", language=language) for index, language in enumerate(languages)]


def _encoded_wav() -> dict:
    with io.BytesIO() as buffer:
        sf.write(buffer, np.zeros(160, dtype=np.float32), 16000, format="WAV")
        return {
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "format": "wav",
        }


async def _with_fake_gateway(coro):
    qwen = _FakeBackend("qwen3-asr")
    asr_service.DEFAULT_BACKEND = "qwen3-asr"
    asr_service.ASR_BACKENDS = {
        "qwen3-asr": qwen,
    }
    asr_service.REQUEST_BUFFERS = {
        name: asr_service.RequestBuffer(backend, max_batch_size=8, max_wait_ms=10)
        for name, backend in asr_service.ASR_BACKENDS.items()
    }
    for buffer in asr_service.REQUEST_BUFFERS.values():
        buffer.start()
    try:
        return await coro(qwen)
    finally:
        for buffer in asr_service.REQUEST_BUFFERS.values():
            await buffer.stop()
        asr_service.ASR_BACKENDS = {}
        asr_service.REQUEST_BUFFERS = {}


def test_default_backend_routes_to_qwen3():
    async def run(qwen):
        response = await asr_service.transcribe(
            asr_service.TranscribeRequest(audio=_encoded_wav(), language="English")
        )
        assert response.text == "qwen3-asr:0"
        assert qwen.batches == [["English"]]

    asyncio.run(_with_fake_gateway(run))


def test_batch_preserves_order_and_language():
    async def run(qwen):
        response = await asr_service.transcribe_batch(
            asr_service.TranscribeBatchRequest(items=[
                asr_service.TranscribeRequest(audio=_encoded_wav(), language=language)
                for language in ["English", None, "Chinese"]
            ])
        )
        assert [item.text for item in response.results] == ["qwen3-asr:0", "qwen3-asr:1", "qwen3-asr:2"]
        assert qwen.batches == [["English", None, "Chinese"]]
    asyncio.run(_with_fake_gateway(run))


def test_unknown_backend_returns_400():
    async def run(qwen):
        with pytest.raises(HTTPException) as exc_info:
            await asr_service.transcribe(
                asr_service.TranscribeRequest(
                    audio=_encoded_wav(),
                    backend="missing-backend",
                )
            )
        assert exc_info.value.status_code == 400

    asyncio.run(_with_fake_gateway(run))
