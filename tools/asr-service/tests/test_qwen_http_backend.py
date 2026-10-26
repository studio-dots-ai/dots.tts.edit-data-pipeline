"""Unit tests for the Qwen3-ASR HTTP backend adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ASR_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASR_SERVICE_ROOT))

from backends.qwen3_asr import Qwen3ASRHTTPBackend


class _FakeResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_qwen_backend_accepts_healthy_service(monkeypatch):
    def fake_get(url: str, timeout: float):
        assert url == "http://example.test:8033/health"
        return _FakeResponse({"status": "ok"})

    monkeypatch.setattr("backends.qwen3_asr.requests.get", fake_get)

    backend = Qwen3ASRHTTPBackend(service_url="http://example.test:8033")
    backend.load()

    assert backend.is_loaded is True
    assert backend.backend_name == "qwen3-asr"
    assert backend.inference_name == "vllm-http"


def test_qwen_backend_calls_chat_completions_and_parses_asr_text(monkeypatch):
    calls = []

    def fake_post(url: str, json: dict, timeout: float):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "language English<asr_text>Hello world",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("backends.qwen3_asr.requests.post", fake_post)

    backend = Qwen3ASRHTTPBackend(service_url="http://example.test:8033", timeout=42)
    backend._ready = True
    result = backend.transcribe_batch([(np.zeros(160, dtype=np.float32), 16000)], [None])[0]

    assert result.text == "Hello world"
    assert result.language == "English"
    assert calls[0]["url"] == "http://example.test:8033/v1/chat/completions"
    assert calls[0]["timeout"] == 42
    assert calls[0]["json"]["temperature"] == 0.0
    assert calls[0]["json"]["max_tokens"] == 512
    messages = calls[0]["json"]["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0]["type"] == "audio_url"
    assert content[0]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_qwen_backend_sends_forced_language_prefill_and_keeps_plain_text_response(monkeypatch):
    calls = []

    def fake_post(url: str, json: dict, timeout: float):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse({"choices": [{"message": {"content": "plain transcript"}}]})

    monkeypatch.setattr("backends.qwen3_asr.requests.post", fake_post)

    backend = Qwen3ASRHTTPBackend(service_url="http://example.test:8033")
    backend._ready = True
    result = backend.transcribe_batch([(np.zeros(160, dtype=np.float32), 16000)], ["Chinese"])[0]

    assert result.text == "plain transcript"
    assert result.language == "Chinese"
    assert calls[0]["json"]["temperature"] == 0.0
    assert calls[0]["json"]["max_tokens"] == 512
    messages = calls[0]["json"]["messages"]
    assert messages[0]["role"] == "user"
    assert messages[1] == {
        "role": "assistant",
        "content": "language Chinese<asr_text>",
    }


def test_qwen_backend_normalizes_forced_language_name(monkeypatch):
    calls = []

    def fake_post(url: str, json: dict, timeout: float):
        calls.append(json)
        return _FakeResponse({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr("backends.qwen3_asr.requests.post", fake_post)

    backend = Qwen3ASRHTTPBackend(service_url="http://example.test:8033")
    backend._ready = True
    result = backend.transcribe_batch([(np.zeros(160, dtype=np.float32), 16000)], ["eng"])[0]

    assert result.language == "English"
    assert calls[0]["messages"][1]["content"] == "language English<asr_text>"
