import sys
import threading
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from backends.openai_compatible_backend import (  # noqa: E402
    OpenAICompatibleBackend,
    OpenAICompatibleBackendError,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.lock = threading.Lock()

    def post(self, url, *, json, timeout):
        with self.lock:
            self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self.handler(url, json, timeout)


def _chat_response(text, model="served-model", usage=None):
    return _FakeResponse(
        payload={
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": usage or {"prompt_tokens": 3, "completion_tokens": 5},
        }
    )


def test_openai_backend_reports_health_metadata():
    backend = OpenAICompatibleBackend(
        name="gemma4_mtp",
        base_url="http://localhost:18112/v1",
        model="gemma-4-31b-it-mtp-tp2-32k",
    )

    assert backend.name == "gemma4_mtp"
    assert backend.is_ready is True
    assert backend.available_models == ["gemma-4-31b-it-mtp-tp2-32k"]
    assert backend.chat_completions_url == "http://localhost:18112/v1/chat/completions"


def test_openai_backend_normalizes_root_base_url():
    backend = OpenAICompatibleBackend(
        name="gemma4_mtp",
        base_url="http://localhost:18112",
        model="gemma-4-31b-it-mtp-tp2-32k",
    )

    assert backend.chat_completions_url == "http://localhost:18112/v1/chat/completions"


def test_openai_generate_maps_llm_service_params_to_chat_completions_payload():
    backend = OpenAICompatibleBackend(
        name="gemma4_mtp",
        base_url="http://server/v1",
        model="default-model",
        timeout=12.5,
    )
    backend.session = _FakeSession(
        lambda _url, _payload, _timeout: _chat_response(
            "ok",
            model="override-model",
            usage={"prompt_tokens": 7, "completion_tokens": 2},
        )
    )

    response = backend.generate(
        "Return JSON only.",
        model="override-model",
        temperature=0.0,
        max_tokens=64,
        json_mode=True,
        stop_sequences=["END"],
    )

    assert response.text == "ok"
    assert response.model == "override-model"
    assert response.usage == {"prompt_tokens": 7, "completion_tokens": 2}
    assert backend.session.calls == [
        {
            "url": "http://server/v1/chat/completions",
            "json": {
                "model": "override-model",
                "messages": [{"role": "user", "content": "Return JSON only."}],
                "temperature": 0.0,
                "max_tokens": 64,
                "stop": ["END"],
                "response_format": {"type": "json_object"},
            },
            "timeout": 12.5,
        }
    ]


def test_openai_generate_maps_audio_to_openai_multimodal_content():
    backend = OpenAICompatibleBackend(
        name="local",
        base_url="http://server/v1",
        model="qwen3-omni",
        enable_thinking=False,
    )
    backend.session = _FakeSession(
        lambda _url, _payload, _timeout: _chat_response("happy")
    )

    backend.generate(
        "Classify emotion.",
        audio_base64="YXVkaW8=",
        audio_mime_type="audio/wav",
        temperature=0.0,
    )

    payload = backend.session.calls[0]["json"]
    assert payload["messages"][0]["content"] == [
        {
            "type": "audio_url",
            "audio_url": {"url": "data:audio/wav;base64,YXVkaW8="},
        },
        {"type": "text", "text": "Classify emotion."},
    ]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_generate_raises_on_http_error():
    backend = OpenAICompatibleBackend(
        name="gemma4_mtp",
        base_url="http://server/v1",
        model="default-model",
    )
    backend.session = _FakeSession(
        lambda _url, _payload, _timeout: _FakeResponse(
            status_code=500,
            text="server exploded",
        )
    )

    with pytest.raises(OpenAICompatibleBackendError, match="server exploded"):
        backend.generate("hello")


def test_openai_generate_raises_on_unexpected_response_schema():
    backend = OpenAICompatibleBackend(
        name="gemma4_mtp",
        base_url="http://server/v1",
        model="default-model",
    )
    backend.session = _FakeSession(
        lambda _url, _payload, _timeout: _FakeResponse(payload={"choices": []})
    )

    with pytest.raises(OpenAICompatibleBackendError, match="Unexpected"):
        backend.generate("hello")


def test_openai_generate_batch_preserves_prompt_order():
    backend = OpenAICompatibleBackend(
        name="gemma4_mtp",
        base_url="http://server/v1",
        model="default-model",
        max_workers=4,
    )

    def _handler(_url, payload, _timeout):
        prompt = payload["messages"][0]["content"]
        return _chat_response(f"response:{prompt}", model=payload["model"])

    backend.session = _FakeSession(_handler)

    responses = backend.generate_batch(
        ["first", "second", "third"],
        temperature=0.1,
        max_tokens=16,
    )

    assert [r.text for r in responses] == [
        "response:first",
        "response:second",
        "response:third",
    ]
    assert [r.model for r in responses] == ["default-model"] * 3
    assert len(backend.session.calls) == 3
