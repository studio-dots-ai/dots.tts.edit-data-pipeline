from __future__ import annotations

from speech_edit.clients.services import LLMClient


class _Response:
    status_code = 200

    def json(self):
        return {"text": "ok", "results": [{"text": "a"}, {"text": "b"}]}


def test_llm_client_omits_backend_when_not_explicit(monkeypatch):
    posted = []

    def _post(url, *, json, timeout):
        posted.append({"url": url, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _post)

    client = LLMClient("http://llm")
    assert client.generate("hello", temperature=0.0) == "ok"
    assert "backend" not in posted[0]["json"]


def test_llm_client_preserves_explicit_backend(monkeypatch):
    posted = []

    def _post(url, *, json, timeout):
        posted.append({"url": url, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _post)

    client = LLMClient("http://llm")
    assert client.generate("hello", backend="gemini", model="2_5pro") == "ok"
    assert posted[0]["json"]["backend"] == "gemini"
    assert posted[0]["json"]["model"] == "2_5pro"
