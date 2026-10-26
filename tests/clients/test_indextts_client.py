import base64
import json

import numpy as np

from speech_edit.clients.services import IndexTTSClient


def _encode_tokens(values: list[int]) -> dict:
    arr = np.asarray(values, dtype=np.int64)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


class _DummyResponse:
    def __init__(self, status_code=200, json_payload=None, content=b"", headers=None):
        self.status_code = status_code
        self._json_payload = json_payload or {}
        self.content = content
        self.headers = headers or {}
        self.text = json.dumps(self._json_payload)

    def json(self):
        return self._json_payload


def test_synthesize_skip_vocode_returns_token_dict(monkeypatch):
    captured = {}

    def _fake_post(url, json, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse(
            json_payload={
                "tokens": _encode_tokens([1, 2, 3]),
                "token_rate": 50.0,
            }
        )

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _fake_post)

    client = IndexTTSClient(base_url="http://localhost:8002")
    result = client.synthesize(
        text="hello",
        spk_audio_bytes=b"RIFFfake",
        skip_vocode=True,
        timeout=12,
    )

    assert captured["json"]["skip_vocode"] is True
    assert captured["timeout"] == 12
    assert result["tokens"].tolist() == [1, 2, 3]
    assert result["token_rate"] == 50.0


def test_synthesize_audio_prompt_keeps_alpha_only_payload(monkeypatch):
    captured = {}

    def _fake_post(url, json, timeout=None):
        captured["json"] = json
        return _DummyResponse(
            content=b"RIFFfakewav", headers={"Content-Type": "audio/wav"}
        )

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _fake_post)

    client = IndexTTSClient(base_url="http://localhost:8002")
    result = client.synthesize(
        text="hello",
        spk_audio_bytes=b"RIFFspk",
        emo_audio_bytes=b"RIFFemo",
        emo_alpha=0.9,
        output_format="wav",
    )

    assert result == b"RIFFfakewav"
    assert captured["json"]["emo_alpha"] == 0.9
    assert "emo_audio_prompt" in captured["json"]


def test_synthesize_uses_service_default_alpha_when_unspecified(monkeypatch):
    captured = {}

    def _fake_post(url, json, timeout=None):
        captured["json"] = json
        return _DummyResponse(
            content=b"RIFFdefaultwav", headers={"Content-Type": "audio/wav"}
        )

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _fake_post)

    client = IndexTTSClient(base_url="http://localhost:8002")
    result = client.synthesize(
        text="hello",
        spk_audio_bytes=b"RIFFspk",
        emo_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        output_format="wav",
    )

    assert result == b"RIFFdefaultwav"
    assert captured["json"]["emo_alpha"] == 0.8


def test_synthesize_from_tokens_forwards_timeout(monkeypatch):
    captured = {}

    def _fake_post(url, json, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse(
            content=b"RIFFtokenwav", headers={"Content-Type": "audio/wav"}
        )

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _fake_post)

    client = IndexTTSClient(base_url="http://localhost:8002")
    codes = np.asarray([1, 2, 3, 4], dtype=np.int64)
    result = client.synthesize_from_tokens(
        codes=codes,
        text="hello",
        spk_audio_bytes=b"RIFFspk",
        emo_alpha=0.8,
        output_format="wav",
        timeout=23,
    )

    assert result == b"RIFFtokenwav"
    assert captured["timeout"] == 23
    assert captured["json"]["emo_alpha"] == 0.8
