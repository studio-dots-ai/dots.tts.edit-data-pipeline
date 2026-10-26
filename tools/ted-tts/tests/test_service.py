"""HTTP contracts for retained TED-TTS synthesis modes."""
import base64
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import api_service as api


@pytest.fixture
def service(tmp_path, monkeypatch):
    calls = []
    audio = b"RIFF-test-wav"
    source = tmp_path / "source.wav"
    source.write_bytes(audio)
    def infer(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(audio)
    monkeypatch.setattr(api, "TTS_MODEL", SimpleNamespace(infer=infer))
    monkeypatch.setattr(api, "TEMP_DIR", str(tmp_path))
    return TestClient(api.app), str(source), calls, audio


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_modes_forward_requests_and_return_audio(service, mode):
    client, source, calls, audio = service
    request = dict(text="Hello", spk_audio_prompt=source, mode=mode, output_format="json")
    extras = {
        0: {}, 1: {"emo_audio_prompt": source, "emo_alpha": 0.75},
        2: {"emo_vector": [0.1] * 8}, 3: {"emo_text": "happy"},
    }[mode]
    response = client.post("/synthesize", json={**request, **extras})
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["audio"]) == audio
    assert response.json()["sample_rate"] == 22050
    actual = calls[0]
    assert actual["text"] == "Hello"
    assert actual["spk_audio_prompt"] == source
    for key, expected in dict(top_p=0.8, top_k=30, temperature=0.8, repetition_penalty=10.0,
                              length_penalty=0.0, num_beams=3, max_mel_tokens=850,
                              max_text_tokens_per_sentence=150, method="hmm", use_random=False).items():
        assert actual[key] == expected
    if mode == 1:
        assert actual["emo_audio_prompt"] == source
        assert actual["emo_alpha"] == 0.75
    elif mode == 2:
        assert actual["emo_vector"] == [[0.1] * 8]
    elif mode == 3:
        assert actual["use_emo_text"] is True
        assert actual["emo_text"] == "happy"
    assert "target_duration_tokens" not in actual
    assert not Path(actual["output_path"]).exists()


def test_tagged_synthesis_uses_adapter_and_forwards_sampling_controls(service):
    client, source, calls, audio = service
    tagged = '<emo type="happy" level="1">Hello</emo><emo desc="angry and tense" type_ref="angry">there</emo>'
    adapted = client.post("/adapt_tagged", json={"tagged_text": tagged})
    assert adapted.status_code == 200
    request = dict(tagged_text=tagged, spk_audio_prompt=source, top_p=0.7, top_k=17,
                   temperature=0.6, repetition_penalty=8.0, num_beams=2,
                   max_mel_tokens=700, method="max_head")
    response = client.post("/synthesize_tagged", json=request)
    assert response.status_code == 200, response.text
    assert response.content == audio
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-sample-rate"] == "22050"
    actual = calls[0]
    assert actual["text"] == adapted.json()["text"]
    assert actual["emo_text"] == adapted.json()["emo_text"]
    assert actual["use_emo_text"] is True
    for key in ("top_p", "top_k", "temperature", "repetition_penalty", "num_beams", "max_mel_tokens", "method"):
        assert actual[key] == request[key]


@pytest.mark.parametrize("mode", [4, 5])
def test_duration_modes_remain_disabled(service, mode):
    client, source, calls, _ = service
    response = client.post("/synthesize", json=dict(
        text="Hello", spk_audio_prompt=source, mode=mode, emo_text="happy", target_seconds=[1.0],
    ))
    assert response.status_code == 501
    assert calls == []


def test_tagged_duration_remains_disabled(service):
    client, source, calls, _ = service
    response = client.post("/synthesize_tagged", json=dict(
        tagged_text='<emo desc="happy" dur="1.0">Hello</emo>', spk_audio_prompt=source,
    ))
    assert response.status_code == 501
    assert calls == []


def test_health_info_and_request_validation(service):
    client, source, calls, _ = service
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/info").json()["emotion_vector_order"] == [
        "happy", "angry", "sad", "fear", "hate", "low", "surprise", "neutral",
    ]
    response = client.post("/synthesize", json=dict(text="Hello", spk_audio_prompt=source, mode=2, emo_vector=[1.0]))
    assert response.status_code == 422
    assert calls == []
