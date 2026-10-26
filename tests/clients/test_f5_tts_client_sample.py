import numpy as np

from speech_edit.clients.services import F5TTSClient


class _Response:
    status_code = 200
    content = b"wav"

    def json(self):
        return {}


def test_f5_tts_client_sample_sends_preserve_timeline_options(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        captured["url"] = url
        captured["json"] = json
        return _Response()

    monkeypatch.setattr("speech_edit.clients.services.requests.post", fake_post)

    result = F5TTSClient("http://f5").sample(
        np.zeros(24000, dtype=np.float32),
        ["hello kind world"],
        100,
        edit_spans_sec=[[0.2, 0.5]],
        preserve_unmasked_waveform=True,
        preserve_spans_sec=[[0.2, 0.5]],
        input_sample_rate=24000,
        source_crossfade_sec=0.01,
        regen_crossfade_sec=0.02,
    )

    assert result == b"wav"
    assert captured["timeout"] is None
    assert captured["url"] == "http://f5/sample"
    payload = captured["json"]
    assert payload["edit_spans_sec"] == [[0.2, 0.5]]
    assert payload["preserve_unmasked_waveform"] is True
    assert payload["preserve_spans_sec"] == [[0.2, 0.5]]
    assert payload["input_sample_rate"] == 24000
    assert payload["source_crossfade_sec"] == 0.01
    assert payload["regen_crossfade_sec"] == 0.02


def test_f5_tts_client_smooth_boundaries_uses_preserve_mode(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        captured["url"] = url
        captured["json"] = json
        return _Response()

    monkeypatch.setattr("speech_edit.clients.services.requests.post", fake_post)

    result = F5TTSClient("http://f5").smooth_boundaries(
        waveform=np.zeros(24000, dtype=np.float32),
        sample_rate=24000,
        text="hello kind world",
        boundary_times=[0.5],
        margin_sec=0.1,
        steps=12,
        seed=7,
    )

    assert result == b"wav"
    assert captured["timeout"] is None
    assert captured["url"] == "http://f5/sample"
    payload = captured["json"]
    assert payload["edit_spans_sec"] == [[0.4, 0.6]]
    assert payload["preserve_unmasked_waveform"] is True
    assert payload["preserve_spans_sec"] == [[0.4, 0.6]]
    assert payload["input_sample_rate"] == 24000
    assert payload["source_crossfade_sec"] == 0.0
    assert payload["regen_crossfade_sec"] == 0.01
    assert payload["steps"] == 12
    assert payload["seed"] == 7


def test_f5_tts_client_smooth_boundaries_rejects_non_24k():
    client = F5TTSClient("http://f5")

    try:
        client.smooth_boundaries(
            waveform=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            text="hello",
            boundary_times=[0.5],
        )
    except ValueError as exc:
        assert "expects 24000 Hz" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-24k smoothing input")
