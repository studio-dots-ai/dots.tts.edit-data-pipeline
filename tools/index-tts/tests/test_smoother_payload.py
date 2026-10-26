import base64
import io

import numpy as np
import soundfile as sf

import api_service as synthesis_service


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


class _FakeTTSModel:
    def infer(self, **kwargs):
        text = kwargs["text"]
        length = max(200, len(text) * 120)
        audio = np.linspace(-0.1, 0.1, length, dtype=np.float32)
        return synthesis_service.SAMPLING_RATE, audio

    def _join_continuation_text(self, left: str, right: str) -> str:
        return f"{left} {right}".strip()


class _FakeResponse:
    def __init__(self, content: bytes):
        self.status_code = 200
        self.content = content
        self.text = "ok"

    def json(self):
        return {"answers": [{"start": 0.0, "end": 0.5}], "backend": "qwen3"}


def test_align_audio_text_forwards_explicit_qwen3_backend(monkeypatch, tmp_path):
    captured = {}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(b"")

    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(_wav_bytes(np.zeros(400, dtype=np.float32), synthesis_service.SAMPLING_RATE))
    monkeypatch.setattr("requests.post", _fake_post)

    result = synthesis_service._align_audio_text(
        str(wav_path),
        "hello world",
        "http://aligner.test",
        queries=["<q>hello</q> world"],
        aligner_backend="qwen3",
        aligner_language="eng",
    )

    assert result["backend"] == "qwen3"
    assert captured["url"] == "http://aligner.test/align"
    assert captured["json"]["backend"] == "qwen3"
    assert captured["json"]["language"] == "eng"
    assert captured["json"]["queries"] == ["<q>hello</q> world"]


def test_continuation_smoothing_uses_smooth_only_payload(monkeypatch):
    captured = {}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            _wav_bytes(np.zeros(800, dtype=np.float32), synthesis_service.SAMPLING_RATE)
        )

    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr(synthesis_service, "TTS_MODEL", _FakeTTSModel())

    request = synthesis_service.SynthesisRequest(
        text="<happy>hello</happy><sad>world</sad>",
        spk_audio_prompt={
            "data": base64.b64encode(
                _wav_bytes(
                    np.zeros(400, dtype=np.float32),
                    synthesis_service.SAMPLING_RATE,
                )
            ).decode("utf-8"),
            "format": "wav",
        },
        use_emotion_tags=True,
        generation_paradigm="continuation",
        output_format="wav",
        smoother_url="http://localhost:8018",
        aligner_backend="qwen3",
        aligner_language="eng",
        smooth_radius_sec=0.45,
        seed=123,
    )
    segments = synthesis_service.parse_emotion_tags(request.text)

    sr, wav_data = synthesis_service._smooth_continuation_boundaries(
        request,
        np.zeros(1200, dtype=np.int16),
        synthesis_service.SAMPLING_RATE,
        segments,
        [0.5],
    )

    assert sr == synthesis_service.SAMPLING_RATE
    assert len(wav_data) > 0
    assert captured["url"] == "http://localhost:8018/smooth"
    assert captured["timeout"] == 300
    assert captured["json"]["aligner_backend"] == "qwen3"
    assert captured["json"]["language"] == "eng"
    assert captured["json"]["f5_seed"] == 123
    point = captured["json"]["points"][0]
    assert point == {
        "mode": "smooth_only",
        "position_sec": 0.5,
        "regen_margin_sec": 0.45,
    }
