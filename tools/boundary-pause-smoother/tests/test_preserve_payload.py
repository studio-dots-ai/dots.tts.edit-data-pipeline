import base64
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_smoother_module():
    path = Path(__file__).resolve().parents[1] / "smoother.py"
    spec = importlib.util.spec_from_file_location("boundary_pause_smoother", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode_array(arr: np.ndarray) -> dict:
    return {
        "data": base64.b64encode(arr.astype(np.float32).tobytes()).decode("utf-8"),
        "shape": list(arr.shape),
        "dtype": "float32",
    }


def test_regenerate_regions_delegates_preserve_splice_to_f5(monkeypatch):
    smoother = _load_smoother_module()
    audio = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    captured = {}

    def fake_resample(arr, orig_sr, target_sr):
        assert orig_sr == 16000
        assert target_sr == smoother.F5_SAMPLE_RATE
        return np.linspace(-0.1, 0.1, 2400, dtype=np.float32)

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "sample_rate": 16000,
                "tensor": _encode_array(audio),
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(smoother, "_resample", fake_resample)
    monkeypatch.setattr(smoother.requests, "post", fake_post)

    result = smoother._regenerate_regions(
        audio,
        16000,
        "hello world",
        [[0.2, 0.4]],
        f5tts_url="http://f5",
        crossfade_sec=0.02,
    )

    np.testing.assert_allclose(result.spliced_audio, audio)
    np.testing.assert_allclose(result.f5_output_audio, audio)
    assert result.f5_output_sample_rate == 16000
    payload = captured["payload"]
    assert captured["url"] == "http://f5/sample"
    assert payload["duration"] == 11
    assert payload["input_sample_rate"] == smoother.F5_SAMPLE_RATE
    assert payload["preserve_unmasked_waveform"] is True
    assert payload["preserve_waveform_sample_rate"] == 16000
    assert payload["preserve_spans_sec"] == [[0.2, 0.4]]
    assert payload["source_crossfade_sec"] == 0.0
    assert payload["regen_crossfade_sec"] == 0.02


def test_regenerate_regions_rejects_overlong_f5_input(monkeypatch):
    smoother = _load_smoother_module()
    sr = smoother.F5_SAMPLE_RATE
    duration_sec = smoother.MAX_F5_EDIT_SECONDS + 0.1
    audio = np.zeros(int(duration_sec * sr), dtype=np.float32)

    def fail_post(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("F5 request should not be sent for overlong audio")

    monkeypatch.setattr(smoother.requests, "post", fail_post)

    with pytest.raises(ValueError, match="single-call F5 edit limit"):
        smoother._regenerate_regions(
            audio,
            sr,
            "hello world",
            [[0.2, 0.4]],
            f5tts_url="http://f5",
        )


def test_regenerate_regions_rejects_near_limit_frame_expansion(monkeypatch):
    smoother = _load_smoother_module()
    sr = smoother.F5_SAMPLE_RATE
    max_model_frames = int(
        smoother.MAX_F5_EDIT_SECONDS * sr / smoother.F5_HOP_LENGTH
    )
    # Raw audio is still under 30s, but smoother sends num_frames + 1 to F5.
    audio = np.zeros(
        (max_model_frames - 1) * smoother.F5_HOP_LENGTH,
        dtype=np.float32,
    )

    def fail_post(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("F5 request should not be sent for overlong frames")

    monkeypatch.setattr(smoother.requests, "post", fail_post)

    with pytest.raises(ValueError, match="requested_duration"):
        smoother._regenerate_regions(
            audio,
            sr,
            "hello world",
            [[0.2, 0.4]],
            f5tts_url="http://f5",
        )
