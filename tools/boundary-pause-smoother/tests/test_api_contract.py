from __future__ import annotations

import base64
import importlib.util
import io
import sys
import types
import asyncio
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
import soundfile as sf


def _load_api_service_module():
    module_path = Path(__file__).resolve().parents[1] / "api_service.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(
        "boundary_pause_smoother_api_service", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


def _make_insert_wav_bytes(num_samples: int = 800) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.ones(num_samples, dtype=np.float32) * 0.1, 16000, format="WAV")
    return buf.getvalue()


def _validate_request(service, payload):
    return service.SmoothRequest.model_validate(payload)


def _call_smooth(service, payload):
    request = _validate_request(service, payload)
    return asyncio.run(service.smooth(request))


def test_smooth_request_accepts_pause_interval_and_optional_position(monkeypatch):
    service = _load_api_service_module()
    captured = {}

    def _fake_smooth_audio(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            points_processed=1,
            details=[],
        )

    monkeypatch.setattr(service, "smooth_audio", _fake_smooth_audio)

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [
            {
                "mode": "adjust",
                "pause_interval_sec": [0.5, 0.7],
                "min_pause_sec": 0.0,
                "max_pause_sec": 0.3,
                "regen_margin_sec": 0.2,
            }
        ],
        "output_format": "numpy",
    }

    response = _call_smooth(service, payload)
    assert response.status == "success"
    assert len(captured["points"]) == 1
    assert captured["points"][0].mode == "adjust"
    assert captured["points"][0].position_sec is None
    assert captured["points"][0].pause_interval_sec == (0.5, 0.7)


def test_smooth_request_decodes_audio_insert_and_sets_effective_pause_bounds(monkeypatch):
    service = _load_api_service_module()
    captured = {}

    def _fake_smooth_audio(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            points_processed=1,
            details=[],
        )

    monkeypatch.setattr(service, "smooth_audio", _fake_smooth_audio)
    insert_bytes = _make_insert_wav_bytes(num_samples=800)

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [
            {
                "mode": "adjust",
                "pause_interval_sec": [0.5, 0.5],
                "min_pause_sec": 0.3,
                "max_pause_sec": 0.0,
                "regen_margin_sec": 0.2,
                "audio_insert": {
                    "data": base64.b64encode(insert_bytes).decode(),
                    "format": "wav",
                },
            }
        ],
        "output_format": "numpy",
    }

    response = _call_smooth(service, payload)
    assert response.status == "success"
    point = captured["points"][0]
    assert len(point.audio_insert) == 800
    assert point.min_pause_sec == 0.05
    assert point.max_pause_sec == 0.05


def test_smooth_request_accepts_audio_insert_intrusion_ratio(monkeypatch):
    service = _load_api_service_module()
    captured = {}

    def _fake_smooth_audio(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            points_processed=1,
            details=[],
        )

    monkeypatch.setattr(service, "smooth_audio", _fake_smooth_audio)
    insert_bytes = _make_insert_wav_bytes(num_samples=800)

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [
            {
                "mode": "adjust",
                "pause_interval_sec": [0.5, 0.5],
                "regen_margin_sec": 0.2,
                "audio_insert_intrusion_ratio": 0.15,
                "audio_insert": {
                    "data": base64.b64encode(insert_bytes).decode(),
                    "format": "wav",
                },
            }
        ],
        "output_format": "numpy",
    }

    response = _call_smooth(service, payload)
    assert response.status == "success"
    assert captured["points"][0].audio_insert_intrusion_ratio == pytest.approx(0.15)


def test_smooth_request_rejects_missing_position_and_interval():
    service = _load_api_service_module()

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [{"mode": "adjust", "max_pause_sec": 0.1}],
        "output_format": "numpy",
    }

    with pytest.raises(ValueError):
        _validate_request(service, payload)


def test_smooth_request_rejects_invalid_pause_bounds():
    service = _load_api_service_module()

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [
            {
                "mode": "adjust",
                "position_sec": 0.5,
                "min_pause_sec": 0.3,
                "max_pause_sec": 0.1,
            }
        ],
        "output_format": "numpy",
    }

    with pytest.raises(ValueError):
        _validate_request(service, payload)


def test_smooth_wav_response_exposes_resolved_debug_dir(monkeypatch):
    service = _load_api_service_module()

    def _fake_smooth_audio(**kwargs):
        return types.SimpleNamespace(
            audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            points_processed=1,
            details=[],
            debug_dir="/tmp/generated-debug-dir",
        )

    monkeypatch.setattr(service, "smooth_audio", _fake_smooth_audio)

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [{"position_sec": 0.5}],
        "output_format": "wav",
        "debug": True,
    }

    response = _call_smooth(service, payload)
    assert response.status_code == 200
    assert response.headers["X-Debug-Dir"] == "/tmp/generated-debug-dir"


def test_smooth_only_rejects_pause_adjustment_fields():
    service = _load_api_service_module()

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [
            {
                "position_sec": 0.5,
                "max_pause_sec": 0.2,
            }
        ],
        "output_format": "numpy",
    }

    with pytest.raises(ValueError):
        _validate_request(service, payload)


def test_smooth_request_accepts_audio_insert_without_extra_ratio_fields():
    service = _load_api_service_module()

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [
            {
                "mode": "adjust",
                "pause_interval_sec": [0.5, 0.5],
                "audio_insert": {
                    "data": base64.b64encode(_make_insert_wav_bytes()).decode(),
                    "format": "wav",
                },
            }
        ],
        "output_format": "numpy",
    }

    request = _validate_request(service, payload)
    assert request.points[0].audio_insert is not None


def test_smooth_request_rejects_legacy_force_silence_ratio_field():
    service = _load_api_service_module()

    payload = {
        "audio": {
            "data": base64.b64encode(_make_wav_bytes()).decode(),
            "format": "wav",
        },
        "text": "hello",
        "points": [
            {
                "mode": "adjust",
                "pause_interval_sec": [0.5, 0.5],
                "force_silence_ratio": 1.2,
            }
        ],
        "output_format": "numpy",
    }

    with pytest.raises(ValidationError, match="force_silence_ratio"):
        _validate_request(service, payload)
