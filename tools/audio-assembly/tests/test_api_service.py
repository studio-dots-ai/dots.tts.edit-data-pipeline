from __future__ import annotations

import base64
import importlib.util
import sys
import types

from pathlib import Path

from fastapi.testclient import TestClient


def _load_api_service_module():
    module_path = Path(__file__).resolve().parents[1] / "api_service.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(
        "audio_assembly_api_service", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assemble_request_accepts_manual_time_spans(monkeypatch):
    service = _load_api_service_module()
    client = TestClient(service.app)
    captured = {}

    def _fake_run_assembly(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            audio=service.np.zeros(16, dtype=service.np.float32),
            sample_rate=service.TARGET_SR,
            splice_points=[],
            source_answers=[],
            target_answers=[],
            target_spans=[],
            details={"assembly_mode": "splice_and_smooth"},
        )

    monkeypatch.setattr(service, "run_assembly", _fake_run_assembly)

    wav_bytes = b"RIFFfake"
    payload = {
        "source_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
        "target_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
        "edit_tagged_text": "AA<target>BB</target>CC",
        "time_spans_from_source": [[0.0, 1.0], [2.0, 3.0]],
        "time_spans_from_target": [[0.5, 0.8]],
        "output_format": "json",
    }

    response = client.post("/assemble", json=payload)
    assert response.status_code == 200, response.text
    assert captured["time_spans_from_source"] == [[0.0, 1.0], [2.0, 3.0]]
    assert captured["time_spans_from_target"] == [[0.5, 0.8]]
    assert captured["preserve_outer_boundaries"] is True


def test_assemble_request_allows_malformed_manual_spans_for_fallback(monkeypatch):
    service = _load_api_service_module()
    client = TestClient(service.app)
    captured = {}

    def _fake_run_assembly(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            audio=service.np.zeros(16, dtype=service.np.float32),
            sample_rate=service.TARGET_SR,
            splice_points=[],
            source_answers=[],
            target_answers=[],
            target_spans=[],
            details={"assembly_mode": "splice_and_smooth"},
        )

    monkeypatch.setattr(service, "run_assembly", _fake_run_assembly)

    wav_bytes = b"RIFFfake"
    payload = {
        "source_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
        "target_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
        "edit_tagged_text": "AA<target>BB</target>CC",
        "time_spans_from_source": [[0.0]],
        "time_spans_from_target": ["bad-shape"],
        "output_format": "json",
    }

    response = client.post("/assemble", json=payload)
    assert response.status_code == 200, response.text
    assert captured["time_spans_from_source"] == [[0.0]]
    assert captured["time_spans_from_target"] == ["bad-shape"]


def test_assemble_request_passes_preserve_outer_boundaries_flag(monkeypatch):
    service = _load_api_service_module()
    client = TestClient(service.app)
    captured = {}

    def _fake_run_assembly(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            audio=service.np.zeros(16, dtype=service.np.float32),
            sample_rate=service.TARGET_SR,
            splice_points=[],
            source_answers=[],
            target_answers=[],
            target_spans=[],
            details={"assembly_mode": "splice_and_smooth"},
        )

    monkeypatch.setattr(service, "run_assembly", _fake_run_assembly)

    wav_bytes = b"RIFFfake"
    payload = {
        "source_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
        "target_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
        "edit_tagged_text": "AA<target>BB</target>CC",
        "preserve_outer_boundaries": False,
        "output_format": "json",
    }

    response = client.post("/assemble", json=payload)
    assert response.status_code == 200, response.text
    assert captured["preserve_outer_boundaries"] is False
