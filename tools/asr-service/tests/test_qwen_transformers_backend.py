"""Unit tests for the stability-first Qwen3-ASR Transformers backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ASR_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASR_SERVICE_ROOT))

from backends import qwen3_transformers


class _FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, *, audio, language):
        self.calls.append((audio, language))
        return [SimpleNamespace(text="hello", language=language)]


def test_backend_loads_once_and_runs_each_input_sequentially(monkeypatch) -> None:
    fake_model = _FakeModel()
    loads = []

    def fake_load(**kwargs):
        loads.append(kwargs)
        return fake_model

    monkeypatch.setattr(qwen3_transformers, "_load_qwen_model", fake_load)
    backend = qwen3_transformers.Qwen3ASRTransformersBackend(
        model_path="Qwen/Qwen3-ASR-1.7B"
    )

    backend.load()
    backend.load()
    results = backend.transcribe_batch(
        [
            (np.ones(160, dtype=np.float32), 16000),
            (np.ones(320, dtype=np.float32), 16000),
        ],
        ["eng", "Chinese"],
    )

    assert len(loads) == 1
    assert loads[0]["device"] == "cuda:0"
    assert loads[0]["dtype_name"] == "bfloat16"
    assert [result.text for result in results] == ["hello", "hello"]
    assert [call[1] for call in fake_model.calls] == ["English", "Chinese"]
    assert backend.inference_name == "transformers"


def test_backend_returns_empty_result_for_exact_silence(monkeypatch) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr(
        qwen3_transformers,
        "_load_qwen_model",
        lambda **_kwargs: fake_model,
    )
    backend = qwen3_transformers.Qwen3ASRTransformersBackend(model_path="model")
    backend.load()

    result = backend.transcribe_single(
        (np.zeros(160, dtype=np.float32), 16000),
        "eng",
    )

    assert result.text == ""
    assert result.language == "English"
    assert fake_model.calls == []


def test_backend_rejects_mismatched_language_count(monkeypatch) -> None:
    monkeypatch.setattr(
        qwen3_transformers,
        "_load_qwen_model",
        lambda **_kwargs: _FakeModel(),
    )
    backend = qwen3_transformers.Qwen3ASRTransformersBackend(model_path="model")
    backend.load()

    with pytest.raises(ValueError, match="Batch size mismatch"):
        backend.transcribe_batch(
            [(np.ones(160, dtype=np.float32), 16000)],
            [],
        )


def test_resolver_uses_existing_snapshot_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    hub_root = tmp_path / "hub"
    repo = hub_root / "models--Qwen--Qwen3-ASR-1.7B"
    snapshot = repo / "snapshots" / "revision-123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("revision-123", encoding="utf-8")
    linked_root = tmp_path / "checkpoints"
    linked_root.mkdir()
    (linked_root / repo.name).symlink_to(repo, target_is_directory=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(linked_root))
    monkeypatch.setenv("CHECKPOINTS_DIR", str(linked_root))

    resolved = qwen3_transformers._resolve_local_model_path(
        "Qwen/Qwen3-ASR-1.7B"
    )

    assert resolved == str(snapshot)
