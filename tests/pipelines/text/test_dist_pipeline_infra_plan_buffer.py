from __future__ import annotations
from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/text_edit/defaults")


import hashlib
import json
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from speech_edit.pipelines.text.stage_adapter import (
    TextEditStageAdapter,
)
from speech_edit.runtime import worker as worker_module
from speech_edit.runtime.artifacts import ArtifactRef, ArtifactStore
from speech_edit.runtime.models import OutputItem, StageResult
from speech_edit.runtime.queue_store import QueueFull
from tests.service_graph_fixtures import direct_services


def _pg_test_run_id(tmp_path: Path) -> str:
    return f"textbuf_{hashlib.sha1(str(tmp_path).encode('utf-8')).hexdigest()[:12]}"


@dataclass
class _FakeQueueItem:
    id: int
    queue_name: str
    item_key: str
    payload_json: str | None = None
    payload_ref: str | None = None
    attempts: int = 0
    status: str = "ready"


class _FakeQueueStore:
    def get_run_state(self):
        return {"desired_state": "running"}

    def completion_stats(self):
        return {"run_complete": 0}

    def __init__(self, *args, **kwargs):
        self._next_id = 1
        self.items: list[_FakeQueueItem] = []
        self.heartbeats: list[dict[str, Any]] = []
        self.release_reasons: list[str] = []
        self.raise_queue_full_once = False
        self.max_in_progress = 0

    def _record_in_progress(self) -> None:
        active = sum(1 for item in self.items if item.status == "in_progress")
        self.max_in_progress = max(self.max_in_progress, active)

    def enqueue(
        self,
        queue_name: str,
        item_key: str,
        *,
        payload_json: dict[str, Any] | None = None,
        payload_ref: str | None = None,
    ) -> None:
        encoded = json.dumps(payload_json) if payload_json is not None else None
        self.items.append(
            _FakeQueueItem(
                id=self._next_id,
                queue_name=queue_name,
                item_key=item_key,
                payload_json=encoded,
                payload_ref=payload_ref,
            )
        )
        self._next_id += 1

    def claim(self, queue_name: str, worker_id: str, lease_seconds: float):
        for item in self.items:
            if item.queue_name == queue_name and item.status == "ready":
                item.status = "in_progress"
                item.attempts += 1
                self._record_in_progress()
                return item
        return None

    def has_queue_capacity(self, queue_name: str, *, requested: int, max_size: int | None) -> bool:
        if max_size is None:
            return True
        current = sum(
            1
            for item in self.items
            if item.queue_name == queue_name and item.status in {"ready", "in_progress"}
        )
        return current + requested <= max_size

    def ack_and_enqueue(
        self,
        item_id,
        *,
        worker_id,
        outputs,
        payload_refs,
        queue_max_sizes,
        clear_input_payload_ref=False,
    ):
        if self.raise_queue_full_once:
            self.raise_queue_full_once = False
            raise QueueFull("text.plan_to_synth", 1, 1, 1)
        for output in outputs:
            max_size = queue_max_sizes.get(output.queue_name) if queue_max_sizes else None
            if not self.has_queue_capacity(output.queue_name, requested=1, max_size=max_size):
                current = sum(
                    1
                    for item in self.items
                    if item.queue_name == output.queue_name and item.status in {"ready", "in_progress"}
                )
                raise QueueFull(output.queue_name, int(max_size), current, 1)
        cleanup_refs: list[str] = []
        for item in self.items:
            if item.id == item_id:
                if clear_input_payload_ref and item.payload_ref:
                    cleanup_refs.append(item.payload_ref)
                    item.payload_ref = None
                item.status = "done"
                break
        for output, payload_ref in zip(outputs, payload_refs):
            if any(
                item.queue_name == output.queue_name and item.item_key == output.item_key
                for item in self.items
            ):
                cleanup_refs.append(payload_ref)
                continue
            self.items.append(
                _FakeQueueItem(
                    id=self._next_id,
                    queue_name=output.queue_name,
                    item_key=output.item_key,
                    payload_ref=payload_ref,
                )
            )
            self._next_id += 1
        return types.SimpleNamespace(cleanup_payload_refs=cleanup_refs)

    def release_item(self, item_id, *, worker_id, reason):
        self.release_reasons.append(reason)
        for item in self.items:
            if item.id == item_id:
                item.status = "ready"
                item.attempts = max(item.attempts - 1, 0)
                return

    def fail_item(self, item_id, *, worker_id, error):
        for item in self.items:
            if item.id == item_id:
                item.status = "failed"
                return

    def queue_stats(self):
        stats: dict[str, dict[str, int]] = {}
        for item in self.items:
            queue_stats = stats.setdefault(item.queue_name, {})
            status = "ready" if item.status == "in_progress" else item.status
            queue_stats[status] = queue_stats.get(status, 0) + 1
        return stats

    def queue_status_counts(self, queue_name):
        return self.queue_stats().get(queue_name, {})

    def heartbeat(self, **kwargs):
        self.heartbeats.append(dict(kwargs))


def _write_run_config(tmp_path: Path, *, queue_limits: dict | None = None) -> tuple[Path, dict]:
    run_config = {
        "run_id": _pg_test_run_id(tmp_path),
        "pipeline": "text_edit",
        "run_dir": str(tmp_path / "run"),
        "artifact_root": str(tmp_path / "artifacts"),
        "pipeline_config": {**load_parameters("pipeline/text_edit/defaults"), },
        "stages": {
            "plan": {
                "worker_python_bin": "envs/pipe-cpu/.venv/bin/python",
                "max_workers": 1,
                "num_clients_per_worker": 2,
                "lease_seconds": 3600,
            }
        },
        "topology": {
            "bindings": {
                "text.sample_to_plan": {"consumers": ["plan"]},
                "text.plan_to_synth": {"producers": ["plan"]},
            }
        },
    }
    if queue_limits:
        run_config["queues"] = queue_limits
    run_config_path = tmp_path / "run.json"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    return run_config_path, run_config


def _patched_plan_adapter(monkeypatch: pytest.MonkeyPatch, process_plan_attempt):
    adapter = TextEditStageAdapter()
    monkeypatch.setattr(
        adapter,
        "_config",
        lambda ctx: types.SimpleNamespace(llm_max_retries=2),
    )
    monkeypatch.setattr(adapter, "process_plan_attempt", process_plan_attempt)
    monkeypatch.setattr(worker_module, "get_adapter", lambda pipeline, run_config: adapter)
    return adapter


def test_plan_worker_runs_client_threads_concurrently_without_extra_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    barrier = threading.Barrier(2, timeout=3)
    call_threads: list[int] = []

    def process_plan_attempt(payload, ctx, *, llm_client, config, attempt, validation_errors=()):
        call_threads.append(threading.get_ident())
        barrier.wait()
        return StageResult(
            outputs=[
                OutputItem(
                    queue_name="text.plan_to_synth",
                    item_key=payload["key"],
                    payload={"key": payload["key"], "attempt": attempt},
                )
            ]
        )

    _patched_plan_adapter(monkeypatch, process_plan_attempt)
    run_config_path, run_config = _write_run_config(tmp_path)
    store = _FakeQueueStore()
    monkeypatch.setattr(worker_module, "QueueStore", lambda *args, **kwargs: store)
    artifacts = ArtifactStore(run_config["artifact_root"])
    store.enqueue("text.sample_to_plan", "item-1", payload_json={"key": "item-1"})
    store.enqueue("text.sample_to_plan", "item-2", payload_json={"key": "item-2"})

    assert worker_module.run_worker(
        run_config_path=str(run_config_path),
        pipeline="text_edit",
        stage_name="plan",
        worker_id="worker-a",
        once=True,
        services=direct_services(llm="http://llm"),
    ) == 0

    assert len(set(call_threads)) == 2
    outputs = []
    for idx in range(2):
        claimed = store.claim("text.plan_to_synth", f"reader-{idx}", lease_seconds=10)
        assert claimed is not None
        outputs.append(artifacts.read_pickle(ArtifactRef.from_json(claimed.payload_ref or "{}")))
    assert sorted(output["key"] for output in outputs) == ["item-1", "item-2"]
    assert store.queue_stats()["text.sample_to_plan"].get("done") == 2
    assert store.max_in_progress == 2
    heartbeat_metadata = [
        item["metadata"]
        for item in store.heartbeats
        if item.get("stage") == "plan" and isinstance(item.get("metadata"), dict)
    ]
    assert any(
        metadata.get("num_clients_per_worker") == 2
        and metadata.get("item_key") in {"item-1", "item-2"}
        and metadata.get("client_index") in {0, 1}
        for metadata in heartbeat_metadata
    )


def test_worker_cleans_consumed_input_artifact_and_keeps_downstream_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def process_plan_attempt(payload, ctx, *, llm_client, config, attempt, validation_errors=()):
        return StageResult(
            outputs=[
                OutputItem(
                    queue_name="text.plan_to_synth",
                    item_key=payload["key"],
                    payload={"key": payload["key"], "attempt": attempt},
                )
            ]
        )

    _patched_plan_adapter(monkeypatch, process_plan_attempt)
    run_config_path, run_config = _write_run_config(tmp_path)
    store = _FakeQueueStore()
    monkeypatch.setattr(worker_module, "QueueStore", lambda *args, **kwargs: store)
    artifacts = ArtifactStore(run_config["artifact_root"])
    input_ref = artifacts.write_pickle(
        "text_edit_sample_outputs",
        "input-1",
        {"key": "item-1"},
    )
    store.enqueue(
        "text.sample_to_plan",
        "item-1",
        payload_ref=json.dumps(input_ref.to_json(), ensure_ascii=False, sort_keys=True),
    )

    assert worker_module.run_worker(
        run_config_path=str(run_config_path),
        pipeline="text_edit",
        stage_name="plan",
        worker_id="worker-a",
        once=True,
        services=direct_services(llm="http://llm"),
    ) == 0

    assert not artifacts.resolve(input_ref).exists()
    claimed = store.claim("text.plan_to_synth", "reader", lease_seconds=10)
    assert claimed is not None
    downstream_ref = ArtifactRef.from_json(claimed.payload_ref or "{}")
    assert artifacts.resolve(downstream_ref).exists()


def test_plan_worker_retries_failed_item_in_same_client_thread_with_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class AttemptFailed(RuntimeError):
        def __init__(self, message: str):
            super().__init__(message)
            self.validation_errors = ["bad forward json"]

    calls: dict[str, list[tuple[int, tuple[str, ...]]]] = {}

    def process_plan_attempt(payload, ctx, *, llm_client, config, attempt, validation_errors=()):
        calls.setdefault(payload["key"], []).append((attempt, tuple(validation_errors)))
        if payload["key"] == "retry" and not validation_errors:
            raise AttemptFailed("try again")
        return StageResult(
            outputs=[
                OutputItem(
                    queue_name="text.plan_to_synth",
                    item_key=payload["key"],
                    payload={"key": payload["key"], "attempt": attempt},
                )
            ]
        )

    _patched_plan_adapter(monkeypatch, process_plan_attempt)
    run_config_path, run_config = _write_run_config(tmp_path)
    store = _FakeQueueStore()
    monkeypatch.setattr(worker_module, "QueueStore", lambda *args, **kwargs: store)
    artifacts = ArtifactStore(run_config["artifact_root"])
    store.enqueue("text.sample_to_plan", "retry", payload_json={"key": "retry"})
    store.enqueue("text.sample_to_plan", "ok", payload_json={"key": "ok"})

    assert worker_module.run_worker(
        run_config_path=str(run_config_path),
        pipeline="text_edit",
        stage_name="plan",
        worker_id="worker-a",
        once=True,
        services=direct_services(llm="http://llm"),
    ) == 0

    assert calls["retry"] == [(1, ()), (2, ("bad forward json",))]
    assert calls["ok"] == [(1, ())]
    outputs = []
    for idx in range(2):
        claimed = store.claim("text.plan_to_synth", f"reader-{idx}", lease_seconds=10)
        assert claimed is not None
        outputs.append(artifacts.read_pickle(ArtifactRef.from_json(claimed.payload_ref or "{}")))
    assert sorted((output["key"], output["attempt"]) for output in outputs) == [
        ("ok", 1),
        ("retry", 2),
    ]
    assert store.queue_stats()["text.sample_to_plan"].get("done") == 2
    assert store.release_reasons == []


def test_plan_worker_stops_buffer_fill_at_output_backpressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def process_plan_attempt(payload, ctx, *, llm_client, config, attempt, validation_errors=()):
        return StageResult(
            outputs=[
                OutputItem(
                    queue_name="text.plan_to_synth",
                    item_key=payload["key"],
                    payload={"key": payload["key"]},
                )
            ]
        )

    _patched_plan_adapter(monkeypatch, process_plan_attempt)
    run_config_path, run_config = _write_run_config(
        tmp_path,
        queue_limits={"text.plan_to_synth": {"max_size": 1}},
    )
    store = _FakeQueueStore()
    monkeypatch.setattr(worker_module, "QueueStore", lambda *args, **kwargs: store)
    store.enqueue("text.sample_to_plan", "item-1", payload_json={"key": "item-1"})
    store.enqueue("text.sample_to_plan", "item-2", payload_json={"key": "item-2"})

    assert worker_module.run_worker(
        run_config_path=str(run_config_path),
        pipeline="text_edit",
        stage_name="plan",
        worker_id="worker-a",
        once=True,
        services=direct_services(llm="http://llm"),
    ) == 0

    assert store.queue_stats()["text.plan_to_synth"].get("ready") == 1
    assert store.queue_stats()["text.sample_to_plan"].get("ready") == 1


def test_plan_worker_releases_input_when_output_queue_fills_during_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def process_plan_attempt(payload, ctx, *, llm_client, config, attempt, validation_errors=()):
        return StageResult(
            outputs=[
                OutputItem(
                    queue_name="text.plan_to_synth",
                    item_key=payload["key"],
                    payload={"key": payload["key"]},
                )
            ]
        )

    _patched_plan_adapter(monkeypatch, process_plan_attempt)
    run_config_path, run_config = _write_run_config(
        tmp_path,
        queue_limits={"text.plan_to_synth": {"max_size": 4}},
    )
    store = _FakeQueueStore()
    store.raise_queue_full_once = True
    monkeypatch.setattr(worker_module, "QueueStore", lambda *args, **kwargs: store)
    store.enqueue("text.sample_to_plan", "item-1", payload_json={"key": "item-1"})

    assert worker_module.run_worker(
        run_config_path=str(run_config_path),
        pipeline="text_edit",
        stage_name="plan",
        worker_id="worker-a",
        once=True,
        services=direct_services(llm="http://llm"),
    ) == 0

    assert store.queue_stats()["text.sample_to_plan"].get("ready") == 1
    assert store.items[0].attempts == 0
    assert store.release_reasons
    assert "text.plan_to_synth" in store.release_reasons[0]
