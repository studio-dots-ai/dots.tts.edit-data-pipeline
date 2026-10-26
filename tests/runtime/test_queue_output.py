from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speech_edit.runtime import local_runner as local_runner_module
from speech_edit.runtime import artifact_gc as artifact_gc_module
from speech_edit.runtime import config as config_module
from speech_edit.runtime import registry as registry_module
from speech_edit.runtime import source_seed as source_seed_module
from speech_edit.runtime.artifacts import ArtifactRef, ArtifactStore
from speech_edit.runtime import submit_run as submit_run_module
from speech_edit.runtime import worker as worker_module
from speech_edit.runtime.models import (
    DEFAULT_STAGE_WORKER_PYTHON_BIN,
    OutputItem,
    ParquetOutputConfig,
    StageResult,
    StageSpec,
)
from tests.service_graph_fixtures import direct_services
from speech_edit.runtime.parquet_output import ParquetOutputWriter
from speech_edit.runtime.queue_store import QueueFull, QueueStore
from speech_edit.data.parquet.reader import read_segments_from_parquet_payload
from speech_edit.runtime.source_seed import seed_parquet_queue, seed_source_from_run_config


_DIST_PIPELINE_DB_TEST_MARKERS = (
    "QueueStore(",
    "seed_parquet_queue(",
    "seed_source_from_run_config(",
    "worker_module.run_worker(",
    "local_runner_module.run_local(",
    "dry_run=False",
)


@pytest.fixture(autouse=True)
def _skip_dist_pipeline_db_tests_without_database_url(request: pytest.FixtureRequest):
    if os.environ.get("DIST_PIPELINE_DATABASE_URL"):
        return
    source = inspect.getsource(request.function)
    if any(marker in source for marker in _DIST_PIPELINE_DB_TEST_MARKERS):
        pytest.skip("DIST_PIPELINE_DATABASE_URL is required for PostgreSQL-backed runtime tests")


def _pg_test_run_id(tmp_path: Path) -> str:
    return f"unit_{hashlib.sha1(str(tmp_path).encode('utf-8')).hexdigest()[:12]}"


def _stage_spec(name: str, input_queue: str | None, output_queue: str | None, **kwargs) -> StageSpec:
    kwargs.setdefault("worker_python_bin", DEFAULT_STAGE_WORKER_PYTHON_BIN)
    return StageSpec(name=name, input_queue=input_queue, output_queue=output_queue, **kwargs)


def test_queue_claim_lease_prevents_duplicate_until_expiry(tmp_path: Path, monkeypatch):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    clock = [time.time()]
    monkeypatch.setattr(store, "_now", lambda: clock[0])
    assert store.enqueue("q1", "item-1", payload_json={"x": 1})

    first = store.claim("q1", "worker-a", lease_seconds=0.05)
    assert first is not None
    assert store.claim("q1", "worker-b", lease_seconds=1.0) is None

    clock[0] += 0.08
    reclaimed = store.claim("q1", "worker-b", lease_seconds=1.0)
    assert reclaimed is not None
    assert reclaimed.id == first.id
    assert reclaimed.attempts == 2


def test_seed_parquet_queue_can_chunk_source_rows(tmp_path: Path):
    parquet_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4, 5]}), parquet_path)

    inserted = seed_parquet_queue(
        run_id=_pg_test_run_id(tmp_path),
        queue_name="source_items",
        sources=[str(parquet_path)],
        rows_per_item=2,
    )

    assert inserted == 3
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    assert store.queue_stats()["source_items"]["ready"] == 3
    claimed = store.claim("source_items", "worker-a", lease_seconds=10)
    assert claimed is not None
    payload = json.loads(claimed.payload_json or "{}")
    assert payload["row_start"] == 0
    assert payload["row_stop"] == 2


def test_output_item_defaults_to_single_attempt():
    item = OutputItem(queue_name="out", item_key="item-1", payload={"x": 1})
    assert item.max_attempts == 1


def test_artifact_store_delete_ref_removes_file_and_empty_parent(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.write_json("payloads", "item-1", {"x": 1})
    path = store.resolve(ref)

    result = store.delete_ref(ref)

    assert result.deleted_count == 1
    assert result.bytes_freed > 0
    assert result.failures == []
    assert not path.exists()
    assert not (tmp_path / "artifacts" / "payloads").exists()
    assert store.delete_ref(ref).deleted_count == 0


def test_artifact_store_delete_ref_can_preserve_shared_parent(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.write_json("payloads", "item-1", {"x": 1})
    payload_dir = store.resolve(ref).parent

    result = store.delete_ref(ref, prune_empty_parents=False)

    assert result.deleted_count == 1
    assert result.failures == []
    assert payload_dir.is_dir()
    replacement = store.write_json("payloads", "item-2", {"x": 2})
    assert store.read_json(replacement) == {"x": 2}


def test_online_worker_cleanup_preserves_shared_artifact_parent(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.write_pickle("shared_stage_outputs", "item-1", {"x": 1})
    payload_dir = store.resolve(ref).parent

    worker_module._cleanup_artifact_refs(
        store,
        [json.dumps(ref.to_json())],
        stage_name="eval",
        worker_id="worker-a",
        reason="ack_success",
    )

    assert payload_dir.is_dir()
    replacement = store.write_pickle("shared_stage_outputs", "item-2", {"x": 2})
    assert store.read_pickle(replacement) == {"x": 2}


def test_artifact_store_delete_ref_rejects_paths_outside_root(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"keep")

    result = store.delete_ref({"path": "../outside.pkl", "sha256": "x", "size_bytes": 4})

    assert result.deleted_count == 0
    assert result.failures
    assert outside.read_bytes() == b"keep"


def test_artifact_gc_dry_run_summarizes_db_without_deleting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_config_path = tmp_path / "run_config.json"
    artifact_root = tmp_path / "artifacts"
    artifact_store = ArtifactStore(artifact_root)
    ref = artifact_store.write_pickle("payloads", "item-1", {"x": 1})
    run_config_path.write_text(
        json.dumps(
            {
                "run_id": "unit",
                "run_dir": str(tmp_path / "run"),
                "artifact_root": str(artifact_root),
            }
        ),
        encoding="utf-8",
    )

    class FakeStore:
        def __init__(self, **kwargs):
            pass

        def artifact_payload_ref_summary(self, *, statuses):
            return [
                {
                    "queue_name": "next",
                    "status": "done",
                    "item_count": 1,
                    "size_bytes": ref.size_bytes,
                }
            ]

    monkeypatch.setenv("DIST_PIPELINE_DATABASE_URL", "postgresql://unit")
    monkeypatch.setattr(artifact_gc_module, "QueueStore", FakeStore)

    result = artifact_gc_module.run_gc(
        run_config_path=str(run_config_path),
        statuses=["done"],
        execute=False,
        batch_size=10,
    )

    assert result["dry_run"] is True
    assert result["totals"]["item_count"] == 1
    assert artifact_store.resolve(ref).exists()


def test_artifact_gc_execute_deletes_done_refs_and_clears_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_config_path = tmp_path / "run_config.json"
    artifact_root = tmp_path / "artifacts"
    artifact_store = ArtifactStore(artifact_root)
    ref = artifact_store.write_pickle("payloads", "item-1", {"x": 1})
    cleared: list[int] = []
    run_config_path.write_text(
        json.dumps(
            {
                "run_id": "unit",
                "run_dir": str(tmp_path / "run"),
                "artifact_root": str(artifact_root),
            }
        ),
        encoding="utf-8",
    )

    class FakeStore:
        def __init__(self, **kwargs):
            pass

        def artifact_payload_ref_summary(self, *, statuses):
            if cleared:
                return []
            return [
                {
                    "queue_name": "next",
                    "status": "done",
                    "item_count": 1,
                    "size_bytes": ref.size_bytes,
                }
            ]

        def artifact_payload_ref_batch(self, *, statuses, limit):
            if cleared:
                return []
            return [
                {
                    "id": 7,
                    "queue_name": "next",
                    "item_key": "item-1",
                    "status": "done",
                    "payload_ref": json.dumps(ref.to_json(), ensure_ascii=False, sort_keys=True),
                }
            ]

        def clear_artifact_payload_refs(self, *, item_ids, statuses):
            cleared.extend(item_ids)
            return len(item_ids)

    monkeypatch.setenv("DIST_PIPELINE_DATABASE_URL", "postgresql://unit")
    monkeypatch.setattr(artifact_gc_module, "QueueStore", FakeStore)

    result = artifact_gc_module.run_gc(
        run_config_path=str(run_config_path),
        statuses=["done"],
        execute=True,
        batch_size=10,
    )

    assert result["deleted_count"] == 1
    assert result["cleared_count"] == 1
    assert cleared == [7]
    assert not artifact_store.resolve(ref).exists()


def test_queue_store_uses_external_connection_pool_without_owning_it():
    class FakeConnection:
        def __init__(self):
            self.commits = 0
            self.rollbacks = 0

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    class FakeConnectionContext:
        def __init__(self, pool):
            self.pool = pool

        def __enter__(self):
            self.pool.checkouts += 1
            return self.pool.connection_obj

        def __exit__(self, exc_type, exc, tb):
            self.pool.returns += 1
            return False

    class FakePool:
        def __init__(self):
            self.connection_obj = FakeConnection()
            self.checkouts = 0
            self.returns = 0
            self.closed = False

        def connection(self):
            return FakeConnectionContext(self)

        def close(self):
            self.closed = True

    pool = FakePool()
    store_cls = QueueStore
    store = store_cls(
        run_id="unit_pool",
        database_url="postgresql://unit",
        connection_pool=pool,
        ensure_schema=False,
    )

    with store.transaction() as conn:
        assert conn is pool.connection_obj

    assert pool.checkouts == 1
    assert pool.returns == 1
    assert pool.connection_obj.commits == 1
    store.close()
    assert pool.closed is False


def test_seed_parquet_queue_is_idempotent_for_resume(tmp_path: Path):
    parquet_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4]}), parquet_path)
    queue_db = tmp_path / "queue.sqlite3"

    first = seed_parquet_queue(
        run_id=_pg_test_run_id(tmp_path),
        queue_name="source_items",
        sources=[str(parquet_path)],
        rows_per_item=2,
    )
    second = seed_parquet_queue(
        run_id=_pg_test_run_id(tmp_path),
        queue_name="source_items",
        sources=[str(parquet_path)],
        rows_per_item=2,
    )

    assert first == 2
    assert second == 0
    assert QueueStore(run_id=_pg_test_run_id(tmp_path)).queue_stats()["source_items"]["ready"] == 2


def test_seed_parquet_queue_uses_source_index_cache_and_progress(tmp_path: Path):
    parquet_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4]}), parquet_path)
    cache_path = tmp_path / "source_index.jsonl"
    messages: list[str] = []

    first = seed_parquet_queue(
        run_id=_pg_test_run_id(tmp_path),
        queue_name="source_items",
        sources=[str(parquet_path)],
        rows_per_item=2,
        source_index_cache=str(cache_path),
        progress=messages.append,
        progress_interval=1,
        metadata_workers=1,
    )
    second = seed_parquet_queue(
        run_id=_pg_test_run_id(tmp_path),
        queue_name="source_items",
        sources=[str(parquet_path)],
        rows_per_item=2,
        source_index_cache=str(cache_path),
        progress=messages.append,
        progress_interval=1,
        metadata_workers=1,
    )

    assert first == 2
    assert second == 0
    assert cache_path.exists()
    assert any(message.startswith("source_index_progress ") for message in messages)
    assert any(message.startswith("source_seed_progress ") for message in messages)
    assert any(message.startswith("source_index_cache_hit ") for message in messages)


def test_seed_parquet_queue_max_items_per_input_limits_each_source(tmp_path: Path):
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4, 5, 6]}), first_path)
    pq.write_table(pa.table({"x": [7, 8, 9, 10, 11, 12]}), second_path)
    cache_path = tmp_path / "source_index.jsonl"
    messages: list[str] = []

    inserted = seed_parquet_queue(
        run_id=_pg_test_run_id(tmp_path),
        queue_name="source_items",
        sources=[str(first_path), str(second_path)],
        rows_per_item=2,
        source_index_cache=str(cache_path),
        progress=messages.append,
        progress_interval=1,
        metadata_workers=1,
        max_items_per_input=2,
    )

    assert inserted == 4
    assert QueueStore(run_id=_pg_test_run_id(tmp_path)).queue_stats()["source_items"]["ready"] == 4
    lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["item_count"] == 6
    assert any(
        message == "source_seed_limited_per_input items=4/6 max_items_per_input=2"
        for message in messages
    )


def test_seed_source_from_run_config_does_not_treat_source_queue_as_backpressure(
    tmp_path: Path,
):
    parquet_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4]}), parquet_path)
    queue_db = tmp_path / "queue.sqlite3"

    inserted = seed_source_from_run_config(
        run_config={
            "run_id": _pg_test_run_id(tmp_path),
            "source": {
                "inputs": [{"path": str(parquet_path)}],
                "queue_name": "source_items",
                "rows_per_item": 1,
            },
            "queues": {"source_items": {"max_size": 1}},
        }
    )

    assert inserted == 4
    assert QueueStore(run_id=_pg_test_run_id(tmp_path)).queue_stats()["source_items"]["ready"] == 4


def test_source_segments_filters_language_and_uses_shuffle_source_rows(tmp_path: Path):
    import io

    import numpy as np
    import pandas as pd
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(2400, dtype=np.float32), 24000, format="WAV")
    audio_bytes = buffer.getvalue()
    parquet_path = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {"clip_id": "en_1", "text": "hello world", "audio": audio_bytes, "duration": 0.1},
            {"clip_id": "zh_1", "text": "你好世界", "audio": audio_bytes, "duration": 0.1},
        ]
    ).to_parquet(parquet_path, index=False)

    segments = read_segments_from_parquet_payload(
        {"parquet_path": str(parquet_path), "row_start": 0, "row_stop": 2},
        {
            "sample_language": "zh",
            "min_duration": 0.0,
            "max_duration": 1.0,
            "shuffle_source_rows": False,
        },
    )

    assert [segment.clip_id for segment in segments] == ["zh_1"]
    assert segments[0].metadata["language"] == "zh"


def test_pipeline_default_dist_pipeline_configs_load(tmp_path: Path):
    config_paths = [
        Path("configs/pipeline/text_edit/defaults.yaml"),
        Path("configs/pipeline/prosody_edit/defaults.yaml"),
        Path("configs/pipeline/emotion_style_edit/defaults.yaml"),
    ]
    for config_path in config_paths:
        loaded = config_module.load_run_config(config_path)
        loaded["run_id"] = "unit"
        loaded["artifact_root"] = str(tmp_path / "text_edit" / "artifacts")
        adapter = registry_module.get_adapter(str(loaded["pipeline"]), loaded)
        specs = adapter.stage_specs(loaded)
        assert "sample" in specs
        assert "save" in specs
        assert specs["sample"].input_queue == "source_items"
        assert specs["save"].parquet_output.enabled is True
        assert specs["save"].parquet_output.output_dir == str(
            Path(loaded["run_dir"]) / "parquet"
        )
        assert specs["save"].lease_seconds == loaded["stages"]["save"]["lease_seconds"]
        assert loaded["source"]["inputs"] == [
            {"path": "/path/to/data/emilia"}
        ]
        normalized_source = config_module.source_config(loaded)
        assert normalized_source["inputs"] == loaded["source"]["inputs"]
        assert loaded["pipeline_config"]["output_dataset_name"]
        for stage in loaded["stages"].values():
            assert stage["queue_store"]["pool"] == {
                "enabled": True,
                "min_size": 2,
                "max_size": 2,
                "timeout_sec": 120,
            }


def test_source_config_preserves_inputs_and_seeder_prefers_them(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_seed_queue(**kwargs):
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(source_seed_module, "seed_parquet_queue", fake_seed_queue)
    run_config = {
        "run_id": "unit",
        "source": {
            "inputs": [{"path": "/data/new.list"}],
            "queue_name": "source_items",
            "rows_per_item": 128,
        },
    }

    source = config_module.source_config(run_config)
    inserted = getattr(source_seed_module, "seed_source_from_run_config")(run_config=run_config)

    assert inserted == 7
    assert source["inputs"] == [{"path": "/data/new.list"}]
    assert captured["sources"] == [{"path": "/data/new.list"}]
    assert captured["rows_per_item"] == 128


def test_stage_specs_from_config_uses_stage_spec_defaults_for_optional_fields():
    run_config = {
        "stages": {
            "sample": {"worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN},
            "save": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "parquet_output": {"enabled": True, "output_dir": "/runs/out/parquet"},
            },
        },
        "topology": {
            "bindings": {
                "source_items": {"consumers": ["sample"]},
                "sample_to_save": {"producers": ["sample"], "consumers": ["save"]},
                "save_results": {"producers": ["save"]},
            }
        },
    }

    specs = config_module.stage_specs_from_config(run_config)

    assert specs["sample"].input_queue == "source_items"
    assert specs["sample"].output_queue == "sample_to_save"
    assert specs["sample"].lease_seconds == _stage_spec(
        name="sample",
        input_queue=None,
        output_queue=None,
    ).lease_seconds
    assert specs["sample"].max_workers == _stage_spec(
        name="sample",
        input_queue=None,
        output_queue=None,
    ).max_workers
    assert specs["save"].input_queue == "sample_to_save"
    assert specs["save"].output_queue == "save_results"
    assert specs["save"].parquet_output.enabled is True


def test_stage_specs_from_config_requires_stage_worker_python_bin():
    run_config = {
        "stages": {"sample": {}},
        "topology": {"bindings": {"source_items": {"consumers": ["sample"]}}},
    }

    with pytest.raises(ValueError, match="stage 'sample' must define worker_python_bin"):
        config_module.stage_specs_from_config(run_config)


def test_stage_specs_from_config_supports_split_and_merged_stage_topology():
    run_config = {
        "stages": {
            "plan": {"worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN},
            "synth_index": {"worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN},
            "synth_ted": {"worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN},
            "eval": {"worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN},
        },
        "topology": {
            "bindings": {
                "plan_in": {"consumers": ["plan"]},
                "plan_to_index": {"producers": ["plan"], "consumers": ["synth_index"]},
                "plan_to_ted": {"producers": ["plan"], "consumers": ["synth_ted"]},
                "synth_to_eval": {
                    "producers": ["synth_index", "synth_ted"],
                    "consumers": ["eval"],
                },
            }
        },
    }

    specs = config_module.stage_specs_from_config(run_config)

    assert specs["plan"].input_queue == "plan_in"
    assert specs["plan"].output_queue is None
    assert specs["synth_index"].input_queue == "plan_to_index"
    assert specs["synth_index"].output_queue == "synth_to_eval"
    assert specs["synth_ted"].input_queue == "plan_to_ted"
    assert specs["synth_ted"].output_queue == "synth_to_eval"
    assert specs["eval"].input_queue == "synth_to_eval"


def test_ack_and_enqueue_is_single_transaction(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    store.enqueue("source", "src-1", payload_json={"x": 1})
    claimed = store.claim("source", "worker-a", lease_seconds=10)
    assert claimed is not None

    store.ack_and_enqueue(
        claimed.id,
        worker_id="worker-a",
        outputs=[
            OutputItem(
                queue_name="next",
                item_key="next-1",
                payload={"not_used": True},
            )
        ],
        payload_refs=[json.dumps({"path": "payloads/next.pkl", "sha256": "x", "size_bytes": 1})],
    )

    stats = store.queue_stats()
    assert stats["source"]["done"] == 1
    assert stats["next"]["ready"] == 1


def test_ack_and_enqueue_respects_output_queue_max_size(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    store.enqueue("source", "src-1", payload_json={"x": 1})
    store.enqueue("next", "existing", payload_json={"x": 2})
    claimed = store.claim("source", "worker-a", lease_seconds=10)
    assert claimed is not None

    with pytest.raises(QueueFull):
        store.ack_and_enqueue(
            claimed.id,
            worker_id="worker-a",
            outputs=[
                OutputItem(
                    queue_name="next",
                    item_key="next-2",
                    payload={"not_used": True},
                )
            ],
            payload_refs=[json.dumps({"path": "payloads/next.pkl", "sha256": "x", "size_bytes": 1})],
            queue_max_sizes={"next": 1},
        )

    stats = store.queue_stats()
    assert stats["source"]["leased"] == 1
    assert stats["next"]["ready"] == 1


def test_ack_and_enqueue_can_clear_consumed_payload_ref(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    input_ref = json.dumps({"path": "payloads/input.pkl", "sha256": "in", "size_bytes": 2})
    output_ref = json.dumps({"path": "payloads/output.pkl", "sha256": "out", "size_bytes": 3})
    store.enqueue("source", "src-1", payload_ref=input_ref)
    claimed = store.claim("source", "worker-a", lease_seconds=10)
    assert claimed is not None

    ack_result = store.ack_and_enqueue(
        claimed.id,
        worker_id="worker-a",
        outputs=[OutputItem(queue_name="next", item_key="next-1", payload={"not_used": True})],
        payload_refs=[output_ref],
        clear_input_payload_ref=True,
    )

    assert ack_result.input_payload_ref == input_ref
    assert ack_result.discarded_payload_refs == []
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT queue_name, item_key, payload_ref, status
                FROM queue_items
                WHERE run_id=%s
                ORDER BY queue_name, item_key
                """,
                (store.run_id,),
            )
            rows = cur.fetchall()
    assert rows[0]["queue_name"] == "next"
    assert rows[0]["payload_ref"] == output_ref
    assert rows[1]["queue_name"] == "source"
    assert rows[1]["payload_ref"] is None
    assert rows[1]["status"] == "done"


def test_ack_and_enqueue_returns_discarded_output_refs_on_conflict(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    input_ref = json.dumps({"path": "payloads/input.pkl", "sha256": "in", "size_bytes": 2})
    existing_ref = json.dumps({"path": "payloads/existing.pkl", "sha256": "old", "size_bytes": 3})
    discarded_ref = json.dumps({"path": "payloads/new.pkl", "sha256": "new", "size_bytes": 4})
    store.enqueue("source", "src-1", payload_ref=input_ref)
    store.enqueue("next", "next-1", payload_ref=existing_ref)
    claimed = store.claim("source", "worker-a", lease_seconds=10)
    assert claimed is not None

    ack_result = store.ack_and_enqueue(
        claimed.id,
        worker_id="worker-a",
        outputs=[OutputItem(queue_name="next", item_key="next-1", payload={"not_used": True})],
        payload_refs=[discarded_ref],
        clear_input_payload_ref=True,
    )

    assert ack_result.input_payload_ref == input_ref
    assert ack_result.discarded_payload_refs == [discarded_ref]
    claimed_next = store.claim("next", "worker-b", lease_seconds=10)
    assert claimed_next is not None
    assert claimed_next.payload_ref == existing_ref


def test_parquet_output_flushes_rows_then_acks_items(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    parquet_dir = tmp_path / "parquet"
    writer = ParquetOutputWriter(
        queue_store=store,
        config=ParquetOutputConfig(
            enabled=True,
            output_dir=str(parquet_dir),
            flush_rows=2,
            flush_interval_sec=999,
        ),
        writer_id="worker-a",
    )

    store.enqueue("eval", "item-1", payload_json={"sample": 1})
    store.enqueue("eval", "item-2", payload_json={"sample": 2})
    first = store.claim("eval", "worker-a", lease_seconds=10)
    second = store.claim("eval", "worker-a", lease_seconds=10)
    assert first is not None
    assert second is not None

    writer.add_item_rows(
        queue_item=first,
        rows=[{"audio_id": "row-1", "from_dataset": "unit", "sample_rate": 24000}],
    )
    assert writer.flush_due(force=False) is None
    assert store.queue_stats()["eval"]["leased"] == 2

    writer.add_item_rows(
        queue_item=second,
        rows=[{"audio_id": "row-2", "from_dataset": "unit", "sample_rate": 24000}],
    )
    output_path = writer.flush_due(force=False)
    assert output_path is not None
    table = pq.read_table(output_path)
    assert table.num_rows == 2
    assert set(table.column("audio_id").to_pylist()) == {"row-1", "row-2"}
    assert store.queue_stats()["eval"]["done"] == 2


def test_parquet_output_empty_rows_ack_immediately(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    writer = ParquetOutputWriter(
        queue_store=store,
        config=ParquetOutputConfig(enabled=True, output_dir=str(tmp_path / "parquet")),
        writer_id="worker-a",
    )
    store.enqueue("eval", "item-1", payload_json={"sample": 1})
    claimed = store.claim("eval", "worker-a", lease_seconds=10)
    assert claimed is not None

    writer.add_item_rows(queue_item=claimed, rows=[])

    assert store.queue_stats()["eval"]["done"] == 1
    assert writer.flush_due(force=True) is None


def test_parquet_output_cleans_input_artifact_after_successful_flush(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    ref = artifact_store.write_pickle("eval_inputs", "item-1", {"sample": 1})
    store.enqueue("eval", "item-1", payload_ref=json.dumps(ref.to_json()))
    writer = ParquetOutputWriter(
        queue_store=store,
        artifact_store=artifact_store,
        config=ParquetOutputConfig(enabled=True, output_dir=str(tmp_path / "parquet"), flush_rows=1),
        writer_id="worker-a",
        cleanup_artifacts=True,
    )
    claimed = store.claim("eval", "worker-a", lease_seconds=10)
    assert claimed is not None

    writer.add_item_rows(
        queue_item=claimed,
        rows=[{"audio_id": "row-1", "from_dataset": "unit", "sample_rate": 24000}],
    )
    assert writer.flush_due(force=False) is not None

    assert not artifact_store.resolve(ref).exists()
    with store._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload_ref FROM queue_items WHERE run_id=%s", (store.run_id,))
            row = cur.fetchone()
    assert row["payload_ref"] is None


def test_parquet_output_releases_items_when_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    writer = ParquetOutputWriter(
        queue_store=store,
        config=ParquetOutputConfig(enabled=True, output_dir=str(tmp_path / "parquet"), flush_rows=1),
        writer_id="worker-a",
    )
    store.enqueue("eval", "item-1", payload_json={"sample": 1})
    claimed = store.claim("eval", "worker-a", lease_seconds=10)
    assert claimed is not None
    writer.add_item_rows(
        queue_item=claimed,
        rows=[{"audio_id": "row-1", "from_dataset": "unit"}],
    )

    def fail_write(rows, final_path):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_part", fail_write)
    with pytest.raises(OSError, match="disk full"):
        writer.flush_due(force=True)
    assert store.queue_stats()["eval"]["ready"] == 1


def test_parquet_output_rejects_expired_item_before_write(tmp_path: Path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    writer = ParquetOutputWriter(
        queue_store=store,
        config=ParquetOutputConfig(enabled=True, output_dir=str(tmp_path / "parquet"), flush_rows=1),
        writer_id="worker-a",
    )
    store.enqueue("eval", "item-1", payload_json={"sample": 1})
    claimed = store.claim("eval", "worker-a", lease_seconds=0.05)
    assert claimed is not None
    writer.add_item_rows(
        queue_item=claimed,
        rows=[{"audio_id": "stale-row", "from_dataset": "unit"}],
    )
    time.sleep(0.08)

    with pytest.raises(RuntimeError, match="lease expired"):
        writer.flush_due(force=True)
    assert list((tmp_path / "parquet").rglob("*.parquet")) == []


def test_worker_once_materializes_save_stage_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeAdapter:
        def stage_specs(self, run_config):
            return {
                "save": _stage_spec(
                    name="save",
                    input_queue="save_q",
                    output_queue=None,
                    parquet_output=ParquetOutputConfig(
                        enabled=True,
                        output_dir=str(Path(run_config["run_dir"]) / "parquet"),
                        flush_rows=10,
                        flush_interval_sec=999,
                    ),
                )
            }

        def process(self, stage_name, payload, ctx):
            assert stage_name == "save"
            assert payload == {"sample": 1}
            return StageResult(
                parquet_rows=[
                    {
                        "audio_id": "row-1",
                        "from_dataset": "unit",
                        "sample_rate": 24000,
                    }
                ]
            )

    monkeypatch.setattr(worker_module, "get_adapter", lambda pipeline, run_config: FakeAdapter())
    run_config_path = tmp_path / "run.json"
    run_config = {
        "run_id": _pg_test_run_id(tmp_path),
        "pipeline": "fake",
        "run_dir": str(tmp_path / "run"),
        "artifact_root": str(tmp_path / "artifacts"),
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    store = QueueStore(run_id=run_config["run_id"])
    store.enqueue("save_q", "item-1", payload_json={"sample": 1})

    run_worker = worker_module.run_worker
    assert run_worker(
        run_config_path=str(run_config_path),
        pipeline="fake",
        stage_name="save",
        worker_id="worker-a",
        once=True,
        services=direct_services(),
    ) == 0

    assert store.queue_stats()["save_q"]["done"] == 1
    parts = list((Path(run_config["run_dir"]) / "parquet" / "worker_worker-a").glob("*.parquet"))
    assert len(parts) == 1
    assert pq.read_table(parts[0]).column("audio_id").to_pylist() == ["row-1"]
    assert store.completion_stats()["run_complete"] == 1


def test_worker_fanout_payload_paths_are_queue_namespaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeAdapter:
        def stage_specs(self, run_config):
            return {
                "split": _stage_spec(
                    name="split",
                    input_queue="source",
                    output_queue=None,
                )
            }

        def process(self, stage_name, payload, ctx):
            assert stage_name == "split"
            return StageResult(
                outputs=[
                    OutputItem(queue_name="left", item_key="same", payload={"branch": "left"}),
                    OutputItem(queue_name="right", item_key="same", payload={"branch": "right"}),
                ]
            )

    monkeypatch.setattr(worker_module, "get_adapter", lambda pipeline, run_config: FakeAdapter())
    run_config_path = tmp_path / "run.json"
    run_config = {
        "run_id": _pg_test_run_id(tmp_path),
        "pipeline": "fake",
        "run_dir": str(tmp_path / "run"),
        "artifact_root": str(tmp_path / "artifacts"),
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    store = QueueStore(run_id=run_config["run_id"])
    artifacts = ArtifactStore(run_config["artifact_root"])
    store.enqueue("source", "item-1", payload_json={"sample": 1})

    worker_module.run_worker(
        run_config_path=str(run_config_path),
        pipeline="fake",
        stage_name="split",
        worker_id="worker-a",
        once=True,
        services=direct_services(),
    )

    left = store.claim("left", "left-worker", lease_seconds=10)
    right = store.claim("right", "right-worker", lease_seconds=10)
    assert left is not None
    assert right is not None
    assert artifacts.read_pickle(ArtifactRef.from_json(left.payload_ref or "{}")) == {"branch": "left"}
    assert artifacts.read_pickle(ArtifactRef.from_json(right.payload_ref or "{}")) == {"branch": "right"}


def test_worker_queue_store_pool_can_be_disabled_by_stage_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store_kwargs = []

    class FakeStore:
        def get_run_state(self):
            return {"desired_state": "running"}

        def completion_stats(self):
            return {"run_complete": 0}


        def claim(self, *args):
            return None

        def heartbeat(self, **kwargs):
            return None


    class FakeAdapter:
        def stage_specs(self, run_config):
            return {"custom": _stage_spec(name="custom", input_queue="in_q", output_queue=None)}


    def fake_queue_store(*args, **kwargs):
        store_kwargs.append(kwargs)
        return FakeStore()

    monkeypatch.setattr(worker_module, "get_adapter", lambda pipeline, run_config: FakeAdapter())
    monkeypatch.setattr(worker_module, "QueueStore", fake_queue_store)
    run_config_path = tmp_path / "run.json"
    run_config = {
        "run_id": _pg_test_run_id(tmp_path),
        "pipeline": "fake",
        "run_dir": str(tmp_path / "run"),
        "artifact_root": str(tmp_path / "artifacts"),
        "stages": {
            "custom": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "queue_store": {"pool": {"enabled": False}},
            }
        },
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    run_worker = worker_module.run_worker
    assert run_worker(
        run_config_path=str(run_config_path),
        pipeline="fake",
        stage_name="custom",
        worker_id="worker-a",
        once=True,
        services=direct_services(),
    ) == 0

    assert store_kwargs[0]["use_pool"] is False
    assert store_kwargs[0]["pool_min_size"] == 2
    assert store_kwargs[0]["pool_max_size"] == 2
    assert store_kwargs[0]["ensure_schema"] is False


def test_worker_queue_store_pool_size_does_not_follow_client_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store_kwargs = []

    class FakeStore:
        def get_run_state(self):
            return {"desired_state": "running"}

        def completion_stats(self):
            return {"run_complete": 0}


        def claim(self, *args):
            return None

        def heartbeat(self, **kwargs):
            return None


    class FakeAdapter:
        def stage_specs(self, run_config):
            return {
                "custom": _stage_spec(
                    name="custom",
                    input_queue="in_q",
                    output_queue=None,
                    num_clients_per_worker=16,
                )
            }


    def fake_queue_store(*args, **kwargs):
        store_kwargs.append(kwargs)
        return FakeStore()

    monkeypatch.setattr(worker_module, "get_adapter", lambda pipeline, run_config: FakeAdapter())
    monkeypatch.setattr(worker_module, "QueueStore", fake_queue_store)
    run_config_path = tmp_path / "run.json"
    run_config = {
        "run_id": _pg_test_run_id(tmp_path),
        "pipeline": "fake",
        "run_dir": str(tmp_path / "run"),
        "artifact_root": str(tmp_path / "artifacts"),
        "stages": {
            "custom": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "queue_store": {"pool": {"max_size": 2}},
            }
        },
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    assert worker_module.run_worker(
        run_config_path=str(run_config_path),
        pipeline="fake",
        stage_name="custom",
        worker_id="worker-a",
        once=True,
        services=direct_services(),
    ) == 0

    assert store_kwargs[0]["pool_max_size"] == 2


def test_worker_rejects_multi_client_parquet_output_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeAdapter:
        def stage_specs(self, run_config):
            return {
                "save": _stage_spec(
                    name="save",
                    input_queue="save_q",
                    output_queue=None,
                    num_clients_per_worker=2,
                    parquet_output=ParquetOutputConfig(enabled=True, output_dir=str(tmp_path / "parquet")),
                )
            }

        def process(self, stage_name, payload, ctx):
            raise AssertionError("worker should reject config before processing")

    monkeypatch.setattr(worker_module, "get_adapter", lambda pipeline, run_config: FakeAdapter())
    monkeypatch.setattr(
        worker_module,
        "QueueStore",
        lambda *args, **kwargs: pytest.fail("QueueStore should not start for invalid parquet clients"),
    )
    run_config_path = tmp_path / "run.json"
    run_config = {
        "run_id": _pg_test_run_id(tmp_path),
        "pipeline": "fake",
        "run_dir": str(tmp_path / "run"),
        "artifact_root": str(tmp_path / "artifacts"),
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(ValueError, match="parquet output stages require 1 client"):
        run_worker = worker_module.run_worker
        run_worker(
            run_config_path=str(run_config_path),
            pipeline="fake",
            stage_name="save",
            worker_id="worker-a",
            once=True,
            services=direct_services(),
        )


def test_queue_store_heartbeat_records_first_hbeat_at_without_real_database(monkeypatch: pytest.MonkeyPatch):
    executed = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params):
            executed.append((query, params))

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    @contextmanager
    def fake_transaction():
        yield FakeConn()

    store_cls = QueueStore
    store = store_cls(run_id="unit_no_db", database_url="postgresql://unit", ensure_schema=False)
    monkeypatch.setattr(store, "transaction", fake_transaction)
    monkeypatch.setattr(store, "_now", lambda: 123.0)

    store.heartbeat(
        worker_id="worker-a",
        pipeline="fake",
        stage="plan",
        pid=123,
        status="working",
        metadata={"processed": 1},
    )

    query, params = executed[0]
    assert "first_hbeat_at" in query
    assert "COALESCE(worker_heartbeats.first_hbeat_at, excluded.first_hbeat_at)" in query
    assert params[6] == 123.0
    assert params[7] == 123.0


def test_entry_sh_resolves_repo_root_for_help():
    env = os.environ.copy()
    env.update(
        {
            "RUN_CONFIG": "unused.json",
            "PIPELINE": "unused",
            "STAGE": "unused",
            "PYTHON_BIN": os.path.relpath(sys.executable, Path.cwd()),
        }
    )
    result = subprocess.run(
        ["bash", "speech_edit/runtime/entry.sh", "--help"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run a distributed-pipeline worker entrypoint" in result.stdout


def test_entry_sh_requires_python_bin():
    env = os.environ.copy()
    env.update(
        {
            "RUN_CONFIG": "unused.json",
            "PIPELINE": "unused",
            "STAGE": "unused",
        }
    )
    env.pop("PYTHON_BIN", None)

    result = subprocess.run(
        ["bash", "speech_edit/runtime/entry.sh", "--help"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "PYTHON_BIN is required" in result.stderr


def test_entry_sh_derives_caches_from_explicit_external_model_root(tmp_path: Path):
    printer = tmp_path / "print_env.sh"
    printer.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f'if [[ "${{1:-}}" == "-c" ]]; then exec {sys.executable} "$@"; fi',
                "printf 'CHECKPOINTS_DIR=%s\\n' \"$CHECKPOINTS_DIR\"",
                "printf 'HF_HOME=%s\\n' \"$HF_HOME\"",
                "printf 'HF_MODULES_CACHE=%s\\n' \"$HF_MODULES_CACHE\"",
                "printf 'HF_HUB_CACHE=%s\\n' \"$HF_HUB_CACHE\"",
                "printf 'TRANSFORMERS_CACHE=%s\\n' \"$TRANSFORMERS_CACHE\"",
                "printf 'TORCH_HOME=%s\\n' \"$TORCH_HOME\"",
                "printf 'HF_HUB_OFFLINE=%s\\n' \"$HF_HUB_OFFLINE\"",
                "printf 'TRANSFORMERS_OFFLINE=%s\\n' \"$TRANSFORMERS_OFFLINE\"",
                "printf 'ARGS=%s\\n' \"$*\"",
            ]
        ),
        encoding="utf-8",
    )
    printer.chmod(0o755)
    (tmp_path / "run.json").write_text(json.dumps({"pipeline": "text_edit", "run_dir": str(tmp_path / "run")}))
    env = os.environ.copy()
    for key in (
        "CHECKPOINTS_DIR",
        "HF_HOME",
        "HF_MODULES_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        env.pop(key, None)
    env.update(
        {
            "RUN_CONFIG": str(tmp_path / "run.json"),
            "CHECKPOINTS_DIR": str(tmp_path / "models"),
            "PIPELINE": "text_edit",
            "STAGE": "plan",
            "WORKER_ID": "worker-a",
            "PYTHON_BIN": str(printer),
        }
    )

    result = subprocess.run(
        ["bash", "speech_edit/runtime/entry.sh"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    expected_checkpoints = str(tmp_path / "models")
    module_cache = Path(env["ARTIFACT_DIR"]) / "shared/cache/hf_modules"
    assert result.returncode == 0, result.stderr
    assert f"CHECKPOINTS_DIR={expected_checkpoints}" in result.stdout
    assert f"HF_HOME={expected_checkpoints}" in result.stdout
    assert f"HF_MODULES_CACHE={module_cache}" in result.stdout
    assert f"HF_HUB_CACHE={expected_checkpoints}" in result.stdout
    assert f"TRANSFORMERS_CACHE={expected_checkpoints}" in result.stdout
    assert f"TORCH_HOME={expected_checkpoints}/torch" in result.stdout
    assert "HF_HUB_OFFLINE=1" in result.stdout
    assert "TRANSFORMERS_OFFLINE=1" in result.stdout


def test_entry_sh_preserves_explicit_model_cache_overrides(tmp_path: Path):
    printer = tmp_path / "print_env.sh"
    printer.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f'if [[ "${{1:-}}" == "-c" ]]; then exec {sys.executable} "$@"; fi',
                "printf 'CHECKPOINTS_DIR=%s\\n' \"$CHECKPOINTS_DIR\"",
                "printf 'HF_HOME=%s\\n' \"$HF_HOME\"",
                "printf 'HF_MODULES_CACHE=%s\\n' \"$HF_MODULES_CACHE\"",
                "printf 'HF_HUB_CACHE=%s\\n' \"$HF_HUB_CACHE\"",
                "printf 'TRANSFORMERS_CACHE=%s\\n' \"$TRANSFORMERS_CACHE\"",
                "printf 'TORCH_HOME=%s\\n' \"$TORCH_HOME\"",
                "printf 'HF_HUB_OFFLINE=%s\\n' \"$HF_HUB_OFFLINE\"",
                "printf 'TRANSFORMERS_OFFLINE=%s\\n' \"$TRANSFORMERS_OFFLINE\"",
            ]
        ),
        encoding="utf-8",
    )
    printer.chmod(0o755)
    (tmp_path / "run.json").write_text(json.dumps({"pipeline": "text_edit", "run_dir": str(tmp_path / "run")}))
    env = os.environ.copy()
    env.update(
        {
            "RUN_CONFIG": str(tmp_path / "run.json"),
            "PIPELINE": "text_edit",
            "STAGE": "plan",
            "WORKER_ID": "worker-a",
            "PYTHON_BIN": str(printer),
            "CHECKPOINTS_DIR": str(tmp_path / "ckpt"),
            "HF_HOME": str(tmp_path / "hf"),
            "HF_MODULES_CACHE": str(tmp_path / "hf_modules"),
            "HF_HUB_CACHE": str(tmp_path / "hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "transformers"),
            "TORCH_HOME": str(tmp_path / "torch"),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
        }
    )

    result = subprocess.run(
        ["bash", "speech_edit/runtime/entry.sh"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"CHECKPOINTS_DIR={tmp_path / 'ckpt'}" in result.stdout
    assert f"HF_HOME={tmp_path / 'hf'}" in result.stdout
    assert f"HF_MODULES_CACHE={tmp_path / 'hf_modules'}" in result.stdout
    assert f"HF_HUB_CACHE={tmp_path / 'hub'}" in result.stdout
    assert f"TRANSFORMERS_CACHE={tmp_path / 'transformers'}" in result.stdout
    assert f"TORCH_HOME={tmp_path / 'torch'}" in result.stdout
    assert "HF_HUB_OFFLINE=0" in result.stdout
    assert "TRANSFORMERS_OFFLINE=0" in result.stdout


def test_queue_store_decodes_postgres_sql_ascii_bytes():
    assert QueueStore._text(b"plan") == "plan"
    assert QueueStore._text(memoryview(b"submitted")) == "submitted"


def test_submit_run_rejects_unsafe_or_long_run_id():
    assert submit_run_module.validate_run_id("b168d7") == "b168d7"
    with pytest.raises(ValueError, match="run_id"):
        submit_run_module.validate_run_id("this-run-id-is-too-long")
    with pytest.raises(ValueError, match="run_id"):
        submit_run_module.validate_run_id("-bad")


def test_dist_pipeline_startup_paths_and_configs_have_no_git_mutation_commands():
    paths = [
        Path("speech_edit/runtime/entry.sh"),
        *Path("speech_edit/runtime").glob("*.py"),
        *Path("configs").glob("**/*.yaml"),
    ]
    banned = ("git pull", "git fetch", "git checkout", "git reset")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{needle!r} found in {path}"


def test_registry_loads_adapter_from_run_config(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("dist_pipeline_test_adapter_mod")

    class LoadedAdapter:
        pass

    module.LoadedAdapter = LoadedAdapter
    monkeypatch.setitem(sys.modules, module.__name__, module)

    adapter = registry_module.get_adapter(
        "fake",
        {"adapter_class": "dist_pipeline_test_adapter_mod:LoadedAdapter"},
    )

    assert isinstance(adapter, LoadedAdapter)

    with pytest.raises(ValueError, match="adapter_class"):
        registry_module.get_adapter("fake", {})


def test_dist_pipeline_has_no_pipeline_specific_imports():
    runtime_root = Path("speech_edit/runtime")
    banned = (
        "speech_edit.pipelines",
        "tools.data-saver",
        "TextEdit",
        "EmotionStyle",
        "Prosody",
        "text_edit",
        "emotion_style_edit",
        "prosody_edit",
    )
    for path in runtime_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{needle!r} leaked into {path}"


def test_local_runner_builds_generic_worker_specs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeAdapter:
        def stage_specs(self, run_config):
            return {
                "plan": _stage_spec("plan", "q1", "q2", max_workers=2),
                "save": _stage_spec(
                    "save",
                    "q2",
                    None,
                    max_workers=2,
                    parquet_output=ParquetOutputConfig(
                        enabled=True,
                        output_dir=str(tmp_path / "parquet"),
                    ),
                ),
            }

    monkeypatch.setattr(local_runner_module, "get_adapter", lambda pipeline, run_config: FakeAdapter())
    run_config = {
        "pipeline": "fake",
        "local_workers": {
            "plan": {},
            "save": {"gpu_ids": [0, 1]},
        },
    }

    specs = local_runner_module.build_local_worker_specs(run_config)

    assert [(spec.stage, spec.worker_index) for spec in specs] == [
        ("plan", 0),
        ("plan", 1),
        ("save", 2),
        ("save", 3),
    ]
    assert [spec.gpu_id for spec in specs if spec.stage == "save"] == ["0", "1"]
    assert all(spec.python_bin == DEFAULT_STAGE_WORKER_PYTHON_BIN for spec in specs)


def test_adapter_context_uses_injected_direct_services():
    ctx = worker_module.AdapterContext(
        run_config={},
        worker_id="worker-a",
        stage=_stage_spec(
            "synth",
            "in",
            "out",
            services={
                "f5": {"url": "http://stage-f5"},
                "aligner": {"url": "http://stage-aligner"},
            },
        ),
        services=direct_services({
            "f5": "http://stage-f5",
            "aligner": "http://stage-aligner",
        }),
    )

    assert ctx.services.as_urls() == {
        "f5": "http://stage-f5",
        "aligner": "http://stage-aligner",
    }


def test_fake_pipeline_runs_eval_to_explicit_save_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeAdapter:
        def stage_specs(self, run_config):
            return {
                "sample": _stage_spec("sample", "source", "eval_q"),
                "eval": _stage_spec("eval", "eval_q", "save_q"),
                "save": _stage_spec(
                    "save",
                    "save_q",
                    None,
                    parquet_output=ParquetOutputConfig(
                        enabled=True,
                        output_dir=str(Path(run_config["run_dir"]) / "parquet"),
                        flush_rows=10,
                        flush_interval_sec=999,
                    ),
                ),
            }

        def process(self, stage_name, payload, ctx):
            if stage_name == "sample":
                return StageResult(
                    outputs=[
                        OutputItem(
                            queue_name="eval_q",
                            item_key="eval-1",
                            payload={"value": payload["value"]},
                        )
                    ]
                )
            if stage_name == "eval":
                return StageResult(
                    outputs=[
                        OutputItem(
                            queue_name="save_q",
                            item_key="save-1",
                            payload={"value": payload["value"], "evaluated": True},
                        )
                    ]
                )
            if stage_name == "save":
                assert payload == {"value": 7, "evaluated": True}
                return StageResult(
                    parquet_rows=[
                        {
                            "audio_id": "row-7",
                            "from_dataset": "unit",
                            "sample_rate": 24000,
                        }
                    ]
                )
            raise AssertionError(stage_name)

    monkeypatch.setattr(worker_module, "get_adapter", lambda pipeline, run_config: FakeAdapter())
    run_config_path = tmp_path / "run.json"
    run_config = {
        "run_id": _pg_test_run_id(tmp_path),
        "pipeline": "fake",
        "run_dir": str(tmp_path / "run"),
        "artifact_root": str(tmp_path / "artifacts"),
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    store = QueueStore(run_id=run_config["run_id"])
    store.enqueue("source", "item-1", payload_json={"value": 7})

    for stage in ("sample", "eval", "save"):
        worker_module.run_worker(
            run_config_path=str(run_config_path),
            pipeline="fake",
            stage_name=stage,
            worker_id=f"{stage}-worker",
            once=True,
            services=direct_services(),
        )

    stats = store.completion_stats()
    assert stats["run_complete"] == 1
    parts = list((Path(run_config["run_dir"]) / "parquet" / "worker_save-worker").glob("*.parquet"))
    assert len(parts) == 1
    assert pq.read_table(parts[0]).column("audio_id").to_pylist() == ["row-7"]


def test_artifact_store_detects_checksum_mismatch(tmp_path: Path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    ref = artifacts.write_json("payloads", "item-1", {"ok": True})
    path = artifacts.resolve(ref)
    path.write_text('{"ok": false}', encoding="utf-8")

    try:
        artifacts.read_json(ref)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch")


def test_external_database_run_creation_and_duplicate_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), source)
    config = tmp_path / "input.json"
    config.write_text(json.dumps({
        "pipeline": "example", "source": {"inputs": [{"path": str(source)}], "rows_per_item": 1},
    }))
    run_id = _pg_test_run_id(tmp_path)[-16:]
    path = submit_run_module.prepare_run(run_config_path=str(config), run_id=run_id, run_dir=str(tmp_path / "run"))
    assert path.is_file()
    store = QueueStore(run_id=run_id)
    assert store.queue_stats()["source_items"]["ready"] == 2
    assert store.get_run_state()["desired_state"] == "running"
    with pytest.raises(ValueError, match="already exists"):
        submit_run_module.prepare_run(run_config_path=str(config), run_id=run_id, run_dir=str(tmp_path / "other"))
    assert store.get_run_state()["desired_state"] == "running"
    assert store.queue_stats()["source_items"]["ready"] == 2


def test_stop_and_resume_preserves_pending_work(tmp_path):
    store = QueueStore(run_id=_pg_test_run_id(tmp_path))
    store.enqueue("q", "item", payload_json={"value": 1})
    store.set_run_state(desired_state="stopped")
    assert worker_module._should_stop_before_claim(store)
    assert store.queue_stats()["q"]["ready"] == 1
    store.set_run_state(desired_state="running")
    assert not worker_module._should_stop_before_claim(store)
    item = store.claim("q", "worker", 60)
    store.ack_and_enqueue(item.id, worker_id="worker", outputs=[], payload_refs=[])
    assert worker_module._should_stop_before_claim(store)


@pytest.mark.parametrize("config_name", [
    "text_edit/emilia_en_multiop", "emotion_style_edit/emilia_en", "prosody_edit/emilia_en", "pause_edit/emilia_en",
])
def test_formal_pipeline_topology_routes_to_parquet_on_postgres(tmp_path, monkeypatch, config_name):
    """Exercise real queues and parquet on each formal DAG with deterministic stage payloads."""
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    config = config_module.load_run_config("pipeline/" + config_name)
    config.update(run_id=_pg_test_run_id(tmp_path), run_dir=str(tmp_path / "run"), artifact_root=str(tmp_path / "payloads"))
    for stage in config["stages"].values():
        stage["num_clients_per_worker"] = 1
        if stage.get("parquet_output", {}).get("enabled"):
            stage["parquet_output"].update(output_dir=str(tmp_path / "run/parquet"), flush_rows=1)
    specs = config_module.stage_specs_from_config(config)
    class Adapter:
        def stage_specs(self, run_config):
            return specs
        def process(self, stage_name, payload, ctx):
            path = payload["path"] + [stage_name]
            if ctx.stage.parquet_output.enabled:
                return StageResult(parquet_rows=[{"audio_id": "row", "from_dataset": "test", "path": json.dumps(path)}])
            return StageResult(outputs=[OutputItem(queue_name=queue, item_key="/".join(path), payload={"path": path})
                for queue, binding in config["topology"]["bindings"].items() if stage_name in binding.get("producers", [])])
    monkeypatch.setattr(worker_module, "get_adapter", lambda *args: Adapter())
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config))
    store = QueueStore(run_id=config["run_id"])
    store.enqueue(config["source"]["queue_name"], "input", payload_json={"path": []})
    # Follow available input work, not stage ordering or concrete stage names.
    for _ in range(len(specs) * 2):
        if store.completion_stats()["run_complete"]:
            break
        progress = False
        for name, spec in specs.items():
            if store.queue_stats().get(spec.input_queue, {}).get("ready", 0):
                worker_module.run_worker(run_config_path=str(config_path), pipeline=config["pipeline"], stage_name=name,
                                         worker_id=name, once=True, services=direct_services())
                progress = True
        assert progress
    assert store.completion_stats()["run_complete"] == 1
    parts = list((tmp_path / "run/parquet").rglob("*.parquet"))
    assert len(parts) == (2 if config_name.startswith("emotion_style_edit/") else 1)
    paths = [json.loads(pq.read_table(part).to_pylist()[0]["path"]) for part in parts]
    assert set().union(*(set(path) for path in paths)) == set(specs)
    for path in paths:
        assert specs[path[-1]].parquet_output.enabled
        for first, second in zip(path, path[1:]):
            assert first in config["topology"]["bindings"][specs[second].input_queue]["producers"]
