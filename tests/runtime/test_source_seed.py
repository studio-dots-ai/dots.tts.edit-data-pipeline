from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speech_edit.runtime.config import source_config
from speech_edit.runtime.source_seed import build_source_index
from speech_edit.runtime.source_seed import (
    _limit_items_per_input,
    _shuffle_source_items,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_limit_items_per_input_keeps_each_source_independently():
    items = [
        {"item_key": "a0", "payload": {"source_entry_index": 0}},
        {"item_key": "a1", "payload": {"source_entry_index": 0}},
        {"item_key": "a2", "payload": {"source_entry_index": 0}},
        {"item_key": "b0", "payload": {"source_entry_index": 1}},
        {"item_key": "b1", "payload": {"source_entry_index": 1}},
        {"item_key": "b2", "payload": {"source_entry_index": 1}},
    ]
    messages: list[str] = []

    limited = _limit_items_per_input(items, max_items_per_input=2, progress=messages.append)

    assert [item["item_key"] for item in limited] == ["a0", "a1", "b0", "b1"]
    assert messages == ["source_seed_limited_per_input items=4/6 max_items_per_input=2"]


def test_shuffle_source_items_is_deterministic_before_limiting():
    items = [{"item_key": f"item-{index}", "payload": {"source_entry_index": 0}} for index in range(8)]

    first = _shuffle_source_items(
        items,
        shuffle_items=True,
        shuffle_seed=20260623,
        progress=lambda _: None,
    )
    second = _shuffle_source_items(
        items,
        shuffle_items=True,
        shuffle_seed=20260623,
        progress=lambda _: None,
    )

    assert [item["item_key"] for item in first] == [item["item_key"] for item in second]
    assert [item["item_key"] for item in first] != [item["item_key"] for item in items]


def test_source_config_rejects_removed_global_max_items():
    try:
        source_config({"source": {"max_items": 2}})
    except ValueError as exc:
        assert "source.max_items was removed" in str(exc)
    else:
        raise AssertionError("source.max_items should be rejected")


def test_default_index_cache_is_shared_and_uses_effective_root(tmp_path, monkeypatch):
    from speech_edit.runtime.source_seed import _default_source_index_cache_path

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "different-shell-root"))
    root = tmp_path / "artifacts"
    kwargs = {"sources": ["/datasets/source.list"], "rows_per_item": 64}
    first = _default_source_index_cache_path(
        run_config={"project_root": str(tmp_path / "checkout-a"), "artifact_dir": str(root)}, **kwargs
    )
    second = _default_source_index_cache_path(
        run_config={"project_root": str(tmp_path / "checkout-b"), "artifact_dir": str(root)}, **kwargs
    )
    assert first == second
    assert first.parent == root / "shared/source_indexes"
    assert not root.exists()


def test_source_index_concurrent_writers_publish_complete_files(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from speech_edit.runtime.source_seed import _read_source_index, _write_source_index

    target = tmp_path / "shared.jsonl"
    batches = [[{"producer": name, "row": row} for row in range(10)] for name in ("a", "b")]
    ready = Barrier(2)
    replace = Path.replace
    temporary_paths = []

    def concurrent_replace(path, destination):
        temporary_paths.append(path)
        ready.wait(timeout=10)
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", concurrent_replace)
    def write(items):
        _write_source_index(target, sources=["input"], rows_per_item=1, items=items)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, batches))
    assert len(set(temporary_paths)) == 2
    assert _read_source_index(target, sources=["input"], rows_per_item=1) in batches
    assert not list(tmp_path.glob("*.tmp"))


def test_source_index_failed_write_preserves_previous_cache(tmp_path):
    import pytest
    from speech_edit.runtime.source_seed import _read_source_index, _write_source_index

    target = tmp_path / "index.jsonl"
    previous = [{"row": 1}]
    _write_source_index(target, sources=["input"], rows_per_item=1, items=previous)
    with pytest.raises(TypeError):
        _write_source_index(target, sources=["input"], rows_per_item=1, items=[{"row": object()}])
    assert _read_source_index(target, sources=["input"], rows_per_item=1) == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_configured_source_cache_override_uses_saved_artifact_root(tmp_path, monkeypatch):
    from speech_edit.runtime import source_seed

    root = tmp_path / "effective-root"
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "other-root"))
    class Queue:
        def __init__(self, **kwargs):
            pass
        def enqueue_json_many(self, *args, **kwargs):
            return 1
    monkeypatch.setattr(source_seed, "QueueStore", Queue)
    monkeypatch.setattr(source_seed, "build_source_index", lambda **kwargs: [{"item_key": "row", "payload": {}}])
    inserted = source_seed.seed_source_from_run_config(run_config={
        "run_id": "unit", "artifact_dir": str(root),
        "source": {"inputs": ["input.parquet"], "source_index_cache": "custom/index.jsonl"},
    })
    assert inserted == 1
    assert (root / "shared/custom/index.jsonl").is_file()
    assert not (tmp_path / "other-root").exists()


def test_source_cache_rejects_external_override_before_connecting(tmp_path, monkeypatch):
    import pytest
    from speech_edit.runtime import source_seed

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "allowed"))
    monkeypatch.setattr(source_seed, "QueueStore", lambda **kwargs: pytest.fail("Must reject before database access"))
    with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
        source_seed.seed_parquet_queue(run_id="unit", queue_name="source_items", sources=["input"],
                                      source_index_cache=str(tmp_path / "outside.jsonl"))
    assert not (tmp_path / "outside.jsonl").exists()


def test_source_index_preserves_multiple_input_provenance(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_parquet(first, [{"audio_id": "a0"}, {"audio_id": "a1"}])
    _write_parquet(second, [{"audio_id": "b0"}])
    items = build_source_index(
        sources=[{"path": str(first)}, {"path": str(second)}],
        rows_per_item=1, progress=lambda _: None, metadata_workers=2,
    )
    assert [item["item_key"] for item in items] == [
        f"{first}:0:1", f"{first}:1:2", f"{second}:0:1",
    ]
    assert [item["payload"]["source_entry_index"] for item in items] == [0, 0, 1]
    assert [item["payload"]["source_file_index"] for item in items] == [0, 0, 1]
    assert all(set(item["payload"]) == {
        "parquet_path", "source_entry_index", "source_file_index", "row_start", "row_stop",
    } for item in items)


def test_source_index_rejects_unknown_input_options():
    import pytest

    with pytest.raises(ValueError, match="unsupported fields"):
        build_source_index(sources=[{"path": "/data/input", "unknown_option": True}], rows_per_item=1)


def test_seeding_requires_explicit_source_inputs():
    import pytest
    from speech_edit.runtime.source_seed import seed_source_from_run_config

    assert "inputs" not in source_config({})
    for inputs in (None, [], ""):
        with pytest.raises(ValueError, match="source.inputs must explicitly select"):
            seed_source_from_run_config(run_config={"source": {"inputs": inputs}})
