from __future__ import annotations

import importlib.util
import base64
import json
import os
import re
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from speech_edit.runtime.paths import get_artifact_paths
from speech_edit.runtime.queue_store import QueueStore
from speech_edit.runtime.submit_run import _default_run_dir, prepare_run_config


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "viewers" / "distributed-pipeline-console"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_DIST_PIPELINE_DB_TEST_MARKERS = (
    "QueueStore(",
    "_create_runtime_store(",
)


def _load_dashboard_module():
    for name in ("distributed_pipeline_console_app", "runtime_discovery", "runtime_store"):
        sys.modules.pop(name, None)
    if str(DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "distributed_pipeline_console_app",
        DASHBOARD_DIR / "app.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["distributed_pipeline_console_app"] = module
    spec.loader.exec_module(module)
    return module


def _write_config(run_root: Path, config: dict) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    config = {**config, "run_dir": str(run_root)}
    (run_root / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    timestamp = datetime(2026, 6, 4, tzinfo=timezone.utc).timestamp()
    os.utime(run_root / "run_config.json", (timestamp, timestamp))


def _store_run_id(run_root: Path) -> str:
    return f"dash_{hashlib.sha1(str(run_root).encode('utf-8')).hexdigest()[:12]}"


@pytest.mark.parametrize("direct_root", [False, True])
def test_discover_one_run_loads_only_its_config(tmp_path, monkeypatch, direct_root):
    root = tmp_path / "distributed_pipeline_runs"
    for name in ("wanted", "other_a", "other_b"):
        _write_config(root / "demo" / name, _run_config(name))
    _load_dashboard_module()
    discovery = sys.modules["runtime_discovery"]
    original_load = discovery.load_run_config
    loaded = []

    def load(path):
        loaded.append(path)
        return original_load(path)

    monkeypatch.setattr(discovery, "load_run_config", load)
    data_root = root if direct_root else tmp_path
    run = discovery.discover_run(data_root, "2026-06-04", "wanted")
    assert run.store_run_id == "wanted"
    assert run.path == root / "demo" / "wanted"
    assert loaded == [run.path / "run_config.json"]
    loaded.clear()
    with pytest.raises(FileNotFoundError):
        discovery.discover_run(data_root, "2026-06-03", "wanted")
    assert loaded == []


def _run_config(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "pipeline": "demo",
        "run_dir": ".",
        "worker_heartbeat_timeout_sec": 30,
        "queues": {
            "source_items": {"max_size": 100},
            "plan_queue": {"max_size": 8},
            "save_queue": {"max_size": 4},
        },
        "topology": {"bindings": {
            "source_items": {"consumers": ["plan"]},
            "plan_queue": {"producers": ["plan"], "consumers": ["eval"]},
            "save_queue": {"producers": ["eval"], "consumers": ["save"]},
        }},
        "stages": {
            "plan": {
                "worker_python_bin": "envs/pipe-cpu/.venv/bin/python",
                "max_workers": 2,
            },
            "eval": {
                "worker_python_bin": "envs/pipe-cpu/.venv/bin/python",
                "max_workers": 1,
            },
            "save": {
                "worker_python_bin": "envs/pipe-cpu/.venv/bin/python",
                "max_workers": 1,
                "parquet_output": {"enabled": True, "flush_rows": 2},
            },
        },
    }


def test_submit_default_run_dir_uses_distributed_pipeline_runs(tmp_path: Path) -> None:
    path = _default_run_dir(tmp_path / "project", "text_edit", "abc123")

    assert path.parent.parent.name == "distributed_pipeline_runs"
    assert path.parent.name == "text_edit"
    assert path.name == "abc123"
    assert path.is_relative_to(get_artifact_paths(tmp_path / "project").worktree)


def test_prepare_run_config_uses_distributed_default_dir_when_unspecified(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            {
                "project_root": str(project_root),
                "pipeline": "text_edit",
                "adapter_class": "demo.Adapter",
                "source": {"kind": "empty"},
                "stages": {},
            }
        ),
        encoding="utf-8",
    )

    run_config, effective_config_path = prepare_run_config(
        run_config_path=str(run_config_path),
        run_id="abc123",
    )

    assert run_config["run_dir"].startswith(
        str(get_artifact_paths(project_root).worktree / "distributed_pipeline_runs")
    )
    assert run_config["run_dir"].endswith("/text_edit/abc123")
    assert effective_config_path == Path(run_config["run_dir"]) / "run_config.yaml"
    _load_dashboard_module()
    runs = list(sys.modules["runtime_discovery"].iter_runs(
        get_artifact_paths(project_root).worktree / "distributed_pipeline_runs"
    ))
    assert any(run.config_path == effective_config_path for run in runs)



def test_runtime_discovery_scans_direct_distributed_pipeline_root(tmp_path: Path) -> None:
    _load_dashboard_module()
    runtime_discovery = sys.modules["runtime_discovery"]
    root = tmp_path / "distributed_pipeline_runs"
    run_root = root / "prosody_edit" / "emilia_zh_debug"
    _write_config(run_root, {**_run_config("direct_debug"), "pipeline": "prosody_edit"})

    runs = list(runtime_discovery.iter_runs(root, start_date="2026-06-04"))

    assert [(run.name, run.path) for run in runs] == [("emilia_zh_debug", run_root)]
    assert DATE_RE.match(runs[0].date)


def test_runtime_discovery_scans_nested_distributed_pipeline_root(tmp_path: Path) -> None:
    _load_dashboard_module()
    runtime_discovery = sys.modules["runtime_discovery"]
    root = tmp_path / "data_samples"
    run_root = root / "distributed_pipeline_runs" / "prosody_edit" / "emilia_zh_debug"
    _write_config(run_root, {**_run_config("nested_debug"), "pipeline": "prosody_edit"})

    runs = list(runtime_discovery.iter_runs(root, start_date="2026-06-04"))

    assert [(run.name, run.path) for run in runs] == [("emilia_zh_debug", run_root)]
    assert DATE_RE.match(runs[0].date)


def test_dashboard_orders_queues_from_declared_stage_specs() -> None:
    _load_dashboard_module()
    runtime_store = sys.modules["runtime_store"]
    run_config = {
        "queues": {
            "q_ingress": {"max_size": 100},
            "q_middle": {"max_size": 10},
        },
        "topology": {"bindings": {
            "q_ingress": {"consumers": ["alpha"]},
            "q_middle": {"producers": ["alpha"], "consumers": ["beta"]},
            "q_terminal": {"producers": ["beta"], "consumers": ["gamma"]},
        }},
        "stages": {
            "alpha": {"worker_python_bin": "python"},
            "beta": {
                "worker_python_bin": "python",
                "queue_max_size": 20,
            },
            "gamma": {"worker_python_bin": "python"},
        },
    }
    queues = [
        {"queue_name": "z_unmatched", "pending": 1},
        {"queue_name": "q_terminal", "pending": 1},
        {"queue_name": "q_ingress", "pending": 1},
        {"queue_name": "q_middle", "pending": 1},
    ]

    ordered = runtime_store._ordered_queues(run_config, queues)
    assert [item["queue_name"] for item in ordered] == [
        "q_ingress",
        "q_middle",
        "q_terminal",
        "z_unmatched",
    ]
    assert ordered[0]["queue_role"] == "source"
    assert ordered[0]["max_size"] == 100
    assert ordered[1]["max_size"] == 10
    assert ordered[2]["max_size"] == 20
    assert ordered[3]["max_size"] is None


def test_dashboard_orders_split_topology_from_bindings() -> None:
    _load_dashboard_module()
    runtime_store = sys.modules["runtime_store"]
    run_config = {
        "queues": {
            "source_items": {"max_size": 100},
            "sample_to_plan": {"max_size": 100},
            "plan_to_index": {"max_size": 100},
            "plan_to_ted": {"max_size": 100},
            "synth_to_eval": {"max_size": 100},
            "eval_to_save": {"max_size": 100},
        },
        "stages": {
            "sample": {"worker_python_bin": "python"},
            "plan": {"worker_python_bin": "python"},
            "synth_index": {"worker_python_bin": "python"},
            "synth_ted": {"worker_python_bin": "python"},
            "eval": {"worker_python_bin": "python"},
            "save": {"worker_python_bin": "python"},
        },
        "topology": {
            "bindings": {
                "source_items": {"consumers": ["sample"]},
                "sample_to_plan": {"producers": ["sample"], "consumers": ["plan"]},
                "plan_to_index": {"producers": ["plan"], "consumers": ["synth_index"]},
                "plan_to_ted": {"producers": ["plan"], "consumers": ["synth_ted"]},
                "synth_to_eval": {
                    "producers": ["synth_index", "synth_ted"],
                    "consumers": ["eval"],
                },
                "eval_to_save": {"producers": ["eval"], "consumers": ["save"]},
            },
        },
    }
    queues = [
        {"queue_name": "plan_to_ted", "pending": 1},
        {"queue_name": "synth_to_eval", "pending": 1},
        {"queue_name": "plan_to_index", "pending": 1},
        {"queue_name": "sample_to_plan", "pending": 1},
    ]

    ordered = runtime_store._ordered_queues(run_config, queues)
    by_name = {item["queue_name"]: item for item in ordered}

    assert [item["queue_name"] for item in ordered] == [
        "source_items",
        "sample_to_plan",
        "plan_to_index",
        "plan_to_ted",
        "synth_to_eval",
        "eval_to_save",
    ]
    assert by_name["source_items"]["queue_role"] == "source"
    assert by_name["sample_to_plan"]["queue_role"] == "intermediate"
    assert by_name["plan_to_index"]["queue_role"] == "intermediate"
    assert by_name["plan_to_ted"]["queue_role"] == "intermediate"
    assert by_name["synth_to_eval"]["queue_role"] == "intermediate"
    assert by_name["eval_to_save"]["queue_role"] == "intermediate"
    assert by_name["plan_to_index"]["producer_stages"] == ["plan"]
    assert by_name["plan_to_ted"]["producer_stages"] == ["plan"]
    assert by_name["synth_to_eval"]["producer_stages"] == ["synth_index", "synth_ted"]


def test_dashboard_rejects_topology_rejected_by_runtime() -> None:
    _load_dashboard_module()
    runtime_store = sys.modules["runtime_store"]
    run_config = {
        "queues": {"source": {}, "next": {}},
        "stages": {"sample": {}, "plan": {}},
        "topology": {
            "bindings": {
                "source": {"consumers": ["sample"]},
                "next": {"producer": "sample", "consumers": ["plan"]},
            }
        },
    }

    with pytest.raises(ValueError, match=r"producers \(a list\)"):
        runtime_store._ordered_queues(run_config, [])


def test_dashboard_sample_detail_page_and_parquet_row_api(tmp_path: Path) -> None:
    root = tmp_path / "data_samples"
    run_root = root / "demo" / "runtime_preview"
    run_id = _store_run_id(run_root)
    _write_config(run_root, _run_config(run_id))
    parquet_path = run_root / "parquet" / "worker_save" / "part-000001.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "audio_id": "row-1",
                "from_dataset": "test",
                "audios": json.dumps(
                    {"src": base64.b64encode(b"fLaCfake").decode("ascii")}
                ),
                "duration": json.dumps({"src": 1.0}),
                "segments": json.dumps({"src": {"label": "hello"}}),
                "instructions": json.dumps({"st": "replace greeting with hello"}),
            }
        ]
    ).to_parquet(parquet_path)

    app_module = _load_dashboard_module()
    client = TestClient(base_url="http://127.0.0.1", app=app_module.create_app(data_root=root))

    page = client.get("/sample.html")
    assert page.status_code == 200

    row = client.get(
        "/api/runs/2026-06-04/runtime_preview/parquet-row",
        params={"file": "parquet/worker_save/part-000001.parquet", "row_index": 0},
    )
    assert row.status_code == 200
    sample = row.json()
    assert sample["audio_id"] == "row-1"
    assert sample["audio_keys"] == ["src"]
    assert sample["instruction"] == "replace greeting with hello"
    assert "audios" not in sample["metadata"]

    audio = client.get(
        "/api/runs/2026-06-04/runtime_preview/parquet-audio",
        params={"file": "parquet/worker_save/part-000001.parquet", "row_index": 0, "key": "src"},
    )
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/flac")
    assert audio.content == b"fLaCfake"


def test_dashboard_preview_sample_falls_back_to_compact_instruction_json(tmp_path: Path) -> None:
    root = tmp_path / "data_samples"
    run_root = root / "demo" / "runtime_preview_instruction"
    run_id = _store_run_id(run_root)
    _write_config(run_root, _run_config(run_id))
    parquet_path = run_root / "parquet" / "worker_save" / "part-000001.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "audio_id": "row-1",
                "instructions": json.dumps({"source": "A", "target": "B"}),
            }
        ]
    ).to_parquet(parquet_path)

    app_module = _load_dashboard_module()
    client = TestClient(base_url="http://127.0.0.1", app=app_module.create_app(data_root=root))

    response = client.get(
        "/api/runs/2026-06-04/runtime_preview_instruction/parquet-preview",
        params={"file": "parquet/worker_save/part-000001.parquet", "mode": "top", "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["samples"][0]["instruction"] == '{"source":"A","target":"B"}'


def test_distributed_pipeline_console_has_no_concrete_pipeline_imports() -> None:
    blocked = [
        "speech_edit.pipelines",
        "text_edit",
        "emotion_style",
        "prosody",
    ]
    for path in DASHBOARD_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in blocked:
            assert token not in text, f"{path} contains concrete pipeline token {token!r}"


@pytest.mark.parametrize("headers,expected", [
    ({"Origin": "http://127.0.0.1"}, 200),
    ({}, 200),
    ({"Origin": "https://untrusted.example"}, 403),
    ({"Origin": "null"}, 403),
    ({"Origin": "http://127.0.0.1:9000"}, 403),
    ({"Sec-Fetch-Site": "cross-site"}, 403),
])
@pytest.mark.parametrize("operation", ["stop", "resume"])
def test_console_write_origin_is_checked_before_run_access(tmp_path, monkeypatch, headers, expected, operation):
    module = _load_dashboard_module()
    app = module.create_app(data_root=tmp_path)
    accessed = []

    def resolve_run(*args):
        accessed.append(True)
        raise module.HTTPException(status_code=200, detail="same-origin request reached handler")

    monkeypatch.setattr(module, "_resolve_run", resolve_run)
    client = TestClient(app, base_url="http://127.0.0.1")
    result = client.post(f"/api/runs/2026-01-01/example/{operation}", headers=headers)
    assert result.status_code == expected
    assert bool(accessed) == (expected == 200)


def test_console_rejects_untrusted_host(tmp_path):
    module = _load_dashboard_module()
    client = TestClient(module.create_app(data_root=tmp_path), base_url="http://untrusted.example")
    assert client.get("/").status_code == 400


def test_console_stop_resume_and_live_snapshot(tmp_path):
    if not os.environ.get("DIST_PIPELINE_DATABASE_URL"):
        pytest.skip("PostgreSQL required")
    root = tmp_path / "runs"
    run_root = root / "demo" / "runtime_demo"
    run_id = _store_run_id(run_root)
    _write_config(run_root, _run_config(run_id))
    store = QueueStore(run_id=run_id)
    store.set_run_state(desired_state="running")
    store.enqueue("source_items", "input", payload_json={"private": "not exposed"})
    store.heartbeat(worker_id="plan-1", pipeline="demo", stage="plan", pid=1, status="working")
    module = _load_dashboard_module()
    client = TestClient(module.create_app(root), base_url="http://127.0.0.1")
    url = "/api/runs/2026-06-04/runtime_demo"
    result = client.get(url + "/dashboard")
    assert result.status_code == 200
    data = result.json()
    assert data["status"] == "running"
    assert data["stages"][0]["active_workers"] == 1
    assert "private" not in result.text
    assert client.post(url + "/stop").json()["desired_state"] == "stopped"
    assert client.get(url + "/dashboard").json()["status"] == "stopping"
    store.heartbeat(worker_id="plan-1", pipeline="demo", stage="plan", pid=1, status="stopped")
    assert client.get(url + "/dashboard").json()["status"] == "stopped"
    assert client.post(url + "/resume").json()["desired_state"] == "running"
    assert client.get(url + "/dashboard").json()["status"] == "waiting_for_workers"
    assert store.queue_stats()["source_items"]["ready"] == 1


def test_console_parquet_preview_rejects_outside_run(tmp_path):
    module = _load_dashboard_module()
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = tmp_path / "outside.parquet"
    pd.DataFrame([{"audio_id": "outside"}]).to_parquet(outside)
    run = SimpleNamespace(run_dir=run_root, path=run_root)
    with pytest.raises(ValueError, match="escapes"):
        module._resolve_run_parquet_path(run, str(outside))
    with pytest.raises(ValueError, match="escapes"):
        module._resolve_run_parquet_path(run, "../outside.parquet")


def test_console_database_failure_is_not_a_running_run(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    _write_config(root / "demo" / "runtime_demo", _run_config("missing"))
    module = _load_dashboard_module()
    monkeypatch.delenv("DIST_PIPELINE_DATABASE_URL", raising=False)
    client = TestClient(module.create_app(root), base_url="http://127.0.0.1")
    assert client.get("/api/runs").json()["runs"][0]["status"] == "unavailable"
    response = client.get("/api/runs/2026-06-04/runtime_demo/dashboard")
    assert response.status_code == 503


def test_console_created_run_is_discoverable(tmp_path, monkeypatch):
    if not os.environ.get("DIST_PIPELINE_DATABASE_URL"):
        pytest.skip("PostgreSQL required")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    root = tmp_path / "runs"
    source = tmp_path / "source.parquet"
    pd.DataFrame([{"value": 1}]).to_parquet(source)
    config = _run_config("unused")
    config["source"] = {"inputs": [{"path": str(source)}], "rows_per_item": 1}
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(config))
    module = _load_dashboard_module()
    client = TestClient(module.create_app(root), base_url="http://127.0.0.1")
    run_id = _store_run_id(tmp_path)[-16:]
    response = client.post("/api/runs/submit", json={"run_config": str(config_path), "run_id": run_id})
    assert response.status_code == 200, response.text
    assert Path(response.json()["config_path"]).parent == root / "demo" / run_id
    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["store_run_id"] == run_id
    assert runs[0]["status"] == "waiting_for_workers"
    assert client.post("/api/runs/submit", json={"run_config": str(config_path), "run_id": run_id}).status_code == 409


@pytest.mark.parametrize("failed", [False, True])
def test_console_reports_terminal_queue_states(tmp_path, failed):
    if not os.environ.get("DIST_PIPELINE_DATABASE_URL"):
        pytest.skip("PostgreSQL required")
    root = tmp_path / "runs"
    run_root = root / "demo" / "terminal"
    run_id = _store_run_id(run_root)
    _write_config(run_root, _run_config(run_id))
    store = QueueStore(run_id=run_id)
    store.set_run_state(desired_state="running")
    store.enqueue("source_items", "row", payload_json={"value": 1}, max_attempts=1)
    item = store.claim("source_items", "worker", 60)
    if failed:
        store.fail_item(item.id, worker_id="worker", error="intentional failure")
    else:
        store.ack_and_enqueue(item.id, worker_id="worker", outputs=[], payload_refs=[])
    module = _load_dashboard_module()
    client = TestClient(module.create_app(root), base_url="http://127.0.0.1")
    response = client.get("/api/runs/2026-06-04/terminal/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == ("failed" if failed else "completed")
    assert data["completion"]["run_complete"] == int(not failed)
    queue = next(q for q in data["queues"] if q["queue_name"] == "source_items")
    assert queue["counts"] == {("dead" if failed else "done"): 1}
    assert queue["total"] == 1 and queue["pending"] == 0
    store.close()
