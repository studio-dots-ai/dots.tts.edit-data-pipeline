from __future__ import annotations

from pathlib import Path

import pytest

from speech_edit.runtime.services import processes as service_processes
from speech_edit.runtime.services.resolver import (
    resolve_service_graph,
)
from speech_edit.runtime.services.single_node_workers import (
    materialize_worker_service_graph,
)


class FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pid = 12345

    def poll(self):
        return 0


class PendingPopen:
    pid = 23456

    def poll(self):
        return None


def test_service_resolver_handles_dependencies_before_parent():
    graph = resolve_service_graph(
        {
            "parent": {
                "config_key": "parent/defaults",
                "instance_key": "parent/defaults:9001",
                "port": 9001,
                "command": "run-parent",
                "dependencies": {
                    "child": {
                        "config_key": "child/defaults",
                        "instance_key": "child/defaults:9000",
                        "port": 9000,
                        "command": "run-child",
                    }
                },
            }
        }
    )

    commands = graph.command_specs()

    assert [command["command"] for command in commands] == ["run-child", "run-parent"]
    assert graph.direct_services.url("parent") == "http://127.0.0.1:9001"
    assert "child" not in graph.direct_services
    assert graph.roots["parent"].dependencies["child"].url == "http://127.0.0.1:9000"


def test_start_services_writes_logs_and_waits(monkeypatch, tmp_path: Path):
    popen_calls = []
    waited = {}
    snapshots = {}

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return FakePopen(*args, **kwargs)

    def fake_wait(urls, *, timeout_sec, diagnostics=None, on_ready=None):
        waited["urls"] = list(urls)
        waited["timeout_sec"] = timeout_sec
        waited["diagnostic_names"] = {url: diagnostic["name"] for url, diagnostic in (diagnostics or {}).items()}
        for url in urls:
            if on_ready is not None:
                on_ready(url)

    monkeypatch.setattr(service_processes.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(service_processes, "wait_for_urls", fake_wait)
    monkeypatch.setattr(
        service_processes,
        "_write_startup_snapshot",
        lambda _log_dir, name, _commands, **kwargs: snapshots.setdefault(name, kwargs),
    )

    graph = resolve_service_graph(
        {
            "svc": {
                "config_key": "svc/defaults",
                "instance_key": "svc/defaults:9010",
                "port": 9010,
                "command": "python service.py",
                "gpu_ids": "2",
                "env": {"EXTRA": "1"},
                "startup_timeout_sec": 12,
            }
        }
    )

    group = service_processes.start_services(
        graph=graph,
        project_root=tmp_path,
        log_dir=tmp_path / "logs",
    )

    assert len(group.processes) == 1
    assert list(group.log_paths) == ["svc"]
    assert waited == {
        "urls": ["http://127.0.0.1:9010/health"],
        "timeout_sec": 24.0,
        "diagnostic_names": {"http://127.0.0.1:9010/health": "svc"},
    }
    kwargs = popen_calls[0][1]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["shell"] is True
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert kwargs["env"]["EXTRA"] == "1"
    assert [
        event["event"]
        for event in snapshots["service_startup_after"]["timeline"]
    ] == [
        "service_group_starting",
        "service_spawned",
        "service_health_ready",
        "service_group_ready",
    ]


def test_start_services_records_spawn_failure_and_stops_started_processes(
    monkeypatch,
    tmp_path: Path,
):
    first_process = PendingPopen()
    popen_calls = 0
    snapshots = {}
    terminated = []

    def fake_popen(*args, **kwargs):
        nonlocal popen_calls
        del args, kwargs
        popen_calls += 1
        if popen_calls == 2:
            raise RuntimeError("spawn failed")
        return first_process

    monkeypatch.setattr(service_processes.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        service_processes,
        "terminate_processes",
        lambda processes: terminated.extend(processes),
    )
    monkeypatch.setattr(
        service_processes,
        "_write_startup_snapshot",
        lambda _log_dir, name, _commands, **kwargs: snapshots.setdefault(name, kwargs),
    )
    graph = resolve_service_graph(
        {
            "first": {
                "config_key": "first/defaults",
                "instance_key": "first/defaults:9010",
                "port": 9010,
                "command": "run-first",
            },
            "second": {
                "config_key": "second/defaults",
                "instance_key": "second/defaults:9011",
                "port": 9011,
                "command": "run-second",
            },
        }
    )

    with pytest.raises(RuntimeError, match="spawn failed"):
        service_processes.start_services(
            graph=graph,
            project_root=tmp_path,
            log_dir=tmp_path / "logs",
        )

    assert terminated == [first_process]
    assert snapshots["service_startup_failed"]["error"] == "RuntimeError: spawn failed"
    assert [
        event["event"]
        for event in snapshots["service_startup_failed"]["timeline"]
    ] == ["service_group_starting", "service_spawned", "service_group_failed"]


def test_wait_for_urls_timeout_includes_service_diagnostics(tmp_path: Path):
    log_path = tmp_path / "svc.log"
    log_path.write_text("service traceback tail\n", encoding="utf-8")

    with pytest.raises(TimeoutError) as exc_info:
        service_processes.wait_for_urls(
            ["http://127.0.0.1:9/health"],
            timeout_sec=0.0,
            diagnostics={
                "http://127.0.0.1:9/health": {
                    "name": "svc",
                    "process": PendingPopen(),
                    "log_path": log_path,
                }
            },
        )

    message = str(exc_info.value)
    assert "service=svc" in message
    assert "pid=23456" in message
    assert "returncode=None" in message
    assert f"log_path={log_path}" in message
    assert "service traceback tail" in message


def test_resolved_graph_scopes_aliases_by_caller_and_preserves_all_callers():
    shared = {
        "config_key": "aligner/defaults",
        "instance_key": "aligner/defaults:9029",
        "port": 9029,
        "command": "run-aligner",
    }
    graph = resolve_service_graph(
        {
            "consumer_a": {
                "config_key": "consumer-a/defaults",
                "instance_key": "consumer-a/defaults:9001",
                "port": 9001,
                "command": "run-a",
                "dependencies": {"aligner": shared},
            },
            "consumer_b": {
                "config_key": "consumer-b/defaults",
                "instance_key": "consumer-b/defaults:9002",
                "port": 9002,
                "command": "run-b",
                "dependencies": {"aligner": shared, "secondary": shared},
            },
        }
    )

    aligner = graph.instances["aligner/defaults:9029"]
    assert [spec["instance_key"] for spec in graph.command_specs()].count(aligner.instance_key) == 1
    assert {(caller.caller_instance_key, caller.dep_ref_key) for caller in aligner.callers} == {
        ("consumer-a/defaults:9001", "aligner"),
        ("consumer-b/defaults:9002", "aligner"),
        ("consumer-b/defaults:9002", "secondary"),
    }


def test_resolved_graph_rejects_same_caller_alias_pointing_to_different_instances():
    parent = {
        "config_key": "parent/defaults",
        "instance_key": "parent/defaults:9000",
        "port": 9000,
        "command": "run-parent",
    }
    with pytest.raises(ValueError, match="dependency 'aligner'.*multiple instances"):
        resolve_service_graph(
            {
                "first": {
                    **parent,
                    "dependencies": {
                        "aligner": {
                            "config_key": "aligner-a/defaults",
                            "instance_key": "aligner-a/defaults:9029",
                            "port": 9029,
                            "command": "run-aligner-a",
                        }
                    },
                },
                "second": {
                    **parent,
                    "dependencies": {
                        "aligner": {
                            "config_key": "aligner-b/defaults",
                            "instance_key": "aligner-b/defaults:9030",
                            "port": 9030,
                            "command": "run-aligner-b",
                        }
                    },
                },
            }
        )


def test_nested_dependency_is_internal_to_its_caller():
    aligner = {
        "config_key": "aligner/defaults",
        "instance_key": "aligner/defaults:9029",
        "port": 9029,
        "command": "run-aligner",
    }
    graph = resolve_service_graph(
        {
            "wdtw": {
                "config_key": "wdtw/defaults",
                "instance_key": "wdtw/defaults:9032",
                "port": 9032,
                "command": "run-wdtw --aligner-url http://127.0.0.1:9029",
                "dependencies": {"aligner": aligner},
            }
        }
    )

    assert graph.direct_services.as_urls() == {"wdtw": "http://127.0.0.1:9032"}
    assert graph.roots["wdtw"].dependencies["aligner"].url == "http://127.0.0.1:9029"
    assert "--aligner-url http://127.0.0.1:9029" in graph.roots["wdtw"].command


def test_worker_service_graph_offsets_ports_and_preserves_gpu_ids():
    services = {
        "parent": {
            "config_key": "parent/defaults",
            "instance_key": "parent/defaults:9001",
            "port": 9001,
            "gpu_ids": "0",
            "command": "python parent.py --port 9001 --child http://127.0.0.1:9000 --child-port 9000 --state-dir logs/9001",
            "dependencies": {
                "child": {
                    "config_key": "child/defaults",
                    "instance_key": "child/defaults:9000",
                    "port": 9000,
                    "gpu_ids": "1",
                    "command": "python child.py --port 9000",
                }
            },
        }
    }

    materialized = materialize_worker_service_graph(
        services,
        worker_index=2,
        port_stride=100,
    )

    assert materialized["parent"]["port"] == 9201
    assert materialized["parent"]["gpu_ids"] == "0"
    assert "--port 9201" in materialized["parent"]["command"]
    assert "--child-port 9200" in materialized["parent"]["command"]
    assert "logs/9201" in materialized["parent"]["command"]
    assert ":9200" in materialized["parent"]["command"]
    assert materialized["parent"]["dependencies"]["child"]["port"] == 9200
    assert materialized["parent"]["dependencies"]["child"]["gpu_ids"] == "1"
