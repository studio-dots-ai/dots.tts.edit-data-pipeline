"""Shared local service process startup and shutdown helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, Mapping, Sequence
from urllib.request import Request, urlopen

from speech_edit.runtime.services.resolver import ResolvedServiceGraph


def _tail_file(path: Path, *, max_bytes: int = 8192) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return "<log file missing>"
    except Exception as exc:
        return f"<failed to read log tail: {exc}>"


def _process_snapshot(process: subprocess.Popen[Any] | None) -> Dict[str, Any]:
    if process is None:
        return {"pid": None, "returncode": None}
    return {"pid": process.pid, "returncode": process.poll()}


def _format_service_diagnostic(url: str, diagnostic: Mapping[str, Any] | None) -> str:
    if diagnostic is None:
        return f"- health_url={url}"
    process = diagnostic.get("process")
    log_path = Path(str(diagnostic["log_path"])) if diagnostic.get("log_path") else None
    parts = [
        f"- service={diagnostic.get('name')}",
        f"health_url={url}",
        f"pid={getattr(process, 'pid', None)}",
        f"returncode={process.poll() if process is not None else None}",
        f"log_path={log_path}",
    ]
    if log_path is not None:
        parts.append(f"log_tail:\n{_tail_file(log_path)}")
    return " ".join(parts)


def wait_for_urls(
    urls: Sequence[str],
    *,
    timeout_sec: float,
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    if not urls:
        return
    deadline = time.time() + timeout_sec
    pending = set(str(url) for url in urls)
    while pending:
        for url in list(pending):
            diagnostic = diagnostics.get(url) if diagnostics else None
            process = diagnostic.get("process") if diagnostic else None
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    "service exited before health check succeeded:\n"
                    + _format_service_diagnostic(url, diagnostic)
                )
            try:
                request = Request(url, method="GET")
                with urlopen(request, timeout=2.0):
                    pending.remove(url)
                    if on_ready is not None:
                        on_ready(url)
            except Exception:
                pass
        if not pending:
            return
        if time.time() >= deadline:
            details = [
                _format_service_diagnostic(url, diagnostics.get(url) if diagnostics else None)
                for url in sorted(pending)
            ]
            raise TimeoutError(
                "timed out waiting for service URLs after "
                f"{timeout_sec:.1f}s:\n" + "\n\n".join(details)
            )
        time.sleep(2.0)


def terminate_processes(
    processes: Sequence[subprocess.Popen[Any]], *, grace_sec: float = 20.0,
) -> None:
    # The shell leader may exit before its service children. Own and terminate
    # the complete session, including groups whose leader has already exited.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_sec
    for process in processes:
        while time.monotonic() < deadline:
            process.poll()  # Reap the leader, but inspect its entire group.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


@dataclass
class ManagedServiceGroup:
    processes: list[subprocess.Popen[Any]]
    log_paths: Dict[str, Path]

    def stop(self) -> None:
        terminate_processes(self.processes)

    def __enter__(self) -> "ManagedServiceGroup":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()


def _capture_command(command: Sequence[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=10.0)
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout[-20000:],
            "stderr": result.stderr[-20000:],
        }
    except FileNotFoundError:
        return {"command": list(command), "error": "command not found"}
    except Exception as exc:
        return {"command": list(command), "error": str(exc)}


def _startup_snapshot(
    commands: Sequence[Mapping[str, Any]],
    *,
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    timeline: Sequence[Mapping[str, Any]] = (),
    error: str | None = None,
) -> Dict[str, Any]:
    diagnostics = diagnostics or {}
    return {
        "timestamp": time.time(),
        "error": error,
        "timeline": [dict(event) for event in timeline],
        "commands": [
            {
                "name": command.get("name"),
                "health_url": command.get("health_url"),
                "port": command.get("port"),
                "gpu_ids": (command.get("env") or {}).get("CUDA_VISIBLE_DEVICES"),
                "process": _process_snapshot(
                    (diagnostics.get(str(command.get("health_url") or "")) or {}).get("process")
                ),
            }
            for command in commands
        ],
        "gpu": _capture_command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "ports": _capture_command(["bash", "-lc", "ss -ltnp 2>/dev/null || lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null"]),
    }


def _write_startup_snapshot(
    log_dir: Path,
    name: str,
    commands: Sequence[Mapping[str, Any]],
    *,
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    timeline: Sequence[Mapping[str, Any]] = (),
    error: str | None = None,
) -> None:
    path = log_dir / f"{name}.json"
    payload = _startup_snapshot(
        commands,
        diagnostics=diagnostics,
        timeline=timeline,
        error=error,
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def start_services(
    *,
    graph: ResolvedServiceGraph,
    project_root: Path,
    log_dir: Path,
    base_env: Mapping[str, str] | None = None,
) -> ManagedServiceGroup:
    """Start the declared service graph, wait for health checks, and return its processes."""

    commands = graph.command_specs()
    if not commands:
        return ManagedServiceGroup(processes=[], log_paths={})
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    timeline: list[Dict[str, Any]] = [
        {"event": "service_group_starting", "elapsed_sec": 0.0}
    ]
    _write_startup_snapshot(
        log_dir,
        "service_startup_before",
        commands,
        timeline=timeline,
    )
    processes: list[subprocess.Popen[Any]] = []
    log_paths: Dict[str, Path] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    try:
        for command_spec in commands:
            service_env = os.environ.copy()
            if base_env:
                service_env.update({str(key): str(value) for key, value in base_env.items()})
            service_env.update(dict(command_spec.get("env") or {}))
            name = str(command_spec.get("name") or "service")
            suffix = uuid.uuid4().hex[:8]
            log_path = log_dir / f"{name}_{suffix}.log"
            log_handle = log_path.open("ab")
            try:
                process = subprocess.Popen(
                    str(command_spec["command"]),
                    cwd=str(project_root),
                    env=service_env,
                    shell=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_handle.close()
            processes.append(process)
            log_paths[name] = log_path
            timeline.append(
                {
                    "event": "service_spawned",
                    "service": name,
                    "pid": process.pid,
                    "elapsed_sec": time.monotonic() - started_at,
                }
            )
            health_url = str(command_spec.get("health_url") or "")
            if health_url:
                diagnostics[health_url] = {
                    "name": name,
                    "process": process,
                    "log_path": log_path,
                    "command": str(command_spec["command"]),
                }

        startup_timeout = graph.startup_timeout_sec * 2.0

        def record_ready(url: str) -> None:
            diagnostic = diagnostics.get(url) or {}
            timeline.append(
                {
                    "event": "service_health_ready",
                    "service": diagnostic.get("name"),
                    "health_url": url,
                    "elapsed_sec": time.monotonic() - started_at,
                }
            )

        wait_for_urls(
            graph.health_urls,
            timeout_sec=startup_timeout,
            diagnostics=diagnostics,
            on_ready=record_ready,
        )
        timeline.append(
            {
                "event": "service_group_ready",
                "elapsed_sec": time.monotonic() - started_at,
            }
        )
        _write_startup_snapshot(
            log_dir,
            "service_startup_after",
            commands,
            diagnostics=diagnostics,
            timeline=timeline,
        )
        extra_sleep = max(float(command.get("startup_sleep_sec", 0.0)) for command in commands)
        if extra_sleep > 0:
            time.sleep(extra_sleep)
        return ManagedServiceGroup(processes=processes, log_paths=log_paths)
    except BaseException as exc:
        timeline.append(
            {
                "event": "service_group_failed",
                "elapsed_sec": time.monotonic() - started_at,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_startup_snapshot(
            log_dir,
            "service_startup_failed",
            commands,
            diagnostics=diagnostics,
            timeline=timeline,
            error=f"{type(exc).__name__}: {exc}",
        )
        terminate_processes(processes)
        raise
