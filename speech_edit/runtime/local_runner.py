"""Local multi-worker runner for distributed-pipeline configs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .config import load_run_config
from .queue_store import QueueStore
from .registry import get_adapter


@dataclass(frozen=True)
class LocalWorkerSpec:
    stage: str
    worker_index: int
    worker_id: str
    python_bin: str
    gpu_id: Optional[str]


def _stage_local_config(raw: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    local_workers = raw.get("local_workers", {})
    if not isinstance(local_workers, dict):
        return {}
    stage_cfg = local_workers.get(stage, {})
    if isinstance(stage_cfg, dict):
        return stage_cfg
    return {}


def build_local_worker_specs(run_config: Mapping[str, Any]) -> List[LocalWorkerSpec]:
    pipeline = str(run_config["pipeline"])
    adapter = get_adapter(pipeline, run_config)
    stage_specs = adapter.stage_specs(run_config)
    specs: List[LocalWorkerSpec] = []
    for stage_name, stage_spec in stage_specs.items():
        local_cfg = _stage_local_config(run_config, stage_name)
        count = int(stage_spec.max_workers)
        gpu_ids = local_cfg.get("gpu_ids", [])
        if gpu_ids is None:
            gpu_ids = []
        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]
        for index in range(count):
            gpu_id = str(gpu_ids[index % len(gpu_ids)]) if gpu_ids else None
            specs.append(
                LocalWorkerSpec(
                    stage=stage_name,
                    worker_index=len(specs),
                    worker_id=f"{stage_name}-local-{index}-{uuid.uuid4().hex[:8]}",
                    python_bin=str(stage_spec.worker_python_bin),
                    gpu_id=gpu_id,
                )
            )
    return specs


def _worker_env(spec: LocalWorkerSpec, base_env: Mapping[str, str]) -> Dict[str, str]:
    env = dict(base_env)
    env["DIST_PIPELINE_WORKER_INDEX"] = str(spec.worker_index)
    if spec.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    return env


def _launch_worker(
    *,
    project_root: Path,
    run_config_path: str,
    pipeline: str,
    spec: LocalWorkerSpec,
    log_dir: Path,
) -> subprocess.Popen[Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.worker_id}.log"
    log_handle = log_path.open("ab")
    args = [
        spec.python_bin,
        "-m",
        "speech_edit.runtime.entry",
        "--run-config",
        run_config_path,
        "--pipeline",
        pipeline,
        "--stage",
        spec.stage,
        "--worker-id",
        spec.worker_id,
    ]
    try:
        return subprocess.Popen(
            args,
            cwd=str(project_root),
            env=_worker_env(spec, os.environ),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def _terminate(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 20.0
    for process in processes:
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_local(
    *,
    run_config_path: str,
    timeout_sec: float | None = None,
    poll_sec: float = 5.0,
) -> Dict[str, int]:
    run_config = load_run_config(run_config_path)
    os.environ["ARTIFACT_DIR"] = run_config["artifact_dir"]
    pipeline = str(run_config["pipeline"])
    project_root = Path(str(run_config.get("project_root") or Path.cwd()))
    run_dir = Path(str(run_config["run_dir"]))
    log_dir = run_dir / "logs" / "local_runner"
    store = QueueStore(run_id=str(run_config["run_id"]))
    specs = build_local_worker_specs(run_config)
    if not specs:
        raise ValueError("No local workers configured")
    if store.get_run_state()["desired_state"] != "running":
        raise ValueError("Resume the run before starting workers")
    processes = []
    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    try:
        for spec in specs:
            processes.append(_launch_worker(
                project_root=project_root, run_config_path=run_config_path,
                pipeline=pipeline, spec=spec, log_dir=log_dir,
            ))
        while deadline is None or time.monotonic() < deadline:
            stats = store.completion_stats()
            if stats.get("run_complete") == 1:
                return stats
            failed_work = {
                key: value
                for key, value in stats.items()
                if key == "queue_dead" and int(value) > 0
            }
            if failed_work:
                raise RuntimeError(f"local run has failed work: {failed_work}")
            if store.get_run_state()["desired_state"] != "running":
                if all(process.poll() is not None for process in processes):
                    return stats
            elif all(process.poll() is not None for process in processes):
                raise RuntimeError("All workers exited before the run completed")
            failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"local worker exited unsuccessfully: {failed}")
            time.sleep(poll_sec)
        raise TimeoutError(f"local run did not complete within {timeout_sec:.1f}s")
    finally:
        _terminate(processes)
        store.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a distributed-pipeline config locally")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    args = parser.parse_args(argv)
    stats = run_local(
        run_config_path=args.run_config,
        timeout_sec=args.timeout_sec,
        poll_sec=args.poll_sec,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
