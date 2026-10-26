"""Worker entrypoint with optional stage-local service startup."""

from __future__ import annotations

import argparse
import os
import signal
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from speech_edit.runtime.services.single_node_workers import (
    materialize_worker_service_graph,
)
from .services import resolve_service_graph, start_services
from .config import load_run_config
from .models import StageSpec
from .registry import get_adapter
from .worker import run_worker


def _install_termination_handlers() -> None:
    def exit_on_signal(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGINT, exit_on_signal)


def _load_stage_spec(run_config: Mapping[str, Any], pipeline: str, stage_name: str) -> StageSpec:
    adapter = get_adapter(pipeline, run_config)
    specs = adapter.stage_specs(run_config)
    if stage_name not in specs:
        raise ValueError(f"stage {stage_name!r} is not declared by pipeline {pipeline!r}")
    return specs[stage_name]


def main(argv: Sequence[str] | None = None) -> int:
    _install_termination_handlers()
    parser = argparse.ArgumentParser(description="Run a distributed-pipeline worker entrypoint")
    parser.add_argument("--run-config", default=os.environ.get("RUN_CONFIG"))
    parser.add_argument("--pipeline", default=os.environ.get("PIPELINE"))
    parser.add_argument("--stage", default=os.environ.get("STAGE"))
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--idle-sleep-sec", type=float, default=5.0)
    args = parser.parse_args(argv)
    if not args.run_config or not args.pipeline or not args.stage:
        raise ValueError("--run-config, --pipeline, and --stage are required")
    worker_id = args.worker_id or f"{args.stage}-{uuid.uuid4().hex[:8]}"
    args.worker_id = worker_id
    run_config = load_run_config(args.run_config)
    os.environ["ARTIFACT_DIR"] = run_config["artifact_dir"]
    stage_spec = _load_stage_spec(run_config, args.pipeline, args.stage)
    project_root = Path(str(run_config.get("project_root") or Path.cwd()))
    run_dir = Path(str(run_config["run_dir"]))
    services = dict(stage_spec.services or {})
    worker_index = int(os.environ.get("DIST_PIPELINE_WORKER_INDEX", "0") or 0)
    service_graph = resolve_service_graph(materialize_worker_service_graph(
        services, worker_index=worker_index, port_stride=stage_spec.port_stride,
    ))
    with start_services(
        graph=service_graph, project_root=project_root,
        log_dir=run_dir / "logs" / "stage_services" / worker_id,
    ):
        return run_worker(
            run_config_path=args.run_config,
            pipeline=args.pipeline,
            stage_name=args.stage,
            worker_id=worker_id,
            once=args.once,
            idle_sleep_sec=args.idle_sleep_sec,
            services=service_graph.direct_services,
        )


if __name__ == "__main__":
    raise SystemExit(main())
