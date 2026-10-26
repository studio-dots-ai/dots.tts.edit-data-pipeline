"""Prepare a pipeline run in an external PostgreSQL database."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import load_run_config, pipeline_config, resolve_run_config, source_config
from .paths import get_artifact_paths
from .source_seed import seed_source_from_run_config
from .queue_store import QueueStore

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$")


def _default_run_dir(
    project_root: Path, pipeline: str, run_id: str, *, artifact_dir: str | None = None
) -> Path:
    return (
        get_artifact_paths(project_root, artifact_dir=artifact_dir).worktree
        / "distributed_pipeline_runs"
        / pipeline
        / run_id
    )


def validate_run_id(run_id: str) -> str:
    """Validate the short run id used in run directories."""
    value = str(run_id).strip()
    if not _RUN_ID_RE.fullmatch(value):
        raise ValueError(
            "run_id must be 1-16 chars, start with a letter/digit, and contain "
            "only letters, digits, '.', '_', or '-'"
        )
    return value


def _write_effective_config(run_config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        output_path.write_text(
            yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except ImportError:  # pragma: no cover
        output_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _run_git(args: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_summary(project_root: Path) -> dict[str, Any]:
    status = _run_git(["status", "--short"], cwd=project_root)
    return {
        "commit": _run_git(["rev-parse", "HEAD"], cwd=project_root),
        "branch": _run_git(["branch", "--show-current"], cwd=project_root),
        "dirty": bool(status),
        "status_short": status.splitlines()[:200] if status else [],
    }


def _write_run_meta(
    *,
    run_config: dict[str, Any],
    original_config_path: str,
) -> None:
    run_dir = Path(str(run_config["run_dir"]))
    source = source_config(run_config)
    project_root = Path(str(run_config["project_root"]))
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_config["run_id"],
        "pipeline": run_config["pipeline"],
        "original_run_config_path": str(Path(original_config_path).expanduser().resolve()),
        "run_dir": str(run_dir),
        "artifact_root": str(run_config["artifact_root"]),
        "source": {
            "inputs": source.get("inputs"),
            "queue_name": source.get("queue_name"),
            "rows_per_item": source.get("rows_per_item"),
        },
        "project_root": str(project_root),
        "git": _git_summary(project_root),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def prepare_run_config(
    *,
    run_config_path: str,
    run_id: str | None = None,
    run_dir: str | None = None,
) -> tuple[dict[str, Any], Path]:
    run_config = load_run_config(run_config_path, resolve=False)
    project_root = Path(str(run_config.get("project_root") or Path(__file__).resolve().parents[2])).resolve()
    run_config["project_root"] = str(project_root)
    if run_id:
        run_config["run_id"] = validate_run_id(run_id)
    elif run_config.get("run_id"):
        run_config["run_id"] = validate_run_id(str(run_config["run_id"]))
    else:
        run_config["run_id"] = validate_run_id(f"r{int(time.time())}")
    pipeline = str(run_config["pipeline"])
    # Apply explicit CLI overrides before interpolation and path validation.
    if run_dir is not None:
        run_config["run_dir"] = run_dir
    elif not run_config.get("run_dir"):
        run_config["run_dir"] = str(_default_run_dir(
            project_root, pipeline, str(run_config["run_id"]),
            artifact_dir=run_config.get("artifact_dir"),
        ))
    run_config.setdefault("artifact_root", "${run_dir}/artifacts")
    runtime = run_config.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    runtime["run_dir"] = "${run_dir}"
    run_config["runtime"] = runtime
    run_config = resolve_run_config(run_config)
    effective_run_dir = Path(run_config["run_dir"])
    run_config["source"] = source_config(run_config)
    run_config["pipeline_config"] = pipeline_config(run_config)
    effective_config_path = effective_run_dir / "run_config.yaml"
    run_config["run_config_path"] = str(effective_config_path)
    effective_run_dir.mkdir(parents=True, exist_ok=False)
    _write_effective_config(run_config, effective_config_path)
    _write_run_meta(
        run_config=run_config,
        original_config_path=run_config_path,
    )
    return run_config, effective_config_path


def prepare_run(*, run_config_path: str, run_id: str | None = None, run_dir: str | None = None) -> Path:
    """Create a new run and seed its input queue; existing runs are resumed separately."""
    if not os.environ.get("DIST_PIPELINE_DATABASE_URL"):
        raise RuntimeError("DIST_PIPELINE_DATABASE_URL must point to an existing PostgreSQL database")
    config, path = prepare_run_config(run_config_path=run_config_path, run_id=run_id, run_dir=run_dir)
    store = QueueStore(run_id=str(config["run_id"]))
    try:
        store.create_run()
        try:
            seed_source_from_run_config(run_config=config)
            store.set_run_state(desired_state="running")
        except Exception:
            store.set_run_state(desired_state="failed")
            raise
    finally:
        store.close()
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a pipeline run using external PostgreSQL")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    parser.add_argument("--local", action="store_true", help="Start workers on this machine after seeding")
    args = parser.parse_args(argv)
    path = prepare_run(run_config_path=args.run_config, run_id=args.run_id, run_dir=args.run_dir)
    print(path, flush=True)
    if args.local:
        from .local_runner import run_local
        run_local(run_config_path=str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
