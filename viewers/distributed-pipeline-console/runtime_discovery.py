"""Discover distributed-pipeline runs without importing pipeline adapters."""

from __future__ import annotations


from speech_edit.runtime.config import load_run_config
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CONFIG_NAMES = ("run_config.json", "run_config.yaml", "run_config.yml")


@dataclass(frozen=True)
class RuntimeRunInfo:
    date: str
    name: str
    path: Path
    config_path: Optional[Path]
    run_config: dict[str, Any]
    config_error: Optional[str] = None

    @property
    def run_id(self) -> str:
        return f"{self.date}/{self.name}"

    @property
    def store_run_id(self) -> str:
        value = self.run_config.get("run_id")
        return str(value) if value else self.run_id

    @property
    def pipeline(self) -> str:
        value = self.run_config.get("pipeline")
        return str(value) if value else ""

    @property
    def run_dir(self) -> Path:
        value = self.run_config.get("run_dir")
        if value:
            return _resolve_config_path(str(value), self.config_path, self.path)
        return self.path


    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "name": self.name,
            "run_id": self.run_id,
            "store_run_id": self.store_run_id,
            "path": str(self.path),
            "run_dir": str(self.run_dir),
            "pipeline": self.pipeline,
            "config_path": str(self.config_path) if self.config_path else None,
            "config_error": self.config_error,
        }


def candidate_run_paths(data_root: Path, date: str, name: str) -> list[Path]:
    if not DATE_RE.fullmatch(date):
        raise ValueError(f"Invalid date: {date}")
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise ValueError(f"Invalid run name: {name}")
    return [
        path for path in _iter_pipeline_run_paths(data_root)
        if path.name == name and distributed_run_display_date(path) == date
    ]


def _find_default_config(run_path: Path) -> Optional[Path]:
    for name in CONFIG_NAMES:
        path = run_path / name
        if path.is_file():
            return path
    return None


def latest_config_mtime(run_path: Path) -> float:
    mtimes: list[float] = []
    for path in [p for p in [_find_default_config(run_path)] if p is not None]:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else 0.0


def distributed_run_display_date(run_path: Path) -> str:
    mtime = latest_config_mtime(run_path)
    if mtime <= 0:
        return datetime.fromtimestamp(run_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")


def _resolve_config_path(value: str, config_path: Optional[Path], run_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_path is not None:
        return (config_path.parent / path).resolve()
    return (run_path / path).resolve()


def _is_runtime_run(
    *,
    config_path: Optional[Path],
    run_config: dict[str, Any],
) -> bool:
    if not config_path:
        return False
    if "adapter_class" in run_config or "run_id" in run_config:
        return True
    return False


def load_run_info(date: str, run_path: Path) -> Optional[RuntimeRunInfo]:
    config_path = _find_default_config(run_path)
    config_error = None
    run_config: dict[str, Any] = {}
    if config_path:
        try:
            run_config = load_run_config(config_path)
        except Exception as exc:
            config_error = str(exc) if not config_error else f"{config_error}; {exc}"
            run_config = {}
    if not _is_runtime_run(
        config_path=config_path,
        run_config=run_config,
    ):
        return None
    return RuntimeRunInfo(
        date=date,
        name=run_path.name,
        path=run_path,
        config_path=config_path,
        run_config=run_config,
        config_error=config_error,
    )


def _iter_pipeline_run_paths(data_root: Path) -> Iterable[Path]:
    root = data_root.resolve()
    distributed_root = root / "distributed_pipeline_runs"
    if not distributed_root.is_dir():
        distributed_root = root
    if not distributed_root.is_dir():
        return
    try:
        pipeline_dirs = sorted(distributed_root.iterdir(), key=lambda path: path.name)
    except OSError:
        return
    for pipeline_dir in pipeline_dirs:
        if not pipeline_dir.is_dir() or DATE_RE.match(pipeline_dir.name):
            continue
        try:
            run_dirs = sorted(
                pipeline_dir.iterdir(),
                key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
                reverse=True,
            )
        except OSError:
            continue
        for run_path in run_dirs:
            if run_path.is_dir():
                yield run_path


def iter_runs(
    data_root: Path,
    *,
    start_date: str = "",
    end_date: Optional[str] = None,
) -> Iterable[RuntimeRunInfo]:
    for run_path in _iter_pipeline_run_paths(data_root):
        display_date = distributed_run_display_date(run_path)
        if display_date < start_date or (end_date and display_date > end_date):
            continue
        run = load_run_info(display_date, run_path)
        if run is not None:
            yield run


def discover_run(data_root: Path, date: str, name: str) -> RuntimeRunInfo:
    for run_path in candidate_run_paths(data_root, date, name):
        if run_path.is_dir():
            run = load_run_info(date, run_path)
            if run is not None:
                return run
    raise FileNotFoundError(f"Stage-runtime run not found: {date}/{name}")
