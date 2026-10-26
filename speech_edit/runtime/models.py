"""Shared data models for distributed-pipeline pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from speech_edit.runtime.services import ResolvedServiceRoots

DEFAULT_STAGE_WORKER_PYTHON_BIN = "envs/pipe-cpu/.venv/bin/python"


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return override


def _stage_config(run_config: Mapping[str, Any], stage_name: str) -> Dict[str, Any]:
    raw_stages = run_config.get("stages", {})
    if not isinstance(raw_stages, dict):
        return {}
    raw = raw_stages.get(stage_name, {})
    return dict(raw) if isinstance(raw, dict) else {}


@dataclass(frozen=True)
class ParquetOutputConfig:
    """Batching policy for save-stage parquet materialization."""

    enabled: bool = False
    output_dir: Optional[str] = None
    dataset_name: Optional[str] = None
    flush_rows: int = 256
    flush_bytes_mb: int = 256
    flush_interval_sec: float = 60.0


@dataclass(frozen=True)
class StageSpec:
    """Pipeline-provided stage declaration.

    The distributed pipeline treats stages as opaque DAG nodes.  Pipeline adapters
    decide how many stages exist and what each stage means.
    """

    name: str
    input_queue: Optional[str]
    output_queue: Optional[str]
    worker_python_bin: str
    max_workers: int = 1
    num_clients_per_worker: int = 1
    port_stride: int = 100
    queue_max_size: Optional[int] = None
    lease_seconds: float = 3600.0
    parquet_output: ParquetOutputConfig = field(default_factory=ParquetOutputConfig)
    services: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterContext:
    run_config: Mapping[str, Any]
    worker_id: str
    stage: StageSpec
    services: ResolvedServiceRoots

    @property
    def pipeline_config(self) -> Dict[str, Any]:
        raw = self.run_config.get("pipeline_config", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("pipeline_config must be a mapping")
        stage = _stage_config(self.run_config, self.stage.name)
        stage_cfg = stage.get("pipeline_config", {})
        if stage_cfg is None:
            stage_cfg = {}
        if not isinstance(stage_cfg, dict):
            raise ValueError(f"stage {self.stage.name!r} has invalid pipeline_config")
        config = _deep_merge(dict(raw), stage_cfg)
        return config

@dataclass(frozen=True)
class OutputItem:
    """One item to enqueue after a stage succeeds."""

    queue_name: str
    item_key: str
    payload: Any
    max_attempts: int = 1


@dataclass(frozen=True)
class StageResult:
    """Result returned by a pipeline stage adapter."""

    outputs: List[OutputItem] = field(default_factory=list)
    parquet_rows: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimedItem:
    """Queue item leased by one worker."""

    id: int
    queue_name: str
    item_key: str
    payload_ref: Optional[str]
    payload_json: Optional[str]
    attempts: int
