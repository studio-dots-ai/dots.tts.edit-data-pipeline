"""Run config loading for distributed-pipeline pipeline workers."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .paths import get_artifact_paths

from .models import (
    DEFAULT_STAGE_WORKER_PYTHON_BIN,
    ParquetOutputConfig,
    StageSpec,
)


DEFAULT_SOURCE_QUEUE_NAME = "source_items"
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _read_config_file(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML run configs") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"run config must be a mapping: {config_path}")
    return data


def load_run_config(path: str | Path, *, resolve: bool = True) -> Dict[str, Any]:
    config_path = _resolve_run_config_path(path)
    if _is_hydra_source_config(config_path):
        data = _compose_hydra_config(config_path, resolve=resolve)
    else:
        data = _read_config_file(config_path)
        if resolve:
            data = resolve_run_config(data)
    data.setdefault("run_config_path", str(config_path))
    return data


def _resolve_run_config_path(path: str | Path) -> Path:
    raw = Path(str(path)).expanduser()
    if raw.suffix.lower() in {".yaml", ".yml", ".json"} or raw.exists():
        return raw.resolve()
    config_name = str(path).strip().removesuffix(".yaml").removesuffix(".yml")
    candidate = CONFIG_ROOT / f"{config_name}.yaml"
    if candidate.exists():
        return candidate.resolve()
    return raw.resolve()


def _is_hydra_source_config(path: Path) -> bool:
    try:
        path.relative_to(CONFIG_ROOT)
    except ValueError:
        return False
    return path.suffix.lower() in {".yaml", ".yml"}


def register_artifact_resolver() -> None:
    """Resolve artifact roots from the effective config, including on remote workers."""
    from omegaconf import OmegaConf

    def artifact(scope: str, *, _root_: Any) -> str:
        if scope not in {"worktree", "shared"}:
            raise ValueError(f"unknown artifact scope: {scope}")
        paths = get_artifact_paths(
            project_root=_root_.get("project_root"),
            artifact_dir=_root_.get("artifact_dir"),
        )
        return str(getattr(paths, scope))

    OmegaConf.register_new_resolver("artifact", artifact, replace=True)
    OmegaConf.register_new_resolver("json", lambda value: json.dumps(OmegaConf.to_container(value, resolve=True)), replace=True)
    OmegaConf.register_new_resolver("shell_quote", lambda value: shlex.quote(str(value)), replace=True)


def resolve_run_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("omegaconf>=2.3 is required to resolve distributed-pipeline configs") from exc
    register_artifact_resolver()
    cfg = OmegaConf.create(dict(data))
    cfg.setdefault("project_root", str(Path(__file__).resolve().parents[2]))
    paths = get_artifact_paths(
        project_root=cfg.get("project_root"), artifact_dir=cfg.get("artifact_dir")
    )
    cfg.artifact_dir = str(paths.root)
    # Resolve run_dir before dependent fields such as parquet and service logs.
    if cfg.get("run_dir") is not None:
        cfg.run_dir = str(paths.resolve(cfg.run_dir))
    # Normalize outputs before resolving commands that interpolate those fields.
    # Otherwise a relative service output can be validated yet remain relative
    # inside its already-expanded command string.
    from omegaconf import DictConfig, ListConfig

    def validate_outputs(value: Any) -> None:
        if isinstance(value, DictConfig):
            for key, item in list(value.items_ex(resolve=False)):
                if key in {"run_dir", "artifact_root", "output_dir", "service_log_root"} and item:
                    value[key] = str(paths.resolve(value[key]))
                else:
                    validate_outputs(item)
        elif isinstance(value, ListConfig):
            for item in value:
                validate_outputs(item)

    validate_outputs(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("run config must resolve to a mapping")
    return resolved


def _compose_hydra_config(config_path: Path, *, resolve: bool) -> Dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("hydra-core==1.3.2 is required to compose distributed-pipeline configs") from exc
    config_name = str(config_path.relative_to(CONFIG_ROOT).with_suffix(""))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
        cfg = compose(config_name=config_name)
    data = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(data, dict):
        raise ValueError(f"run config must compose to a mapping: {config_path}")
    return resolve_run_config(data) if resolve else dict(data)


def _parquet_output_from_mapping(
    raw: Mapping[str, Any] | None,
    default: ParquetOutputConfig,
) -> ParquetOutputConfig:
    if not raw:
        return default
    return ParquetOutputConfig(
        enabled=bool(raw.get("enabled", default.enabled)),
        output_dir=raw.get("output_dir", default.output_dir),
        dataset_name=raw.get("dataset_name", default.dataset_name),
        flush_rows=int(raw.get("flush_rows", default.flush_rows)),
        flush_bytes_mb=int(raw.get("flush_bytes_mb", default.flush_bytes_mb)),
        flush_interval_sec=float(raw.get("flush_interval_sec", default.flush_interval_sec)),
    )


def _queue_bindings(run_config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    topology = run_config.get("topology", {})
    if topology is None:
        return {}
    if not isinstance(topology, dict):
        raise ValueError("topology must be a mapping")
    bindings = topology.get("bindings", {})
    if bindings is None:
        return {}
    if not isinstance(bindings, dict):
        raise ValueError("topology.bindings must be a mapping")
    return {str(queue_name): dict(raw or {}) for queue_name, raw in bindings.items()}


def _binding_producers(queue_name: str, binding: Mapping[str, Any]) -> List[str]:
    producers: List[str] = []
    raw_producers = binding.get("producers", [])
    if raw_producers is None:
        raw_producers = []
    if isinstance(raw_producers, str):
        raw_producers = [raw_producers]
    if not isinstance(raw_producers, list):
        raise ValueError(f"topology binding for {queue_name!r} producers must be a list")
    producers.extend(str(item) for item in raw_producers if item)
    if "producer" in binding:
        raise ValueError("Use topology.bindings producers (a list), not producer")
    return list(dict.fromkeys(producers))


def _stage_queue_maps(run_config: Mapping[str, Any]) -> tuple[Dict[str, str], Dict[str, Optional[str]]]:
    inputs: Dict[str, str] = {}
    outputs: Dict[str, Optional[str]] = {}
    for queue_name, binding in _queue_bindings(run_config).items():
        for stage_name in _binding_producers(queue_name, binding):
            if stage_name in outputs and outputs[stage_name] != queue_name:
                outputs[stage_name] = None
            else:
                outputs[stage_name] = queue_name
        consumers = binding.get("consumers", [])
        if consumers is None:
            consumers = []
        if isinstance(consumers, str):
            consumers = [consumers]
        if not isinstance(consumers, list):
            raise ValueError(f"topology binding for {queue_name!r} consumers must be a list")
        for consumer in consumers:
            stage_name = str(consumer)
            if stage_name in inputs and inputs[stage_name] != queue_name:
                raise ValueError(f"stage {stage_name!r} declares multiple input queues")
            inputs[stage_name] = queue_name
    return inputs, outputs


def stage_specs_from_config(run_config: Mapping[str, Any]) -> Dict[str, StageSpec]:
    raw_stages = run_config.get("stages", {})
    if not isinstance(raw_stages, dict):
        raise ValueError("stages must be a mapping")
    if not isinstance(run_config.get("topology"), dict):
        raise ValueError("run config must declare topology.bindings")
    inputs, outputs = _stage_queue_maps(run_config)
    specs: Dict[str, StageSpec] = {}
    for stage_name, raw in raw_stages.items():
        name = str(stage_name)
        if not isinstance(raw, dict):
            raise ValueError(f"stage {name!r} must be a mapping")
        worker_python_bin = raw.get("worker_python_bin")
        if not worker_python_bin:
            raise ValueError(f"stage {name!r} must define worker_python_bin")
        default_spec = StageSpec(
            name=name,
            input_queue=None,
            output_queue=None,
            worker_python_bin=DEFAULT_STAGE_WORKER_PYTHON_BIN,
        )
        parquet_output = _parquet_output_from_mapping(raw.get("parquet_output"), default_spec.parquet_output)
        lease_seconds = float(raw.get("lease_seconds", default_spec.lease_seconds))
        queue_max_size = raw.get("queue_max_size")
        max_workers = int(raw.get("max_workers", default_spec.max_workers))
        num_clients_per_worker = max(
            1,
            int(raw.get("num_clients_per_worker", default_spec.num_clients_per_worker)),
        )
        spec = StageSpec(
            name=name,
            input_queue=inputs.get(name),
            output_queue=outputs.get(name),
            worker_python_bin=str(worker_python_bin),
            max_workers=max_workers,
            num_clients_per_worker=num_clients_per_worker,
            port_stride=int(raw.get("port_stride", default_spec.port_stride)),
            queue_max_size=int(queue_max_size) if queue_max_size is not None else None,
            lease_seconds=lease_seconds,
            parquet_output=parquet_output,
            services=dict(raw.get("services") or {}),
        )
        specs[name] = spec
    for stage_name in set(inputs) | set(outputs):
        if stage_name not in specs:
            raise ValueError(f"topology references undefined stage {stage_name!r}")
    return specs


def pipeline_config(run_config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = run_config.get("pipeline_config", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("pipeline_config must be a mapping")
    config = dict(raw)
    return config


def source_config(run_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return source queue seeding config with stable distributed-pipeline defaults."""
    raw = run_config.get("source", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("source must be a mapping")
    source = dict(raw)
    if "max_items" in source:
        raise ValueError("source.max_items was removed; use source.max_items_per_input")
    retired_keys = {"input_manifest", "input_manifests", "parquet_list", "parquet_source"} & source.keys()
    if retired_keys:
        raise ValueError("Use source.inputs instead of " + ", ".join(sorted(retired_keys)))
    source.setdefault("queue_name", DEFAULT_SOURCE_QUEUE_NAME)
    source.setdefault("rows_per_item", 256)
    return source


def queue_max_sizes(
    run_config: Mapping[str, Any],
    stage_specs: Mapping[str, StageSpec],
) -> Dict[str, Optional[int]]:
    """Return logical queue capacity limits from run config and stage specs."""
    limits: Dict[str, Optional[int]] = {}
    raw_queues = run_config.get("queues", {})
    if isinstance(raw_queues, dict):
        for queue_name, raw in raw_queues.items():
            if isinstance(raw, dict):
                max_size = raw.get("max_size")
            else:
                max_size = raw
            if max_size is not None:
                limits[str(queue_name)] = int(max_size)
    for spec in stage_specs.values():
        if spec.output_queue and spec.queue_max_size is not None:
            limits.setdefault(spec.output_queue, int(spec.queue_max_size))
    return limits
