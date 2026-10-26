"""Dynamic adapter loading for distributed-pipeline pipelines."""

from __future__ import annotations

import importlib
from typing import Any, Mapping


def load_object(import_path: str) -> Any:
    """Load ``module:attribute`` or ``module.attribute``."""
    if ":" in import_path:
        module_name, attr_name = import_path.split(":", 1)
    else:
        module_name, sep, attr_name = import_path.rpartition(".")
        if not sep:
            raise ValueError(
                "adapter_class must be an import path like "
                "'package.module:ClassName'"
            )
    if not module_name or not attr_name:
        raise ValueError(
            "adapter_class must be an import path like 'package.module:ClassName'"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ValueError(f"Failed to import adapter module {module_name!r}") from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Adapter module {module_name!r} has no attribute {attr_name!r}"
        ) from exc


def get_adapter(pipeline: str, run_config: Mapping[str, Any]) -> Any:
    """Instantiate the pipeline-owned adapter declared by a run config."""
    adapter_path = run_config.get("adapter_class")
    if not adapter_path:
        raise ValueError(
            f"Run config for pipeline {pipeline!r} must declare adapter_class"
        )
    adapter_cls = load_object(str(adapter_path))
    return adapter_cls()
