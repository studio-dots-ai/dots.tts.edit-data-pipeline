"""Materialize and manage local service graphs for single-node workers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _replace_strings(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _visit_services(services: Mapping[str, Any]):
    for service in services.values():
        if not isinstance(service, Mapping):
            continue
        yield service
        dependencies = service.get("dependencies") or {}
        if isinstance(dependencies, Mapping):
            yield from _visit_services(dependencies)


def _port_replacements(services: Mapping[str, Any], *, port_offset: int) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for service in _visit_services(services):
        raw_port = service.get("port")
        if raw_port in (None, ""):
            continue
        old_port = int(raw_port)
        new_port = old_port + port_offset
        replacements[f":{old_port}"] = f":{new_port}"
        replacements[f"/{old_port}"] = f"/{new_port}"
        replacements[f"-port {old_port}"] = f"-port {new_port}"
        replacements[f"--port {old_port}"] = f"--port {new_port}"
    return replacements


def _apply_port_offset(service: dict[str, Any], *, port_offset: int) -> None:
    raw_port = service.get("port")
    if raw_port not in (None, ""):
        service["port"] = int(raw_port) + port_offset
    dependencies = service.get("dependencies") or {}
    if isinstance(dependencies, dict):
        for dependency in dependencies.values():
            if isinstance(dependency, dict):
                _apply_port_offset(dependency, port_offset=port_offset)


def materialize_worker_service_graph(
    services: Mapping[str, Any],
    *,
    worker_index: int,
    port_stride: int = 100,
) -> dict[str, Any]:
    """Copy a graph with worker-specific ports, preserving declared GPU IDs."""

    if worker_index < 0 or port_stride < 1:
        raise ValueError("worker_index must be nonnegative and port_stride positive")
    materialized = deepcopy(dict(services or {}))
    port_offset = worker_index * port_stride
    replacements = _port_replacements(materialized, port_offset=port_offset)
    if replacements:
        materialized = _replace_strings(materialized, replacements)
        for service in materialized.values():
            if isinstance(service, dict):
                _apply_port_offset(service, port_offset=port_offset)
    return materialized
