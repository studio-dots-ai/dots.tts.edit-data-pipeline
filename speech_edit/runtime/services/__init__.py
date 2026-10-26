"""Shared local service graph resolution and process management."""

from .processes import ManagedServiceGroup, start_services, terminate_processes, wait_for_urls
from .resolver import (
    ResolvedServiceGraph,
    ResolvedServiceNode,
    ResolvedServiceEndpoint,
    ResolvedServiceRoots,
    ServiceCaller,
    resolve_service_graph,
)

__all__ = [
    "ManagedServiceGroup",
    "ResolvedServiceGraph",
    "ResolvedServiceNode",
    "ResolvedServiceEndpoint",
    "ResolvedServiceRoots",
    "ServiceCaller",
    "resolve_service_graph",
    "start_services",
    "terminate_processes",
    "wait_for_urls",
]
