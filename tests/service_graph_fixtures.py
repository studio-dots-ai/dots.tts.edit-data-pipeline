"""Test fixtures for direct service views."""

from types import MappingProxyType
from typing import Mapping

from speech_edit.runtime.services import ResolvedServiceEndpoint, ResolvedServiceRoots


def direct_services(urls: Mapping[str, str] | None = None, /, **kwargs: str) -> ResolvedServiceRoots:
    declared = {**dict(urls or {}), **kwargs}
    return ResolvedServiceRoots(
        endpoints=MappingProxyType(
            {
                key: ResolvedServiceEndpoint(
                    dep_ref_key=key,
                    instance_key=f"test:{key}",
                    url=url,
                )
                for key, url in declared.items()
            }
        )
    )
