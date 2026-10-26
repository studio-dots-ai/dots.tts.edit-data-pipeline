"""Interfaces for direct parquet row construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ParquetRowProcessor(ABC):
    """Convert evaluated stage payloads directly into parquet-ready rows."""

    @abstractmethod
    def rows_from_payload(
        self,
        payload: Any,
        *,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """Build parquet-ready rows from a durable queue payload."""
        ...
