"""Generic adapter protocol for distributed-pipeline pipeline execution."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .config import stage_specs_from_config
from .models import (
    AdapterContext,
    StageResult,
    StageSpec,
)


class PipelineAdapter:
    """Base class for pipeline-owned stage adapters.

    Concrete adapters live with their pipeline packages.  The distributed pipeline only
    applies config overrides and calls the adapter through this small protocol.
    """

    name = ""

    def stage_specs(self, run_config: Mapping[str, Any]) -> Dict[str, StageSpec]:
        return stage_specs_from_config(run_config)

    def process(self, stage_name: str, payload: Any, ctx: AdapterContext) -> StageResult:
        raise NotImplementedError
