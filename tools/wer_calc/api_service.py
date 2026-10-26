"""
WER/CER Evaluation HTTP Service.

FastAPI service for computing Word Error Rate (WER) and Character Error Rate (CER).
Delegates to wer_calc.py for core computation.

Usage:
    source envs/pipe-cpu/.venv/bin/activate
    python api_service.py --port 8011

    # API docs: http://localhost:8011/docs
"""

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path so tools.common is importable when running directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.common import ServiceRunner, setup_logging

from wer_calc import (
    aggregate_error_rate_results,
    compute_error_rate,
    compute_error_rate_batch,
    WERResult,
)

logger = setup_logging(__name__)

app = FastAPI(
    title="WER/CER Evaluation Service",
    description="Compute Word Error Rate (WER) and Character Error Rate (CER)",
    version="1.0.0",
)


# -------------------------------- Models ----------------------------------- #


class WERRequest(BaseModel):
    reference: str = Field(..., description="Ground-truth text")
    hypothesis: str = Field(..., description="Predicted / transcribed text")
    language: str = Field(
        "auto",
        description=(
            "Language: 'auto'/'mixed' (CJK chars + non-CJK words), "
            "'zh'/'chinese' (CER), or others (WER)"
        ),
    )


class WERResponse(BaseModel):
    error_rate: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int
    hyp_length: int
    reference_normalized: str
    hypothesis_normalized: str
    metric: str


class WERAggregateResponse(BaseModel):
    error_rate: Optional[float]
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int
    hyp_length: int
    metric: str
    count: int
    valid_count: int
    skipped_count: int


class WERBatchRequest(BaseModel):
    pairs: List[WERRequest] = Field(
        ..., description="List of reference/hypothesis pairs"
    )
    include_aggregate: bool = Field(
        False,
        description="If true, also return sequence-level aggregate over all pairs.",
    )


class WERBatchResponse(BaseModel):
    results: List[WERResponse]
    aggregate: Optional[WERAggregateResponse] = None


class WERAggregateRequest(BaseModel):
    results: List[Dict[str, Any]] = Field(
        ..., description="WER/CER result dictionaries to aggregate"
    )


# -------------------------------- Endpoints -------------------------------- #


@app.get("/health")
async def health():
    return {"status": "ok", "service": "wer_calc"}


@app.post("/compute_wer", response_model=WERResponse)
async def compute_wer_endpoint(req: WERRequest):
    """Compute WER or CER for a single reference/hypothesis pair."""
    try:
        result = compute_error_rate(req.reference, req.hypothesis, req.language)
        return WERResponse(**result.to_dict())
    except Exception as e:
        logger.error(f"WER computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compute_wer_batch", response_model=WERBatchResponse)
async def compute_wer_batch_endpoint(req: WERBatchRequest):
    """Compute WER/CER for a batch of pairs."""
    try:
        references = [p.reference for p in req.pairs]
        hypotheses = [p.hypothesis for p in req.pairs]
        languages = [p.language for p in req.pairs]

        results = []
        for ref, hyp, lang in zip(references, hypotheses, languages):
            r = compute_error_rate(ref, hyp, lang)
            results.append(WERResponse(**r.to_dict()))

        aggregate = None
        if req.include_aggregate:
            result_payloads = [
                result.model_dump() if hasattr(result, "model_dump") else result.dict()
                for result in results
            ]
            aggregate = WERAggregateResponse(
                **aggregate_error_rate_results(result_payloads).to_dict()
            )

        return WERBatchResponse(results=results, aggregate=aggregate)
    except Exception as e:
        logger.error(f"Batch WER computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/aggregate_wer_results", response_model=WERAggregateResponse)
async def aggregate_wer_results_endpoint(req: WERAggregateRequest):
    """Aggregate existing WER/CER result dictionaries by total reference length."""
    try:
        return WERAggregateResponse(
            **aggregate_error_rate_results(req.results).to_dict()
        )
    except Exception as e:
        logger.error(f"Aggregate WER computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------- Main ------------------------------------- #


def _extra_args(parser):
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")


runner = ServiceRunner(
    app=app,
    service_name="WER/CER",
    default_port=8011,
    extra_args=_extra_args,
)

if __name__ == "__main__":
    runner.run()
