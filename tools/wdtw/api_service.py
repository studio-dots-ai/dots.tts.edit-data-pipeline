"""WDTW metric HTTP service."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.common import EncodedAudio, ServiceRunner, setup_logging

try:
    from .wdtw import (
        aggregate_wdtw_results,
        alignment_to_segments,
        compute_wdtw_dur_from_segments,
        compute_wdtw_f0_from_spans,
        compute_wdtw_from_segments,
        parse_edit_st_inst,
        tokenize_text,
        unavailable_wdtw_f0_result,
    )
except ImportError:
    from wdtw import (
        aggregate_wdtw_results,
        alignment_to_segments,
        compute_wdtw_dur_from_segments,
        compute_wdtw_f0_from_spans,
        compute_wdtw_from_segments,
        parse_edit_st_inst,
        tokenize_text,
        unavailable_wdtw_f0_result,
    )


logger = setup_logging(__name__)

CONFIGURED_ALIGNER_URL: Optional[str] = None
CONFIGURED_ALIGNER_BACKEND: Optional[str] = None
CONFIGURED_F0_URL: Optional[str] = None

app = FastAPI(
    title="WDTW Metric Service",
    description="Compute Word-level Dynamic Time Warping for speech editing.",
    version="1.1.0",
)


class AlignmentItem(BaseModel):
    char: Optional[str] = None
    word: Optional[str] = None
    text: Optional[str] = None
    token: Optional[str] = None
    start: float
    end: float
    is_punctuation: Optional[bool] = None


class WDTWRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_transcript: Optional[str] = Field(None, description="Source transcript")
    target_transcript: Optional[str] = Field(None, description="Target transcript")
    edit_st_inst: Optional[str] = Field(
        None,
        description=(
            "Optional structured edit instruction. If source/target transcripts "
            "are omitted, they are derived from this field."
        ),
    )
    preservation_st_inst: Optional[str] = Field(
        None,
        description=(
            "Structured edit instruction used only by WDTW-Dur and WDTW-F0. "
            "It does not alter legacy WDTW selection."
        ),
    )
    source_alignments: Optional[List[AlignmentItem]] = Field(
        None, description="Precomputed source word alignments"
    )
    target_alignments: Optional[List[AlignmentItem]] = Field(
        None, description="Precomputed target word alignments"
    )
    source_audio: Optional[Union[EncodedAudio, str]] = Field(
        None, description="Source audio if alignments are not provided"
    )
    target_audio: Optional[Union[EncodedAudio, str]] = Field(
        None, description="Target audio if alignments are not provided"
    )
    language: str = Field("eng", description="Gateway language code")
    mode: str = Field(
        "preserved", description="Metric mode: preserved or all"
    )
    word_mismatch_penalty: float = Field(
        1.0,
        ge=0.0,
        description="Penalty added when DTW aligns different words",
    )
    align_timeout: float = Field(120.0, gt=0.0, description="Aligner request timeout")
    f0_timeout: float = Field(120.0, gt=0.0, description="F0 service request timeout")
    include_segments: bool = Field(
        True, description="Return selected normalized word-duration segments"
    )


class WDTWResponse(BaseModel):
    wdtw: Optional[float]
    distance: float
    normalizer: float
    mode: str
    status: str
    num_source_segments: int
    num_target_segments: int
    num_preserved_pairs: int
    preserved_pairs: List[List[int]]
    source_segments: Optional[List[Dict[str, Any]]] = None
    target_segments: Optional[List[Dict[str, Any]]] = None
    aligner_backend: Optional[str] = None
    wdtw_dur: Optional[float] = None
    wdtw_dur_result: Optional[Dict[str, Any]] = None
    wdtw_f0: Optional[float] = None
    wdtw_f0_result: Optional[Dict[str, Any]] = None


class WDTWAggregateResponse(BaseModel):
    wdtw: Optional[float]
    distance: float
    normalizer: float
    mode: str
    status: str
    count: int
    valid_count: int
    skipped_count: int
    skipped_no_segments: int
    skipped_missing: int


class WDTWBatchRequest(BaseModel):
    items: List[WDTWRequest]
    include_aggregate: bool = Field(
        False,
        description="If true, also return distance/normalizer aggregate over results.",
    )


class WDTWBatchResponse(BaseModel):
    results: List[WDTWResponse]
    aggregate: Optional[WDTWAggregateResponse] = None


class WDTWAggregateRequest(BaseModel):
    results: List[Dict[str, Any]]


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return dict(value)


def _alignment_payload(items: List[AlignmentItem]) -> List[Dict[str, Any]]:
    return [_model_to_dict(item) for item in items]


def _infer_alignment_level(items: List[Dict[str, Any]]) -> Optional[str]:
    if not items:
        return None
    word_like = 0
    char_like = 0
    for item in items:
        text = item.get("word") or item.get("text") or item.get("token")
        char = item.get("char")
        if isinstance(text, str) and text:
            word_like += 1
        elif isinstance(char, str) and char:
            if len(char) == 1:
                char_like += 1
            else:
                word_like += 1
    if word_like and not char_like:
        return "word"
    if char_like and not word_like:
        return "char"
    return None


def _audio_payload(audio: Union[EncodedAudio, str]) -> Union[str, Dict[str, Any]]:
    if isinstance(audio, str):
        return audio
    return _model_to_dict(audio)


def _resolve_transcripts(req: WDTWRequest) -> Dict[str, str]:
    source_transcript = req.source_transcript
    target_transcript = req.target_transcript

    if source_transcript is None or target_transcript is None:
        if not req.edit_st_inst:
            raise ValueError(
                "Provide source_transcript/target_transcript, or provide edit_st_inst "
                "so they can be derived."
            )
        parsed = parse_edit_st_inst(req.edit_st_inst)
        if source_transcript is None:
            source_transcript = parsed.source_text
        if target_transcript is None:
            target_transcript = parsed.target_text

    if req.edit_st_inst:
        parsed = parse_edit_st_inst(req.edit_st_inst)
        if tokenize_text(parsed.source_text) != tokenize_text(source_transcript):
            raise ValueError(
                "edit_st_inst source text does not match source_transcript tokens"
            )
        if tokenize_text(parsed.target_text) != tokenize_text(target_transcript):
            raise ValueError(
                "edit_st_inst target text does not match target_transcript tokens"
            )

    return {"source": source_transcript, "target": target_transcript}


def _call_aligner(
    *,
    audio: Union[EncodedAudio, str],
    transcript: str,
    language: str,
    aligner_url: str,
    backend: str,
    timeout: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "backend": backend,
        "audio": _audio_payload(audio),
        "text": transcript,
        "language": language,
        "word_level": True,
    }

    response = requests.post(
        f"{aligner_url.rstrip('/')}/align",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    alignments = data.get("alignments")
    if not isinstance(alignments, list):
        raise RuntimeError(f"Aligner response missing alignments: {data}")
    backend_used = data.get("backend")
    if backend_used is not None and not isinstance(backend_used, str):
        raise RuntimeError(f"Aligner response has invalid backend field: {data}")
    level_used = data.get("level")
    if level_used not in {"word", "char"}:
        raise RuntimeError(
            "WDTW requires word-level alignments; aligner returned "
            f"level={level_used!r}: {data}"
        )
    return {
        "alignments": alignments,
        "backend": backend_used or backend,
        "level": level_used,
    }


def _resolve_alignments(req: WDTWRequest, transcripts: Dict[str, str]) -> Dict[str, Any]:
    source_alignments = req.source_alignments
    target_alignments = req.target_alignments
    if source_alignments is not None and target_alignments is not None:
        source_payload = _alignment_payload(source_alignments)
        target_payload = _alignment_payload(target_alignments)
        source_level = _infer_alignment_level(source_payload)
        target_level = _infer_alignment_level(target_payload)
        if source_level and target_level and source_level != target_level:
            raise RuntimeError(
                "Cached alignment level mismatch between source and target: "
                f"{source_level} vs {target_level}"
            )
        return {
            "source": source_payload,
            "target": target_payload,
            "backend": None,
        }

    source_audio = req.source_audio
    target_audio = req.target_audio
    if (source_alignments is None and source_audio is None) or (
        target_alignments is None and target_audio is None
    ):
        raise ValueError(
            "Provide source/target alignments, or audio for each side whose "
            "alignment is missing."
        )

    if not CONFIGURED_ALIGNER_URL or not CONFIGURED_ALIGNER_BACKEND:
        raise ValueError("aligner dependency is not configured")
    source_payload = (
        _alignment_payload(source_alignments)
        if source_alignments is not None
        else None
    )
    target_payload = (
        _alignment_payload(target_alignments)
        if target_alignments is not None
        else None
    )
    source = (
        {
            "alignments": source_payload,
            "backend": None,
            "level": _infer_alignment_level(source_payload or []),
        }
        if source_alignments is not None
        else _call_aligner(
            audio=source_audio,
            transcript=transcripts["source"],
            language=req.language,
            aligner_url=CONFIGURED_ALIGNER_URL,
            backend=CONFIGURED_ALIGNER_BACKEND,
            timeout=req.align_timeout,
        )
    )
    target = (
        {
            "alignments": target_payload,
            "backend": None,
            "level": _infer_alignment_level(target_payload or []),
        }
        if target_alignments is not None
        else _call_aligner(
            audio=target_audio,
            transcript=transcripts["target"],
            language=req.language,
            aligner_url=CONFIGURED_ALIGNER_URL,
            backend=CONFIGURED_ALIGNER_BACKEND,
            timeout=req.align_timeout,
        )
    )
    backend_used = source["backend"] or target["backend"]
    if (
        source["backend"] is not None
        and target["backend"] is not None
        and target["backend"] != backend_used
    ):
        raise RuntimeError(
            "Aligner backend mismatch between source and target audio: "
            f"{backend_used} vs {target['backend']}"
        )
    level_used = source["level"] or target["level"]
    if (
        source["level"] is not None
        and target["level"] is not None
        and target["level"] != level_used
    ):
        raise RuntimeError(
            "Aligner level mismatch between source and target audio: "
            f"{level_used} vs {target['level']}"
        )
    return {
        "source": source["alignments"],
        "target": target["alignments"],
        "backend": backend_used,
    }


def _extract_http_error_detail(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    try:
        data = response.json()
    except ValueError:
        body = response.text.strip()
        return body or str(exc)
    if isinstance(data, dict):
        detail = data.get("detail") or data.get("message")
        if detail:
            return str(detail)
    return str(data)


def _validate_preservation_instruction(
    preservation_st_inst: Optional[str],
    transcripts: Dict[str, str],
) -> None:
    if not preservation_st_inst:
        return
    parsed = parse_edit_st_inst(preservation_st_inst)
    if tokenize_text(parsed.source_text) != tokenize_text(transcripts["source"]):
        raise ValueError(
            "preservation_st_inst source text does not match source_transcript tokens"
        )
    if tokenize_text(parsed.target_text) != tokenize_text(transcripts["target"]):
        raise ValueError(
            "preservation_st_inst target text does not match target_transcript tokens"
        )


def _call_f0(
    *,
    audio: Union[EncodedAudio, str],
    segments: List[Any],
    timeout: float,
) -> List[Dict[str, Any]]:
    if not CONFIGURED_F0_URL:
        raise RuntimeError("f0 dependency is not configured")
    spans = [
        {
            "start": segment.start,
            "end": segment.end,
            "label": f"{index}:{segment.word}",
        }
        for index, segment in enumerate(segments)
    ]
    response = requests.post(
        f"{CONFIGURED_F0_URL.rstrip('/')}/analyze",
        json={"audio": _audio_payload(audio), "spans": spans},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("spans")
    if not isinstance(results, list):
        raise RuntimeError(f"F0 response missing spans: {data}")
    return results


def _compute(req: WDTWRequest) -> WDTWResponse:
    transcripts = _resolve_transcripts(req)
    alignments = _resolve_alignments(req, transcripts)
    preservation_st_inst = req.preservation_st_inst or req.edit_st_inst
    _validate_preservation_instruction(preservation_st_inst, transcripts)
    source_segments_all = alignment_to_segments(alignments["source"])
    target_segments_all = alignment_to_segments(alignments["target"])
    result = compute_wdtw_from_segments(
        source_segments_all=source_segments_all,
        target_segments_all=target_segments_all,
        source_transcript=transcripts["source"],
        target_transcript=transcripts["target"],
        edit_st_inst=req.edit_st_inst,
        mode=req.mode,
        word_mismatch_penalty=req.word_mismatch_penalty,
    )
    duration_result = compute_wdtw_dur_from_segments(
        source_segments_all=source_segments_all,
        target_segments_all=target_segments_all,
        source_transcript=transcripts["source"],
        target_transcript=transcripts["target"],
        preservation_st_inst=preservation_st_inst,
        word_mismatch_penalty=req.word_mismatch_penalty,
    )
    f0_selection = compute_wdtw_from_segments(
        source_segments_all=source_segments_all,
        target_segments_all=target_segments_all,
        source_transcript=transcripts["source"],
        target_transcript=transcripts["target"],
        edit_st_inst=preservation_st_inst,
        mode="preserved",
        word_mismatch_penalty=req.word_mismatch_penalty,
    )
    if not f0_selection.source_segments:
        f0_result = unavailable_wdtw_f0_result(
            "no_preserved_words",
            "No preserved words are eligible for F0 comparison",
        )
    elif req.source_audio is None or req.target_audio is None:
        f0_result = unavailable_wdtw_f0_result(
            "unavailable",
            "source_audio and target_audio are required for WDTW-F0",
            eligible_word_count=len(f0_selection.source_segments),
        )
    elif not CONFIGURED_F0_URL:
        raise RuntimeError("F0 dependency is required when comparing audio")
    else:
        source_f0 = _call_f0(
            audio=req.source_audio,
            segments=f0_selection.source_segments,
            timeout=req.f0_timeout,
        )
        target_f0 = _call_f0(
            audio=req.target_audio,
            segments=f0_selection.target_segments,
            timeout=req.f0_timeout,
        )
        f0_result = compute_wdtw_f0_from_spans(
            source_segments=f0_selection.source_segments,
            target_segments=f0_selection.target_segments,
            source_f0_spans=source_f0,
            target_f0_spans=target_f0,
        )

    data = result.to_dict(include_segments=req.include_segments)
    data["aligner_backend"] = alignments["backend"]
    data["wdtw_dur"] = duration_result.wdtw_dur
    data["wdtw_dur_result"] = duration_result.to_dict(
        include_segments=req.include_segments
    )
    data["wdtw_f0"] = f0_result.wdtw_f0
    data["wdtw_f0_result"] = f0_result.to_dict(include_words=req.include_segments)
    return WDTWResponse(**data)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "wdtw",
        "f0_configured": bool(CONFIGURED_F0_URL),
    }


@app.get("/info")
async def info() -> Dict[str, Any]:
    return {
        "service": "wdtw",
        "version": "1.1.0",
        "default_port": 8032,
        "default_mode": "preserved",
        "supported_modes": ["preserved", "all"],
        "metrics": ["wdtw", "wdtw_dur", "wdtw_f0"],
    }


@app.post("/compute_wdtw", response_model=WDTWResponse)
async def compute_wdtw_endpoint(req: WDTWRequest) -> WDTWResponse:
    try:
        return _compute(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(status_code=status_code, detail=_extract_http_error_detail(exc))
    except requests.RequestException as exc:
        logger.error("WDTW dependency request failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("WDTW computation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/compute_wdtw_batch", response_model=WDTWBatchResponse)
async def compute_wdtw_batch_endpoint(req: WDTWBatchRequest) -> WDTWBatchResponse:
    try:
        results = [_compute(item) for item in req.items]
        aggregate = None
        if req.include_aggregate:
            result_payloads = [
                result.model_dump() if hasattr(result, "model_dump") else result.dict()
                for result in results
            ]
            aggregate = WDTWAggregateResponse(
                **aggregate_wdtw_results(result_payloads).to_dict()
            )
        return WDTWBatchResponse(results=results, aggregate=aggregate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(status_code=status_code, detail=_extract_http_error_detail(exc))
    except requests.RequestException as exc:
        logger.error("WDTW batch dependency request failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Batch WDTW computation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/aggregate_wdtw_results", response_model=WDTWAggregateResponse)
async def aggregate_wdtw_results_endpoint(req: WDTWAggregateRequest) -> WDTWAggregateResponse:
    try:
        return WDTWAggregateResponse(**aggregate_wdtw_results(req.results).to_dict())
    except Exception as exc:
        logger.error("Aggregate WDTW computation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


def _extra_args(parser):
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument(
        "--aligner-url",
        required=True,
        help="Aligner gateway used when a request does not include alignments",
    )
    parser.add_argument(
        "--aligner-backend",
        required=True,
        choices=("qwen3", "auto"),
        help="Backend selected on the configured aligner gateway",
    )
    parser.add_argument(
        "--f0-url",
        default=None,
        help="Optional F0 service used for WDTW-F0",
    )


def _configure(args) -> None:
    global CONFIGURED_ALIGNER_BACKEND, CONFIGURED_ALIGNER_URL, CONFIGURED_F0_URL
    CONFIGURED_ALIGNER_URL = str(args.aligner_url).rstrip("/")
    CONFIGURED_ALIGNER_BACKEND = str(args.aligner_backend)
    CONFIGURED_F0_URL = (
        str(args.f0_url).rstrip("/") if getattr(args, "f0_url", None) else None
    )


runner = ServiceRunner(
    app=app,
    service_name="WDTW",
    default_port=8032,
    extra_args=_extra_args,
    on_startup=_configure,
)


if __name__ == "__main__":
    runner.run()
