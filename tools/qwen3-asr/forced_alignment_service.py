"""
Standalone Qwen3 Forced Aligner HTTP service.

This service is intentionally separate from the ASR service. It exposes the
Qwen3-ForcedAligner model as a dedicated FastAPI service on its own port.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR")

if _DEFAULT_CHECKPOINTS_DIR:
    os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_CHECKPOINTS_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_CHECKPOINTS_DIR)

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from qwen3_asr_service.aligner import Qwen3ForcedAligner
from qwen3_asr_service.query_handling import (
    QueryValidationError,
    build_query_mode_plan,
    resolve_query_answers,
)
from qwen_asr.inference.utils import normalize_audios, normalize_language_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


ALIGNER: Optional[Qwen3ForcedAligner] = None
MODEL_NAME: Optional[str] = None
MODEL_DTYPE: Optional[str] = None
MODEL_DEVICE_MAP: Optional[str] = None
MAX_QWEN3_FORCE_ALIGN_SECONDS = 180.0
ALIGNER_INFERENCE_LOCK = threading.Lock()


class EncodedAudio(BaseModel):
    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format (wav, flac, mp3, etc.)")


class PreviewRequest(BaseModel):
    text: str = Field(..., description="Transcript to tokenize for alignment")
    language: str = Field(..., description="Explicit language name, e.g. Chinese")


class PreviewResponse(BaseModel):
    status: str = "success"
    language: str
    num_units: int
    units: List[str]
    encoded_text: str


class AlignRequest(BaseModel):
    audio: Union[EncodedAudio, str] = Field(
        ..., description="Base64-encoded audio dict or path/URL/data URI string"
    )
    text: str = Field(..., description="Transcript to align")
    language: str = Field(..., description="Explicit language name, e.g. Chinese")
    word_level: bool = Field(
        False,
        description=(
            "If true, report Qwen alignment units as word-level timestamps for "
            "space-delimited languages and character-level timestamps for Chinese."
        ),
    )
    separate_punctuation: bool = Field(
        False,
        description="Unsupported for Qwen3 aligner. Must remain false.",
    )
    debug: bool = Field(
        False,
        description="Unsupported for Qwen3 aligner. Must remain false.",
    )
    debug_dir: Optional[str] = Field(
        None,
        description="Unsupported for Qwen3 aligner.",
    )
    queries: Optional[List[str]] = Field(
        None,
        description=(
            "Optional tagged region queries. When present, Qwen3 computes query "
            "answers using query-boundary-aware tokenization and also reports whether "
            "that tokenization matches default Qwen tokenization for the plain text."
        ),
    )


class BatchAlignRequest(BaseModel):
    items: List[AlignRequest]


class AlignmentItem(BaseModel):
    char: str
    start: float
    end: float
    is_punctuation: Optional[bool] = None


class AlignResponse(BaseModel):
    status: str = "success"
    backend: str = "qwen3"
    language: str
    level: str = "token"
    num_chars: int
    alignments: List[AlignmentItem]
    answers: Optional[List[Dict[str, float]]] = None
    query_tokenization_matches_default: Optional[bool] = None


class BatchAlignResponse(BaseModel):
    results: List[AlignResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model: Optional[str]


class InfoResponse(BaseModel):
    service: str
    model: Optional[str]
    dtype: Optional[str]
    device_map: Optional[str]
    cache_dir: str
    supported_languages: List[str]


def _parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Supported: {sorted(mapping)}"
        )
    return mapping[dtype_name]


def _ensure_model_loaded() -> Qwen3ForcedAligner:
    if ALIGNER is None:
        raise HTTPException(status_code=503, detail="Qwen3 aligner not ready")
    return ALIGNER


def _supported_languages(aligner: Qwen3ForcedAligner) -> List[str]:
    raw = aligner.get_supported_languages() or []
    return sorted(normalize_language_name(x) for x in raw)


def _validate_language(language: str) -> str:
    aligner = _ensure_model_loaded()
    canonical = normalize_language_name(language)
    supported = {x.lower() for x in _supported_languages(aligner)}
    if canonical.lower() not in supported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported language '{language}'. Supported: "
                f"{_supported_languages(aligner)}"
            ),
        )
    return canonical


def _reject_unsupported_options(req: AlignRequest) -> None:
    unsupported = []
    if req.separate_punctuation:
        unsupported.append("separate_punctuation")
    if req.debug:
        unsupported.append("debug")
    if req.debug_dir:
        unsupported.append("debug_dir")
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=("Qwen3 aligner does not support: " + ", ".join(unsupported)),
        )


def _decode_audio(
    audio_input: Union[EncodedAudio, str],
) -> Union[str, Tuple[np.ndarray, int]]:
    if isinstance(audio_input, str):
        return audio_input
    audio_bytes = base64.b64decode(audio_input.data)
    with io.BytesIO(audio_bytes) as buf:
        waveform, sr = sf.read(buf, dtype="float32", always_2d=False)
    return np.asarray(waveform, dtype=np.float32), int(sr)


def _audio_duration_sec(audio_input: Union[str, Tuple[np.ndarray, int]]) -> Optional[float]:
    if isinstance(audio_input, tuple):
        waveform, sr = audio_input
        if sr <= 0:
            return None
        return float(np.asarray(waveform).shape[0]) / float(sr)
    try:
        path = Path(audio_input)
        if path.exists():
            info = sf.info(str(path))
            if info.samplerate > 0:
                return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def _validate_audio_duration(audio_input: Union[str, Tuple[np.ndarray, int]]) -> None:
    duration_sec = _audio_duration_sec(audio_input)
    if duration_sec is None:
        return
    if duration_sec > MAX_QWEN3_FORCE_ALIGN_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Qwen3 forced aligner input is too long: "
                f"{duration_sec:.2f}s > {MAX_QWEN3_FORCE_ALIGN_SECONDS:.2f}s. "
                "Split the audio before alignment."
            ),
        )


def _preview_units(text: str, language: str) -> Tuple[List[str], str]:
    aligner = _ensure_model_loaded()
    canonical = _validate_language(language)
    units, encoded_text = aligner.aligner_processor.encode_timestamp(text, canonical)
    return units, encoded_text


def _response_level(language: str, *, word_level: bool) -> str:
    if not word_level:
        return "token"
    if language.lower() in {"chinese", "cantonese"}:
        return "char"
    return "word"


def _result_to_items(result) -> List[AlignmentItem]:
    return [
        AlignmentItem(
            char=item.text,
            start=float(item.start_time),
            end=float(item.end_time),
            is_punctuation=None,
        )
        for item in result
    ]


def _serialize_align_response(response: AlignResponse) -> Dict[str, Any]:
    payload = response.model_dump()
    if response.answers is None:
        payload.pop("answers", None)
    if response.query_tokenization_matches_default is None:
        payload.pop("query_tokenization_matches_default", None)
    return payload


def _align_one(req: AlignRequest) -> AlignResponse:
    return _align_many([req])[0]


def _align_many(reqs: List[AlignRequest]) -> List[AlignResponse]:
    if not reqs:
        return []

    decoded: List[Dict[str, Any]] = []
    try:
        for req in reqs:
            _reject_unsupported_options(req)
            audio_input = _decode_audio(req.audio)
            _validate_audio_duration(audio_input)
            decoded.append({"request": req, "audio_input": audio_input})

        aligner = _ensure_model_loaded()
        if not hasattr(aligner, "_align_prepared"):
            return [_align_one_unbatched(req) for req in reqs]

        prepared: List[Dict[str, Any]] = []
        for item in decoded:
            req = item["request"]
            canonical = _validate_language(req.language)

            response_fields: Dict[str, Any] = {}
            if req.queries is None:
                word_list, aligner_input_text = (
                    aligner.aligner_processor.encode_timestamp(req.text, canonical)
                )
                plan = None
            else:
                plan = build_query_mode_plan(
                    req.text,
                    req.queries,
                    tokenize_text=lambda text, lang=canonical: (
                        aligner.aligner_processor.tokenize_text(text, lang)
                    ),
                )
                word_list = list(plan.query_mode_units)
                aligner_input_text = aligner.aligner_processor.build_aligner_input_text(
                    word_list
                )
                response_fields["query_tokenization_matches_default"] = (
                    plan.tokenization_matches_default
                )

            prepared.append(
                {
                    "request": req,
                    "audio_input": item["audio_input"],
                    "canonical": canonical,
                    "word_list": word_list,
                    "aligner_input_text": aligner_input_text,
                    "plan": plan,
                    "response_fields": response_fields,
                }
            )

        with ALIGNER_INFERENCE_LOCK:
            results = aligner._align_prepared(
                audios=normalize_audios([item["audio_input"] for item in prepared]),
                word_lists=[item["word_list"] for item in prepared],
                aligner_input_texts=[item["aligner_input_text"] for item in prepared],
            )
    except QueryValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Qwen3 alignment failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    responses: List[AlignResponse] = []
    for item, result in zip(prepared, results):
        response_fields = dict(item["response_fields"])
        plan = item["plan"]
        if plan is not None:
            alignments_payload = [
                {
                    "char": aligned_item.text,
                    "start": float(aligned_item.start_time),
                    "end": float(aligned_item.end_time),
                }
                for aligned_item in result
            ]
            response_fields["answers"] = resolve_query_answers(
                plan,
                alignments_payload,
            )
        items = _result_to_items(result)
        responses.append(
            AlignResponse(
                language=item["canonical"],
                level=_response_level(
                    item["canonical"],
                    word_level=item["request"].word_level,
                ),
                num_chars=len(items),
                alignments=items,
                **response_fields,
            )
        )
    return responses


def _align_one_unbatched(req: AlignRequest) -> AlignResponse:
    _reject_unsupported_options(req)
    audio_input = _decode_audio(req.audio)
    _validate_audio_duration(audio_input)
    canonical = _validate_language(req.language)
    aligner = _ensure_model_loaded()

    try:
        response_fields: Dict[str, Any] = {}
        if req.queries is None:
            with ALIGNER_INFERENCE_LOCK:
                result = aligner.align(
                    audio=audio_input,
                    text=req.text,
                    language=canonical,
                )[0]
        else:
            plan = build_query_mode_plan(
                req.text,
                req.queries,
                tokenize_text=lambda text: aligner.aligner_processor.tokenize_text(
                    text, canonical
                ),
            )
            with ALIGNER_INFERENCE_LOCK:
                result = aligner.align_with_units(
                    audio=audio_input,
                    units=plan.query_mode_units,
                )
            alignments_payload = [
                {
                    "char": item.text,
                    "start": float(item.start_time),
                    "end": float(item.end_time),
                }
                for item in result
            ]
            response_fields["answers"] = resolve_query_answers(plan, alignments_payload)
            response_fields["query_tokenization_matches_default"] = (
                plan.tokenization_matches_default
            )
    except QueryValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Qwen3 alignment failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    items = _result_to_items(result)
    return AlignResponse(
        language=canonical,
        level=_response_level(canonical, word_level=req.word_level),
        num_chars=len(items),
        alignments=items,
        **response_fields,
    )


app = FastAPI(
    title="Qwen3 Forced Aligner Service",
    description="Standalone HTTP service for Qwen/Qwen3-ForcedAligner-0.6B.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if ALIGNER is not None else "not_ready",
        service="qwen3-aligner",
        model_loaded=ALIGNER is not None,
        model=MODEL_NAME,
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    aligner = _ensure_model_loaded()
    return InfoResponse(
        service="qwen3-aligner",
        model=MODEL_NAME,
        dtype=MODEL_DTYPE,
        device_map=MODEL_DEVICE_MAP,
        cache_dir=os.environ["HF_HUB_CACHE"],
        supported_languages=_supported_languages(aligner),
    )


@app.get("/supported_languages", response_model=List[str])
def supported_languages() -> List[str]:
    return _supported_languages(_ensure_model_loaded())


@app.post("/preview_units", response_model=PreviewResponse)
def preview_units(req: PreviewRequest) -> PreviewResponse:
    units, encoded_text = _preview_units(req.text, req.language)
    canonical = _validate_language(req.language)
    return PreviewResponse(
        language=canonical,
        num_units=len(units),
        units=units,
        encoded_text=encoded_text,
    )


@app.post("/align", response_model=AlignResponse)
def align(req: AlignRequest) -> JSONResponse:
    return JSONResponse(content=_serialize_align_response(_align_one(req)))


@app.post("/align_batch", response_model=BatchAlignResponse)
def align_batch(req: BatchAlignRequest) -> JSONResponse:
    return JSONResponse(
        content={
            "results": [
                _serialize_align_response(item) for item in _align_many(req.items)
            ]
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 Forced Aligner Service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8028)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Model name or local path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=_DEFAULT_CHECKPOINTS_DIR,
        required=_DEFAULT_CHECKPOINTS_DIR is None,
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda:0",
        help="Transformers device_map value",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype used when loading the model",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="",
        help="Optional attention implementation, e.g. flash_attention_2",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    dtype = _parse_dtype(args.dtype)
    kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    global ALIGNER, MODEL_NAME, MODEL_DTYPE, MODEL_DEVICE_MAP
    MODEL_NAME = args.model
    MODEL_DTYPE = args.dtype
    MODEL_DEVICE_MAP = args.device_map

    logger.info(
        "Loading Qwen3 forced aligner model=%s dtype=%s device_map=%s cache_dir=%s",
        args.model,
        args.dtype,
        args.device_map,
        args.cache_dir,
    )
    ALIGNER = Qwen3ForcedAligner.from_pretrained(args.model, **kwargs)
    logger.info("Qwen3 forced aligner ready")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
