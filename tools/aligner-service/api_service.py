"""
Unified aligner gateway service.

Routes alignment requests to the Qwen3 forced aligner with language-code
normalization, query forwarding and input-duration validation.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import httpx
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

QWEN3_URL: Optional[str] = None
MAX_QWEN3_FORCE_ALIGN_SECONDS = 180.0

QWEN3_LANGUAGE_CODE_TO_NAME: Dict[str, str] = {
    "ara": "Arabic",
    "ces": "Czech",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "fas": "Persian",
    "fil": "Filipino",
    "fin": "Finnish",
    "fra": "French",
    "hin": "Hindi",
    "hun": "Hungarian",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "kor": "Korean",
    "mkd": "Macedonian",
    "msa": "Malay",
    "nld": "Dutch",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "spa": "Spanish",
    "swe": "Swedish",
    "tha": "Thai",
    "tur": "Turkish",
    "vie": "Vietnamese",
    "yue": "Cantonese",
    "zho": "Chinese",
}
QWEN3_LANGUAGE_NAME_TO_CODE: Dict[str, str] = {
    name.lower(): code for code, name in QWEN3_LANGUAGE_CODE_TO_NAME.items()
}
QWEN3_LANGUAGE_NAME_TO_CODE.update({"en": "eng", "zh": "zho"})
QWEN3_LANGUAGE_ALIASES_TO_NAME: Dict[str, str] = {
    **{code: name for code, name in QWEN3_LANGUAGE_CODE_TO_NAME.items()},
    **{name.lower(): name for code, name in QWEN3_LANGUAGE_CODE_TO_NAME.items()},
    "en": "English",
    "zh": "Chinese",
    "chi": "Chinese",
    "cze": "Czech",
    "dut": "Dutch",
    "fre": "French",
    "ger": "German",
    "gre": "Greek",
    "mac": "Macedonian",
    "per": "Persian",
    "rum": "Romanian",
}

_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)

_CJK_RANGES = (
    *_HAN_RANGES,
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x31F0, 0x31FF),
    (0xAC00, 0xD7AF),
)


@dataclass(frozen=True)
class BackendResolution:
    backend: str
    reason: str
    inferred_language: Optional[str] = None


class EncodedAudio(BaseModel):
    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format (wav, flac, etc.)")


class AlignRequest(BaseModel):
    backend: str = Field(
        "auto",
        description=("Alignment backend: 'auto' or 'qwen3'. Defaults to auto."),
    )
    audio: Union[EncodedAudio, str] = Field(
        ..., description="Base64-encoded audio dict or path/URL/data URI string"
    )
    text: str = Field(..., description="Text to align")
    language: Optional[str] = Field(
        None,
        description=(
            "Language hint. Gateway requests use ISO 639-3 language codes like "
            "'eng' or 'zho'. The gateway maps those codes to Qwen3 language "
            "names internally when routing to the qwen3 backend."
        ),
    )
    word_level: bool = Field(False)
    separate_punctuation: bool = Field(False)
    debug: bool = Field(False)
    debug_dir: Optional[str] = Field(None)
    queries: Optional[List[str]] = Field(
        None,
        description=(
            "Optional tagged region queries. Passed through to the selected backend. "
            "Qwen3 responses may also include a response-level validation flag for "
            "query-mode tokenization."
        ),
    )


class BatchAlignRequest(BaseModel):
    backend: str = Field(
        "auto",
        description=("Alignment backend: 'auto' or 'qwen3'. Defaults to auto."),
    )
    items: List[AlignRequest]


class HealthResponse(BaseModel):
    status: str
    backends: Dict[str, Optional[str]]


app = FastAPI(
    title="Aligner Gateway Service",
    description="Qwen3 alignment gateway.",
    version="1.0.0",
)


def _normalize_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in {"auto", "qwen3"}:
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown backend '{backend}'. Supported: 'auto', 'qwen3'"),
        )
    return normalized


def _backend_url(backend: str) -> str:
    normalized = _normalize_backend(backend)
    if not QWEN3_URL:
        raise HTTPException(status_code=503, detail="Qwen3 backend not configured")
    return QWEN3_URL


def _clean_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = str(language).strip()
    return value or None


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def _is_han_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in _HAN_RANGES)


def _infer_auto_language_code(text: str) -> Optional[str]:
    has_han = False
    has_latin = False
    has_other_script_letters = False

    for ch in text:
        if _is_han_char(ch):
            has_han = True
            continue
        if not ch.isalpha():
            continue
        if _is_cjk_char(ch):
            has_other_script_letters = True
            continue

        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            has_latin = True
            continue
        has_other_script_letters = True

    if has_han and not has_other_script_letters:
        return "zho"
    if has_latin and not has_han and not has_other_script_letters:
        return "eng"
    return None


def _resolve_backend(
    payload: Dict[str, Any], *, requested_backend: str, request_label: str
) -> BackendResolution:
    backend = _normalize_backend(requested_backend)
    inferred_language = None
    if backend == "auto" and _clean_language(payload.get("language")) is None:
        inferred_language = _infer_auto_language_code(str(payload.get("text") or ""))
    return BackendResolution(
        backend="qwen3", reason=f"{backend}:qwen3", inferred_language=inferred_language
    )


def _qwen3_language_name(language: Optional[str], *, request_label: str) -> str:
    value = _clean_language(language)
    if value is None:
        supported = ", ".join(sorted(QWEN3_LANGUAGE_CODE_TO_NAME))
        raise HTTPException(
            status_code=400,
            detail=(
                f"{request_label}: language is required for backend='qwen3'. "
                f"Use an ISO 639-3 code such as: {supported}"
            ),
        )

    mapped = QWEN3_LANGUAGE_ALIASES_TO_NAME.get(value.lower())
    if mapped is None:
        supported = ", ".join(sorted(QWEN3_LANGUAGE_CODE_TO_NAME))
        raise HTTPException(
            status_code=400,
            detail=(
                f"{request_label}: unsupported qwen3 language '{value}'. "
                f"Use one of the supported ISO 639-3 codes: {supported}"
            ),
        )
    return mapped


def _gateway_language_code_from_qwen3(language: Optional[str]) -> Optional[str]:
    value = _clean_language(language)
    if value is None:
        return None

    lowered = value.lower()
    if lowered in QWEN3_LANGUAGE_CODE_TO_NAME:
        return lowered

    mapped = QWEN3_LANGUAGE_NAME_TO_CODE.get(lowered)
    if mapped is None:
        raise HTTPException(
            status_code=502,
            detail=f"Qwen3 backend returned unsupported language '{value}'",
        )
    return mapped


def _normalize_align_request_payload(
    payload: Dict[str, Any], *, backend: str, request_label: str
) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["language"] = _qwen3_language_name(
        normalized.get("language"), request_label=request_label
    )
    return normalized


def _audio_duration_sec(audio_payload: Any) -> Optional[float]:
    try:
        if isinstance(audio_payload, dict) and audio_payload.get("data"):
            audio_bytes = base64.b64decode(audio_payload["data"])
            with io.BytesIO(audio_bytes) as buf:
                info = sf.info(buf)
            if info.samplerate > 0:
                return float(info.frames) / float(info.samplerate)
            return None

        if isinstance(audio_payload, str):
            value = audio_payload.strip()
            if value.startswith("data:") and "," in value:
                audio_bytes = base64.b64decode(value.split(",", 1)[1])
                with io.BytesIO(audio_bytes) as buf:
                    info = sf.info(buf)
                if info.samplerate > 0:
                    return float(info.frames) / float(info.samplerate)
                return None

            path = Path(value)
            if path.exists():
                info = sf.info(str(path))
                if info.samplerate > 0:
                    return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def _validate_qwen3_audio_duration(
    payload: Dict[str, Any], *, request_label: str
) -> None:
    duration_sec = _audio_duration_sec(payload.get("audio"))
    if duration_sec is None:
        return
    if duration_sec > MAX_QWEN3_FORCE_ALIGN_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{request_label}: Qwen3 forced aligner input is too long: "
                f"{duration_sec:.2f}s > {MAX_QWEN3_FORCE_ALIGN_SECONDS:.2f}s. "
                "Split the audio before alignment."
            ),
        )


def _prepare_align_request(
    payload: Dict[str, Any], *, requested_backend: str, request_label: str
) -> tuple[BackendResolution, str, Dict[str, Any]]:
    resolution = _resolve_backend(
        payload, requested_backend=requested_backend, request_label=request_label
    )
    prepared_payload = dict(payload)
    if (
        resolution.backend == "qwen3"
        and resolution.inferred_language is not None
        and _clean_language(prepared_payload.get("language")) is None
    ):
        prepared_payload["language"] = resolution.inferred_language

    normalized_payload = _normalize_align_request_payload(
        prepared_payload,
        backend=resolution.backend,
        request_label=request_label,
    )
    if resolution.backend == "qwen3":
        _validate_qwen3_audio_duration(
            normalized_payload, request_label=request_label
        )
    return resolution, _backend_url(resolution.backend), normalized_payload


def _normalize_alignment_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    normalized.pop("text", None)
    return normalized


def _normalize_align_response(
    payload: Dict[str, Any], *, backend: str
) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["backend"] = backend

    if backend == "qwen3":
        normalized["language"] = _gateway_language_code_from_qwen3(
            normalized.get("language")
        )

    alignments = normalized.get("alignments")
    if isinstance(alignments, list):
        normalized["alignments"] = [
            _normalize_alignment_item(item) for item in alignments
        ]

    return normalized


def _normalize_align_batch_response(
    payload: Dict[str, Any], *, backend: str
) -> Dict[str, Any]:
    normalized = dict(payload)
    results = normalized.get("results")
    if isinstance(results, list):
        normalized["results"] = [
            _normalize_align_response(item, backend=backend)
            if isinstance(item, dict)
            else item
            for item in results
        ]
    return normalized


async def _proxy_json(method: str, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.request(method, url, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503, detail=f"Backend unreachable: {exc}"
        ) from exc


async def _backend_health_ok(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{url}/health")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@app.get("/health", response_model=HealthResponse)
async def health(deep: bool = False) -> HealthResponse:
    backends = {
        "qwen3": QWEN3_URL,
    }
    if not any(backends.values()):
        raise HTTPException(
            status_code=503,
            detail="No aligner backends configured",
        )

    if not deep:
        return HealthResponse(
            status="ok",
            backends=backends,
        )

    healthy_backends: Dict[str, Optional[str]] = {}
    unhealthy_backends: List[str] = []
    for name, url in backends.items():
        if not url:
            healthy_backends[name] = None
            continue
        if await _backend_health_ok(url):
            healthy_backends[name] = url
        else:
            healthy_backends[name] = None
            unhealthy_backends.append(name)

    if unhealthy_backends:
        raise HTTPException(
            status_code=503,
            detail=("Unhealthy aligner backend(s): " + ", ".join(unhealthy_backends)),
        )

    return HealthResponse(
        status="ok",
        backends=healthy_backends,
    )


@app.post("/align")
async def align(req: AlignRequest):
    resolution, url, payload = _prepare_align_request(
        req.model_dump(exclude={"backend"}, exclude_none=True),
        requested_backend=req.backend,
        request_label="align",
    )
    logger.info("Routing align via %s (%s)", resolution.backend, resolution.reason)
    endpoint = "align"
    response = await _proxy_json("POST", f"{url}/{endpoint}", payload)
    return _normalize_align_response(response, backend=resolution.backend)


@app.post("/align_batch")
async def align_batch(req: BatchAlignRequest):
    backend = _normalize_backend(req.backend)
    if backend == "auto":
        results = []
        for index, item in enumerate(req.items):
            resolution, url, item_payload = _prepare_align_request(
                item.model_dump(exclude={"backend"}, exclude_none=True),
                requested_backend="auto",
                request_label=f"align_batch.items[{index}]",
            )
            logger.info(
                "Routing align_batch item %s via %s (%s)",
                index,
                resolution.backend,
                resolution.reason,
            )
            response = await _proxy_json("POST", f"{url}/align", item_payload)
            results.append(
                _normalize_align_response(response, backend=resolution.backend)
            )
        return {"results": results}

    url = _backend_url(backend)
    items = []
    for index, item in enumerate(req.items):
        item_payload = _normalize_align_request_payload(
            item.model_dump(exclude={"backend"}, exclude_none=True),
            backend=backend,
            request_label=f"align_batch.items[{index}]",
        )
        if backend == "qwen3":
            _validate_qwen3_audio_duration(
                item_payload, request_label=f"align_batch.items[{index}]"
            )
        items.append(item_payload)
    response = await _proxy_json("POST", f"{url}/align_batch", {"items": items})
    return _normalize_align_batch_response(response, backend=backend)


@app.get("/info")
async def info():
    return {
        "service": "aligner-gateway",
        "default_backend": "auto",
        "backends": {
            "qwen3": QWEN3_URL,
        },
    }


def main() -> None:
    global QWEN3_URL

    parser = argparse.ArgumentParser(description="Aligner Gateway Service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8029)
    parser.add_argument(
        "--ctc-url",
        type=str,
        default="",
        choices=[""],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen3-url",
        type=str,
        default="http://localhost:8028",
        help="URL of the Qwen3 aligner backend",
    )
    args = parser.parse_args()

    QWEN3_URL = args.qwen3_url or None

    logger.info("Aligner gateway starting on %s:%s", args.host, args.port)
    logger.info("  Qwen3 backend: %s", QWEN3_URL or "(not configured)")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
