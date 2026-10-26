"""Repo-native ASR gateway with per-backend request buffering.

The gateway exposes one stable contract for pipeline code:

  - GET /health
  - GET /info
  - POST /transcribe
  - POST /transcribe_batch

Backend model services are started separately. This process only routes
requests to configured backend URLs and normalizes responses to ``text`` and
``language`` fields.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = {"qwen3-asr"}
DEFAULT_BACKEND = "qwen3-asr"
ASR_BACKENDS: Dict[str, Any] = {}
REQUEST_BUFFERS: Dict[str, "RequestBuffer"] = {}


@dataclass
class BufferedRequest:
    """A single buffered ASR request."""

    audio: Any
    language: Optional[str]
    future: asyncio.Future


class RequestBuffer:
    """Async buffer that batches requests for exactly one backend."""

    def __init__(
        self,
        backend: Any,
        *,
        max_batch_size: int = 16,
        max_wait_ms: float = 500,
    ) -> None:
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[BufferedRequest] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._consumer())
        logger.info(
            "Request buffer started for %s (max_batch=%s, max_wait=%sms)",
            self.backend.backend_name,
            self.max_batch_size,
            self.max_wait_ms,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def submit(self, audio: Any, language: Optional[str] = None) -> Any:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._queue.put(BufferedRequest(audio=audio, language=language, future=future))
        return await future

    async def _consumer(self) -> None:
        while True:
            batch: List[BufferedRequest] = []
            try:
                first = await self._queue.get()
                batch.append(first)
                deadline = time.monotonic() + self.max_wait_ms / 1000.0

                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break

                audios = [request.audio for request in batch]
                languages = [request.language for request in batch]
                logger.info("Processing %s ASR requests with %s", len(batch), self.backend.backend_name)
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    self.backend.transcribe_batch,
                    audios,
                    languages,
                )

                for request, result in zip(batch, results):
                    if not request.future.done():
                        request.future.set_result(result)
            except asyncio.CancelledError:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(RuntimeError("Service shutting down"))
                raise
            except Exception as exc:
                logger.error("Batch inference error: %s", exc, exc_info=True)
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)


def decode_audio_input(audio_input: dict) -> Tuple[np.ndarray, int]:
    """Decode base64-encoded audio to ``(numpy_array, sample_rate)``."""

    audio_bytes = base64.b64decode(audio_input["data"])
    with io.BytesIO(audio_bytes) as buffer:
        wav, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sample_rate)


class AudioInput(BaseModel):
    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format (wav, mp3, flac, etc.)")


class TranscribeRequest(BaseModel):
    audio: AudioInput = Field(..., description="Base64-encoded audio")
    language: Optional[str] = Field(
        None,
        description="Language hint (e.g. 'Chinese', 'English'). None = auto",
    )
    backend: Optional[str] = Field(None, description="ASR backend key")


class TranscribeResponse(BaseModel):
    text: str
    language: Optional[str] = None


class TranscribeBatchRequest(BaseModel):
    items: List[TranscribeRequest]
    backend: Optional[str] = Field(None, description="Default backend for batch items")


class TranscribeBatchResponse(BaseModel):
    results: List[TranscribeResponse]


_cli_args = None


def _model_dump(model: BaseModel) -> dict:
    dump = getattr(model, "model_dump", None)
    if dump is not None:
        return dump()
    return model.dict()


def _parse_backend_urls(entries: List[str], *, default_backend: str) -> Dict[str, str]:
    urls: Dict[str, str] = {}
    for entry in entries:
        text = str(entry).strip()
        if not text:
            continue
        if "=" in text:
            backend, url = text.split("=", 1)
            backend = backend.strip()
        else:
            backend, url = default_backend, text
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported ASR backend {backend!r}")
        urls[backend] = url.strip().rstrip("/")
    if not urls:
        urls[default_backend] = "http://127.0.0.1:8033"
    return urls


def _build_backend(backend_name: str, *, service_url: str, timeout: float) -> Any:
    if backend_name == "qwen3-asr":
        from backends.qwen3_asr import Qwen3ASRHTTPBackend

        return Qwen3ASRHTTPBackend(
            backend_name=backend_name,
            service_url=service_url,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported ASR backend {backend_name!r}")


def _build_qwen_transformers_backend(args: argparse.Namespace) -> Any:
    from backends.qwen3_transformers import Qwen3ASRTransformersBackend

    return Qwen3ASRTransformersBackend(
        backend_name="qwen3-asr",
        model_path=args.qwen_transformers_model,
        device=args.qwen_transformers_device,
        dtype_name=args.qwen_transformers_dtype,
        max_new_tokens=args.qwen_transformers_max_new_tokens,
    )


def _load_backends(args: argparse.Namespace) -> Dict[str, Any]:
    direct_qwen_model = str(args.qwen_transformers_model or "").strip()
    if direct_qwen_model:
        backend_urls = (
            _parse_backend_urls(
                args.backend_url,
                default_backend=args.default_backend,
            )
            if args.backend_url
            else {}
        )
        if "qwen3-asr" in backend_urls:
            raise ValueError(
                "Configure either --qwen-transformers-model or a qwen3-asr "
                "--backend-url, not both"
            )
    else:
        backend_urls = _parse_backend_urls(
            args.backend_url,
            default_backend=args.default_backend,
        )

    backends: Dict[str, Any] = {}
    if direct_qwen_model:
        backend = _build_qwen_transformers_backend(args)
        backend.load()
        backends["qwen3-asr"] = backend
    for backend_name, service_url in backend_urls.items():
        backend = _build_backend(
            backend_name,
            service_url=service_url,
            timeout=args.backend_timeout,
        )
        _wait_for_backend(backend, timeout_sec=args.backend_startup_timeout)
        backends[backend_name] = backend

    if args.default_backend not in backends:
        raise ValueError(
            f"default backend {args.default_backend!r} is not configured"
        )
    return backends


def _wait_for_backend(backend: Any, *, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while True:
        attempt += 1
        try:
            backend.load()
            return
        except Exception as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"ASR backend {backend.backend_name!r} did not become ready "
                    f"within {timeout_sec:.1f}s"
                ) from exc
            logger.info(
                "Waiting for ASR backend %s at %s (attempt %s): %s",
                backend.backend_name,
                backend.model_path,
                attempt,
                exc,
            )
            time.sleep(min(5.0, max(0.5, remaining)))


def _resolve_backend_name(requested: Optional[str]) -> str:
    backend_name = (requested or DEFAULT_BACKEND).strip()
    if backend_name not in REQUEST_BUFFERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ASR backend {backend_name!r}; available: {sorted(REQUEST_BUFFERS)}",
        )
    return backend_name


def _select_buffer(requested: Optional[str]) -> RequestBuffer:
    return REQUEST_BUFFERS[_resolve_backend_name(requested)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ASR_BACKENDS, REQUEST_BUFFERS, DEFAULT_BACKEND

    args = _cli_args
    DEFAULT_BACKEND = args.default_backend
    ASR_BACKENDS = _load_backends(args)
    REQUEST_BUFFERS = {
        backend_name: RequestBuffer(
            backend,
            max_batch_size=args.buffer_batch_size,
            max_wait_ms=args.buffer_wait_ms,
        )
        for backend_name, backend in ASR_BACKENDS.items()
    }
    for buffer in REQUEST_BUFFERS.values():
        buffer.start()

    logger.info("ASR gateway ready with backends: %s", sorted(ASR_BACKENDS))
    yield

    for buffer in REQUEST_BUFFERS.values():
        await buffer.stop()
    logger.info("ASR gateway shut down")


app = FastAPI(
    title="ASR Gateway",
    description="Repo-native ASR gateway with per-backend request buffering.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    backend = ASR_BACKENDS.get(DEFAULT_BACKEND)
    return {
        "status": "ok" if backend is not None else "starting",
        "service": "asr",
        "backend": DEFAULT_BACKEND,
        "inference": backend.inference_name if backend else None,
        "model_loaded": bool(ASR_BACKENDS) and all(item.is_loaded for item in ASR_BACKENDS.values()),
        "default_backend": DEFAULT_BACKEND,
        "backends": {
            name: {
                "inference": item.inference_name,
                "model": item.model_path,
                "model_loaded": item.is_loaded,
            }
            for name, item in ASR_BACKENDS.items()
        },
    }


@app.get("/info")
async def info():
    return {
        "backend": DEFAULT_BACKEND,
        "default_backend": DEFAULT_BACKEND,
        "backends": {
            name: {
                "inference": backend.inference_name,
                "model": backend.model_path,
                "buffer_batch_size": REQUEST_BUFFERS[name].max_batch_size,
                "buffer_wait_ms": REQUEST_BUFFERS[name].max_wait_ms,
            }
            for name, backend in ASR_BACKENDS.items()
        },
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    if not REQUEST_BUFFERS:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        buffer = _select_buffer(req.backend)
        audio_tuple = decode_audio_input(_model_dump(req.audio))
        result = await buffer.submit(audio_tuple, req.language)
        return TranscribeResponse(text=result.text, language=result.language)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Transcribe failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/transcribe_batch", response_model=TranscribeBatchResponse)
async def transcribe_batch(req: TranscribeBatchRequest):
    if not REQUEST_BUFFERS:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        tasks = []
        for item in req.items:
            buffer = _select_buffer(item.backend or req.backend)
            audio_tuple = decode_audio_input(_model_dump(item.audio))
            tasks.append(buffer.submit(audio_tuple, item.language))
        results = await asyncio.gather(*tasks)
        return TranscribeBatchResponse(
            results=[
                TranscribeResponse(text=result.text, language=result.language)
                for result in results
            ]
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch transcribe failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    global _cli_args

    parser = argparse.ArgumentParser(description="Repo-native ASR gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers. Keep 1 so each backend buffer is single-process.",
    )
    parser.add_argument(
        "--backend-url",
        action="append",
        default=[],
        metavar="BACKEND=URL",
        help="Backend service URL. May be repeated, e.g. qwen3-asr=http://127.0.0.1:8033.",
    )
    parser.add_argument(
        "--default-backend",
        default="qwen3-asr",
        choices=sorted(SUPPORTED_BACKENDS),
        help="Backend used when a request does not specify one.",
    )
    parser.add_argument(
        "--qwen-transformers-model",
        default="",
        help="Load Qwen3-ASR directly with Transformers instead of a vLLM backend URL.",
    )
    parser.add_argument(
        "--qwen-transformers-device",
        default="cuda:0",
        help="Device map used by the direct Qwen3-ASR Transformers backend.",
    )
    parser.add_argument(
        "--qwen-transformers-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used by the direct Qwen3-ASR Transformers backend.",
    )
    parser.add_argument(
        "--qwen-transformers-max-new-tokens",
        type=int,
        default=512,
        help="Generation limit for the direct Qwen3-ASR Transformers backend.",
    )
    parser.add_argument(
        "--backend-timeout",
        type=float,
        default=300.0,
        help="Backend request timeout in seconds.",
    )
    parser.add_argument(
        "--backend-startup-timeout",
        type=float,
        default=1200.0,
        help="Maximum seconds to wait for configured backend services at startup.",
    )
    parser.add_argument(
        "--buffer-batch-size",
        type=int,
        default=16,
        help="Max requests to batch per backend before transcribing.",
    )
    parser.add_argument(
        "--buffer-wait-ms",
        type=float,
        default=500,
        help="Max milliseconds to wait for more requests per backend.",
    )
    args = parser.parse_args()
    _cli_args = args

    logger.info(
        "Starting ASR gateway on %s:%s "
        "(default_backend=%s, backend_urls=%s, qwen_transformers_model=%s)",
        args.host,
        args.port,
        args.default_backend,
        args.backend_url,
        args.qwen_transformers_model or None,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
