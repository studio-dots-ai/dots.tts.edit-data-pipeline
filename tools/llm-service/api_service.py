"""
Unified LLM Generation Service.

FastAPI gateway that routes text generation requests to either:
  - Gemini API backend (remote)
  - Local vLLM backend (Qwen3.5, Qwen3-Omni, Gemma 4, or compatible instruct models)
  - OpenAI-compatible backend (external vLLM server, e.g. Gemma 4 MTP)

Prompt rendering can happen client-side (send pre-built ``prompt``) or
server-side (send ``template`` + ``template_vars``).

Usage:
    source envs/pipe-llm/.venv/bin/activate
    python api_service.py --local-model Qwen/Qwen3.5-27B --port 8012
    python api_service.py --local-model google/gemma-4-31B --port 8012

    # API docs: http://localhost:8012/docs
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backends import LLMBackend, LLMResponse
from backends.gemini_backend import GeminiBackend, VALID_MODELS
from backends.local_vllm_backend import SUPPORTED_LOCAL_MODELS
from prompt_template import PromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Globals
# --------------------------------------------------------------------------- #

BACKENDS: Dict[str, LLMBackend] = {}
DEFAULT_BACKEND: Optional[str] = None
DEFAULT_GEMMA4_MTP_MODEL = "gemma-4-31b-it-mtp-tp2-32k"
DEFAULT_GEMMA4_MTP_LAUNCH_SCRIPT = (
    Path(__file__).resolve().parent / "scripts" / "launch_gemma4_mtp_vllm.sh"
)
LAUNCHED_GEMMA4_MTP_PID: Optional[int] = None
LAUNCHED_GEMMA4_MTP_LOG_FILE: Optional[str] = None


# --------------------------------------------------------------------------- #
# External backend launcher helpers
# --------------------------------------------------------------------------- #


def _parse_key_value_output(output: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines emitted by service launcher scripts."""
    values = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip()
    return values


def _maybe_launch_gemma4_mtp_vllm() -> Optional[str]:
    """Launch or reuse the external Gemma 4 MTP vLLM server when requested."""
    global LAUNCHED_GEMMA4_MTP_LOG_FILE
    global LAUNCHED_GEMMA4_MTP_PID

    args = app.state.args
    if not args.gemma4_mtp_gpus:
        return None

    launch_script = args.gemma4_mtp_launch_script
    gpus = args.gemma4_mtp_gpus
    cmd = [
        launch_script,
        "--gpus", gpus,
        "--port", str(args.gemma4_mtp_port),
        "--served-model-name", args.openai_model,
        "--timeout", str(args.gemma4_mtp_launch_timeout),
    ]
    for option in (
        "model_path", "draft_model_path", "vllm_python",
        "log_dir", "max_model_len", "max_num_batched_tokens", "max_num_seqs",
        "gpu_memory_utilization", "tensor_parallel_size", "data_parallel_size",
        "data_parallel_size_local",
    ):
        value = getattr(args, f"gemma4_mtp_{option}")
        if value is not None:
            cmd.extend(["--" + option.replace("_", "-"), str(value)])
    subprocess_timeout = args.gemma4_mtp_launch_timeout + 60.0

    logger.info(
        "Launching/reusing Gemma 4 MTP vLLM via %s on GPUs %s",
        launch_script,
        gpus,
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=subprocess_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Gemma 4 MTP launcher timed out before the vLLM server became ready"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to run Gemma 4 MTP launcher: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            "Gemma 4 MTP launcher failed "
            f"(exit={result.returncode}). stdout={result.stdout[-2000:]!r} "
            f"stderr={result.stderr[-2000:]!r}"
        )

    values = _parse_key_value_output(result.stdout)
    base_url = values.get("BASE_URL")
    if not base_url:
        raise RuntimeError(
            "Gemma 4 MTP launcher did not emit BASE_URL. "
            f"stdout={result.stdout[-2000:]!r}"
        )
    if values.get("STATUS") == "started" and values.get("PID", "").isdigit():
        LAUNCHED_GEMMA4_MTP_PID = int(values["PID"])
        LAUNCHED_GEMMA4_MTP_LOG_FILE = values.get("LOG_FILE")
    else:
        LAUNCHED_GEMMA4_MTP_PID = None
        LAUNCHED_GEMMA4_MTP_LOG_FILE = None

    logger.info(
        "Gemma 4 MTP vLLM %s at %s (pid=%s, log=%s)",
        values.get("STATUS", "ready"),
        base_url,
        values.get("PID", "existing"),
        values.get("LOG_FILE", "existing"),
    )
    return base_url


def _terminate_launched_gemma4_mtp_vllm(timeout: float = 10.0) -> None:
    """Stop the vLLM process this service launched, but never reused servers."""
    global LAUNCHED_GEMMA4_MTP_LOG_FILE
    global LAUNCHED_GEMMA4_MTP_PID

    pid = LAUNCHED_GEMMA4_MTP_PID
    if pid is None:
        return

    logger.info(
        "Stopping Gemma 4 MTP vLLM launched by this service: pid=%s log=%s",
        pid,
        LAUNCHED_GEMMA4_MTP_LOG_FILE,
    )
    try:
        owns_group = os.getpgid(pid) == pid
    except ProcessLookupError:
        # The launcher uses setsid. Its workers can outlive the group leader.
        owns_group = True
    send_signal = os.killpg if owns_group else os.kill
    try:
        send_signal(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            send_signal(pid, 0)
        except ProcessLookupError:
            LAUNCHED_GEMMA4_MTP_PID = None
            LAUNCHED_GEMMA4_MTP_LOG_FILE = None
            return
        time.sleep(1)

    try:
        send_signal(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    LAUNCHED_GEMMA4_MTP_PID = None
    LAUNCHED_GEMMA4_MTP_LOG_FILE = None


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    """Single-prompt generation request."""

    backend: Optional[str] = Field(
        None,
        description=(
            "Backend name, e.g. 'gemini', 'local', or 'gemma4_mtp'. "
            "Uses the service default backend when omitted."
        ),
    )
    model: Optional[str] = Field(None, description="Model name override")
    prompt: Optional[str] = Field(None, description="Pre-rendered prompt text")
    template: Optional[str] = Field(None, description="Template with {placeholders}")
    template_vars: Optional[Dict[str, Any]] = Field(
        None, description="Variables for template"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=65536)
    json_mode: bool = Field(False, description="Constrain output to valid JSON")
    stop_sequences: Optional[List[str]] = None
    audio_base64: Optional[str] = Field(
        None,
        description="Base64-encoded audio data for a multimodal backend.",
    )
    audio_mime_type: Optional[str] = Field(
        None,
        description="MIME type of the audio (e.g. 'audio/wav', 'audio/mpeg'). "
        "Defaults to 'audio/wav' when audio_base64 is provided.",
    )


class GenerateResponse(BaseModel):
    """Single generation response."""

    text: str
    model: str
    backend: str
    usage: Dict[str, Any] = {}


class BatchGenerateRequest(BaseModel):
    """Batch generation request — same params applied to all prompts."""

    backend: Optional[str] = Field(
        None,
        description=(
            "Backend name, e.g. 'gemini', 'local', or 'gemma4_mtp'. "
            "Uses the service default backend when omitted."
        ),
    )
    model: Optional[str] = None
    prompts: Optional[List[str]] = Field(None, description="Pre-rendered prompts")
    template: Optional[str] = Field(None, description="Shared template")
    template_vars_list: Optional[List[Dict[str, Any]]] = Field(
        None, description="Per-prompt template variables"
    )
    temperature: float = 0.7
    max_tokens: int = 4096
    json_mode: bool = False
    stop_sequences: Optional[List[str]] = None
    audio_base64_list: Optional[List[Optional[str]]] = Field(
        None,
        description="Per-prompt base64 audio data; entries may be null.",
    )
    audio_mime_type: Optional[str] = Field(
        None,
        description="Shared MIME type for audio inputs; defaults to audio/wav.",
    )


class BatchGenerateResponse(BaseModel):
    """Batch generation response."""

    results: List[GenerateResponse]
    count: int


class BackendInfo(BaseModel):
    name: str
    ready: bool
    models: List[str]


class ServiceInfo(BaseModel):
    service: str
    backends: List[BackendInfo]


# --------------------------------------------------------------------------- #
# Prompt resolution helpers
# --------------------------------------------------------------------------- #


def _resolve_prompt(
    prompt: Optional[str],
    template: Optional[str],
    template_vars: Optional[Dict[str, Any]],
) -> str:
    """Resolve a prompt from either raw text or template + vars."""
    if prompt is not None:
        return prompt
    if template is not None:
        tpl = PromptTemplate(template)
        return tpl.render(**(template_vars or {}))
    raise ValueError("Either 'prompt' or 'template' must be provided")


def _resolve_prompts(
    prompts: Optional[List[str]],
    template: Optional[str],
    template_vars_list: Optional[List[Dict[str, Any]]],
) -> List[str]:
    """Resolve a list of prompts."""
    if prompts is not None:
        return prompts
    if template is not None and template_vars_list is not None:
        tpl = PromptTemplate(template)
        return [tpl.render(**v) for v in template_vars_list]
    raise ValueError(
        "Either 'prompts' or ('template' + 'template_vars_list') must be provided"
    )


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize backends on startup."""
    global DEFAULT_BACKEND
    args = app.state.args
    # Credentials and the API endpoint remain external to service parameters.
    gemini_model = args.gemini_model
    BACKENDS["gemini"] = GeminiBackend(model=gemini_model)
    logger.info("Gemini backend configured: %s (model: %s)", BACKENDS["gemini"].is_ready, gemini_model)

    # OpenAI-compatible external backend, e.g. custom Gemma 4 MTP vLLM server.
    openai_base_url = args.openai_base_url
    launched_openai_base_url = _maybe_launch_gemma4_mtp_vllm()
    if launched_openai_base_url:
        openai_base_url = launched_openai_base_url
    if openai_base_url:
        from backends.openai_compatible_backend import OpenAICompatibleBackend

        backend_name = args.openai_backend_name
        backend_model = args.openai_model
        timeout = args.openai_timeout
        batch_workers = args.openai_batch_workers

        if backend_name in BACKENDS:
            raise RuntimeError(
                f"Backend name '{backend_name}' conflicts with an existing backend"
            )

        backend = OpenAICompatibleBackend(
            name=backend_name,
            base_url=openai_base_url,
            model=backend_model,
            timeout=timeout,
            max_workers=batch_workers,
        )
        BACKENDS[backend.name] = backend
        logger.info(
            "OpenAI-compatible backend ready: %s -> %s (model=%s)",
            backend.name,
            backend.chat_completions_url,
            backend.available_models[0],
        )

    # Local vLLM backend (loaded if --local-model provided)
    local_model = args.local_model
    if local_model:
        from backends.local_vllm_backend import LocalVLLMBackend

        backend = LocalVLLMBackend(
            model=local_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_thinking=args.enable_thinking,
            enforce_eager=args.enforce_eager,
            max_cudagraph_capture_size=args.max_cudagraph_capture_size,
        )
        backend.load()
        BACKENDS["local"] = backend
        logger.info(
            "Local vLLM backend ready: %s (served as %s)",
            backend.model_path,
            backend.available_models[0],
        )
    else:
        logger.info("Local backend: no model specified, skipping")

    DEFAULT_BACKEND = args.default_backend
    logger.info(
        "Active backends: %s (default=%s)",
        list(BACKENDS.keys()),
        DEFAULT_BACKEND,
    )
    try:
        yield
    finally:
        local_backend = BACKENDS.get("local")
        close_local = getattr(local_backend, "close", None)
        if callable(close_local):
            close_local()
        _terminate_launched_gemma4_mtp_vllm()

    logger.info("LLM service shutting down")


app = FastAPI(
    title="Unified LLM Generation Service",
    description=(
        "Routes text generation to Gemini, local vLLM, or external "
        "OpenAI-compatible backends"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _resolve_backend_name(name: Optional[str]) -> str:
    resolved = (name or DEFAULT_BACKEND or "").strip()
    if not resolved:
        available = list(BACKENDS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"No backend requested and no default backend configured. Available: {available}",
        )
    return resolved


def _get_backend(name: Optional[str]) -> tuple[str, LLMBackend]:
    resolved = _resolve_backend_name(name)
    backend = BACKENDS.get(resolved)
    if backend is None:
        available = list(BACKENDS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{resolved}' not available. Available: {available}",
        )
    if not backend.is_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Backend '{resolved}' is not ready",
        )
    return resolved, backend


def _usage_count(usage: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "default_backend": DEFAULT_BACKEND,
        "backends": {
            name: {"ready": b.is_ready, "models": b.available_models}
            for name, b in BACKENDS.items()
        },
    }


@app.get("/info", response_model=ServiceInfo)
async def info():
    return ServiceInfo(
        service="Unified LLM Generation Service",
        backends=[
            BackendInfo(name=name, ready=b.is_ready, models=b.available_models)
            for name, b in BACKENDS.items()
        ],
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate text from a single prompt."""
    backend_name, backend = _get_backend(req.backend)

    try:
        prompt = _resolve_prompt(req.prompt, req.template, req.template_vars)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    started = time.monotonic()
    try:
        resp: LLMResponse = backend.generate(
            prompt,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            json_mode=req.json_mode,
            stop_sequences=req.stop_sequences,
            audio_base64=req.audio_base64,
            audio_mime_type=req.audio_mime_type,
        )
    except Exception as e:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        logger.error(
            "Generation failed: backend=%s model=%s latency_ms=%.1f prompt_chars=%s max_tokens=%s error=%s",
            backend_name,
            req.model,
            elapsed_ms,
            len(prompt),
            req.max_tokens,
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))

    usage = resp.usage or {}
    logger.info(
        "Generation completed: backend=%s model=%s latency_ms=%.1f prompt_chars=%s max_tokens=%s "
        "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        backend_name,
        resp.model,
        (time.monotonic() - started) * 1000.0,
        len(prompt),
        req.max_tokens,
        _usage_count(usage, "prompt_tokens", "input_tokens"),
        _usage_count(usage, "completion_tokens", "output_tokens"),
        _usage_count(usage, "total_tokens"),
    )
    return GenerateResponse(
        text=resp.text,
        model=resp.model,
        backend=backend_name,
        usage=usage,
    )


@app.post("/generate_batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest):
    """Generate text for multiple prompts in a batch."""
    backend_name, backend = _get_backend(req.backend)

    try:
        prompts = _resolve_prompts(req.prompts, req.template, req.template_vars_list)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        responses: List[LLMResponse] = backend.generate_batch(
            prompts,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            json_mode=req.json_mode,
            stop_sequences=req.stop_sequences,
            audio_base64_list=req.audio_base64_list,
            audio_mime_type=req.audio_mime_type,
        )
    except Exception as e:
        logger.error(f"Batch generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    results = [
        GenerateResponse(
            text=r.text,
            model=r.model,
            backend=backend_name,
            usage=r.usage,
        )
        for r in responses
    ]
    return BatchGenerateResponse(results=results, count=len(results))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified LLM Generation Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--default-backend",
        type=str,
        default="local",
        help="Backend used when /generate requests omit the backend field.",
    )

    # Local vLLM backend
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID, local path, or supported alias for the "
            "vLLM backend (aliases: "
            f"{', '.join(sorted(SUPPORTED_LOCAL_MODELS))})"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Maximum concurrent sequences for vLLM scheduler/warmup. "
            "Lower this for large models that OOM during startup."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable tokenizer thinking mode when supported (Qwen only today)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graph capture entirely",
    )
    parser.add_argument(
        "--max-cudagraph-capture-size",
        type=int,
        default=None,
        help="Max batch size for CUDA graph capture (default 512; use 256 for Qwen3.5 Mamba arch)",
    )

    # OpenAI-compatible external backend
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help=(
            "Base URL for an external OpenAI-compatible server. "
            "Use this for custom vLLM servers such as Gemma 4 MTP "
            "(e.g. http://localhost:18112/v1)."
        ),
    )
    parser.add_argument(
        "--openai-backend-name",
        type=str,
        default="gemma4_mtp",
        help="Backend name exposed by /generate for the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=DEFAULT_GEMMA4_MTP_MODEL,
        help="Default served model name for the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--openai-timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds for OpenAI-compatible requests.",
    )
    parser.add_argument(
        "--openai-batch-workers",
        type=int,
        default=8,
        help="Maximum concurrent requests used by /generate_batch for this backend.",
    )
    parser.add_argument("--gemma4-mtp-model-path", help="Local Gemma checkpoint directory.")
    parser.add_argument("--gemma4-mtp-draft-model-path", help="Local MTP checkpoint directory.")
    parser.add_argument("--gemma4-mtp-vllm-python", help="Existing Python interpreter for vLLM.")
    parser.add_argument(
        "--gemma4-mtp-gpus",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Optional comma-separated GPU ids for the bundled Gemma 4 MTP vLLM "
            "launcher. Passing the flag without a value, or using 'auto', uses "
            "all currently visible GPUs."
        ),
    )
    parser.add_argument(
        "--gemma4-mtp-launch-script",
        type=str,
        default=str(DEFAULT_GEMMA4_MTP_LAUNCH_SCRIPT),
        help="Path to the Gemma 4 MTP vLLM launcher script.",
    )
    parser.add_argument(
        "--gemma4-mtp-port",
        type=int,
        default=18112,
        help="Port for the external Gemma 4 MTP vLLM server.",
    )
    parser.add_argument(
        "--gemma4-mtp-launch-timeout",
        type=int,
        default=900,
        help="Seconds to wait for the external Gemma 4 MTP vLLM server to become ready.",
    )
    parser.add_argument(
        "--gemma4-mtp-log-dir",
        type=str,
        default=None,
        help="Optional directory for Gemma 4 MTP vLLM launcher logs and pid files.",
    )
    parser.add_argument(
        "--gemma4-mtp-max-model-len",
        type=int,
        default=None,
        help="Override Gemma 4 MTP vLLM max model length.",
    )
    parser.add_argument(
        "--gemma4-mtp-max-num-batched-tokens",
        type=int,
        default=None,
        help="Override Gemma 4 MTP vLLM max batched tokens.",
    )
    parser.add_argument(
        "--gemma4-mtp-max-num-seqs",
        type=int,
        default=None,
        help="Override Gemma 4 MTP vLLM max concurrent sequences.",
    )
    parser.add_argument(
        "--gemma4-mtp-gpu-memory-utilization",
        type=float,
        default=None,
        help="Override Gemma 4 MTP vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--gemma4-mtp-tensor-parallel-size",
        type=int,
        default=None,
        help="Override Gemma 4 MTP vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--gemma4-mtp-data-parallel-size",
        type=int,
        default=None,
        help="Override Gemma 4 MTP vLLM data parallel size.",
    )
    parser.add_argument(
        "--gemma4-mtp-data-parallel-size-local",
        type=int,
        default=None,
        help="Override Gemma 4 MTP vLLM local data parallel size.",
    )

    # Gemini backend
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="3flash",
        choices=VALID_MODELS,
        help="Gemini model to use: '3flash' (default) or '2_5pro'",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()
    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


app.state.args = parse_args([])


if __name__ == "__main__":
    main()
