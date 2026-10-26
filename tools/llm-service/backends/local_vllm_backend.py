"""Local vLLM backend with genuine cross-request continuous batching.

The unified service remains the public API. A private vLLM OpenAI server owns
the model and its asynchronous scheduler, while this backend forwards each
incoming request independently. This preserves vLLM continuous batching for
both text-only models and multimodal Qwen3-Omni requests.
"""

import ctypes
import logging
import signal
import socket
import subprocess
import sys
import time
from typing import List, Optional

import requests

from . import LLMBackend, LLMResponse
from .openai_compatible_backend import OpenAICompatibleBackend

logger = logging.getLogger(__name__)


SUPPORTED_LOCAL_MODELS = {
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3-omni-30b-a3b-instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B-it",
    "gemma-4-31b": "google/gemma-4-31B",
}


def resolve_local_model(model: str) -> str:
    """Resolve a supported shortcut to its HuggingFace model ID."""
    return SUPPORTED_LOCAL_MODELS.get(model.strip().lower(), model)


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate_child_with_parent() -> None:
    """Ask Linux to SIGTERM the private server if the gateway disappears."""
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        raise OSError("prctl(PR_SET_PDEATHSIG) failed")


class LocalVLLMBackend(LLMBackend):
    """Manage a private asynchronous vLLM server and proxy requests to it."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-27B",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        max_num_seqs: Optional[int] = None,
        tensor_parallel_size: int = 1,
        device: str = "auto",
        enable_thinking: bool = False,
        enforce_eager: bool = False,
        max_cudagraph_capture_size: Optional[int] = None,
        startup_timeout: float = 1800.0,
        internal_port: Optional[int] = None,
    ):
        self.model_path = resolve_local_model(model)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.tensor_parallel_size = tensor_parallel_size
        self.device = device
        self.enable_thinking = enable_thinking
        self.enforce_eager = enforce_eager
        self.max_cudagraph_capture_size = max_cudagraph_capture_size
        self.startup_timeout = startup_timeout
        self.internal_port = internal_port
        self._model_name = self.model_path.rsplit("/", 1)[-1].lower()
        self._supports_audio = "qwen3-omni" in self._model_name
        self._process: Optional[subprocess.Popen] = None
        self._delegate: Optional[OpenAICompatibleBackend] = None

    def _build_server_command(self, port: int) -> List[str]:
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--served-model-name",
            self._model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--trust-remote-code",
            "--disable-log-stats",
        ]
        if self.max_num_seqs is not None:
            command.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.enforce_eager:
            command.append("--enforce-eager")
        if self.max_cudagraph_capture_size is not None:
            command.extend(
                [
                    "--max-cudagraph-capture-size",
                    str(self.max_cudagraph_capture_size),
                ]
            )
        return command

    def load(self) -> None:
        """Start the private vLLM server and wait for model readiness."""
        if self._process is not None:
            raise RuntimeError("Local vLLM backend is already loaded")
        port = self.internal_port or _reserve_loopback_port()
        command = self._build_server_command(port)
        logger.info(
            "Starting private vLLM server: model=%s tp=%s port=%s",
            self.model_path,
            self.tensor_parallel_size,
            port,
        )
        self._process = subprocess.Popen(
            command,
            preexec_fn=_terminate_child_with_parent,
        )
        models_url = f"http://127.0.0.1:{port}/v1/models"
        deadline = time.monotonic() + self.startup_timeout
        last_error = "not ready"
        while time.monotonic() < deadline:
            returncode = self._process.poll()
            if returncode is not None:
                self._process = None
                raise RuntimeError(
                    f"Private vLLM server exited during startup with code {returncode}"
                )
            try:
                response = requests.get(models_url, timeout=5.0)
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    if any(item.get("id") == self._model_name for item in models):
                        self._delegate = OpenAICompatibleBackend(
                            name="local",
                            base_url=f"http://127.0.0.1:{port}/v1",
                            model=self._model_name,
                            timeout=300.0,
                            # HTTP concurrency may exceed the number of GPU
                            # sequences; vLLM queues excess independent requests.
                            max_workers=max(32, self.max_num_seqs or 8),
                            enable_thinking=(
                                self.enable_thinking
                                if "qwen" in self._model_name
                                else None
                            ),
                        )
                        logger.info(
                            "Private vLLM server ready: model=%s port=%s",
                            self._model_name,
                            port,
                        )
                        return
                last_error = f"HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            time.sleep(1.0)
        self.close()
        raise RuntimeError(
            f"Private vLLM server was not ready after {self.startup_timeout}s: "
            f"{last_error}"
        )

    def close(self, timeout: float = 30.0) -> None:
        """Stop only the private vLLM process launched by this backend."""
        process = self._process
        self._delegate = None
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)

    @property
    def name(self) -> str:
        return "local"

    @property
    def available_models(self) -> List[str]:
        return [self._model_name]

    @property
    def is_ready(self) -> bool:
        return (
            self._delegate is not None
            and self._process is not None
            and self._process.poll() is None
        )

    def _ready_delegate(self) -> OpenAICompatibleBackend:
        if not self.is_ready or self._delegate is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._delegate

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop_sequences: Optional[List[str]] = None,
        audio_base64: Optional[str] = None,
        audio_mime_type: Optional[str] = None,
    ) -> LLMResponse:
        if audio_base64 is not None and not self._supports_audio:
            raise ValueError(
                f"Local model {self._model_name!r} does not support audio input"
            )
        return self._ready_delegate().generate(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stop_sequences=stop_sequences,
            audio_base64=audio_base64,
            audio_mime_type=audio_mime_type,
        )

    def generate_batch(
        self,
        prompts: List[str],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop_sequences: Optional[List[str]] = None,
        audio_base64_list: Optional[List[Optional[str]]] = None,
        audio_mime_type: Optional[str] = None,
    ) -> List[LLMResponse]:
        if audio_base64_list is not None and len(audio_base64_list) != len(prompts):
            raise ValueError(
                "audio_base64_list must have the same length as prompts "
                f"({len(audio_base64_list)} != {len(prompts)})"
            )
        if (
            audio_base64_list
            and any(audio is not None for audio in audio_base64_list)
            and not self._supports_audio
        ):
            raise ValueError(
                f"Local model {self._model_name!r} does not support audio input"
            )
        return self._ready_delegate().generate_batch(
            prompts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stop_sequences=stop_sequences,
            audio_base64_list=audio_base64_list,
            audio_mime_type=audio_mime_type,
        )
