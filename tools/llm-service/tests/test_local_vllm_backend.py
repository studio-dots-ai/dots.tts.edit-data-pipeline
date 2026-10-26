import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from backends.local_vllm_backend import LocalVLLMBackend, SUPPORTED_LOCAL_MODELS
from backends import LLMResponse


class _FakeProcess:
    pid = 12345

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        self.returncode = 0
        return 0


class _FakeDelegate:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return LLMResponse(text=f"response:{prompt}", model="served-model")

    def generate_batch(self, prompts, **kwargs):
        self.calls.append((list(prompts), kwargs))
        return [
            LLMResponse(text=f"response:{prompt}", model="served-model")
            for prompt in prompts
        ]


def _mark_ready(backend, delegate=None):
    backend._process = _FakeProcess()
    backend._delegate = delegate or _FakeDelegate()
    return backend._delegate


def test_supported_local_models_include_gemma_aliases():
    assert SUPPORTED_LOCAL_MODELS["gemma-4-26b-a4b-it"] == "google/gemma-4-26B-A4B-it"
    assert SUPPORTED_LOCAL_MODELS["gemma-4-31b"] == "google/gemma-4-31B"
    assert SUPPORTED_LOCAL_MODELS["qwen3-omni-30b-a3b-instruct"] == (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )


@pytest.mark.parametrize(
    ("requested_model", "expected_path", "expected_name"),
    [
        (
            "gemma-4-26b-a4b-it",
            "google/gemma-4-26B-A4B-it",
            "gemma-4-26b-a4b-it",
        ),
        ("gemma-4-31b", "google/gemma-4-31B", "gemma-4-31b"),
        ("google/gemma-4-31B", "google/gemma-4-31B", "gemma-4-31b"),
    ],
)
def test_local_backend_resolves_gemma_model_names(
    requested_model,
    expected_path,
    expected_name,
):
    backend = LocalVLLMBackend(model=requested_model)
    assert backend.model_path == expected_path
    assert backend.available_models == [expected_name]


def test_gemma_private_server_command_uses_served_model_alias():
    backend = LocalVLLMBackend(model="gemma-4-31b")
    command = backend._build_server_command(19001)

    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--served-model-name") + 1] == "gemma-4-31b"
    assert command[command.index("--port") + 1] == "19001"


def test_qwen_requests_are_forwarded_to_private_server():
    backend = LocalVLLMBackend(model="qwen3.5-27b", enable_thinking=True)
    delegate = _mark_ready(backend)

    backend.generate("hello qwen")

    assert delegate.calls[0][1]["audio_base64"] is None
    assert backend.enable_thinking is True


def test_gemma_raw_prompt_is_forwarded_to_private_server():
    backend = LocalVLLMBackend(model="gemma-4-31b")
    delegate = _mark_ready(backend)

    response = backend.generate("raw prompt")

    assert response.text == "response:raw prompt"
    assert delegate.calls[0][0] == "raw prompt"


def test_private_server_command_includes_scheduler_options():
    backend = LocalVLLMBackend(
        model="gemma-4-31b",
        tensor_parallel_size=2,
        max_num_seqs=16,
        enforce_eager=True,
    )
    command = backend._build_server_command(19002)

    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--max-num-seqs") + 1] == "16"
    assert "--enforce-eager" in command


def test_load_passes_max_num_seqs_to_vllm(monkeypatch):
    import backends.local_vllm_backend as local_backend_module

    captured = {}
    process = _FakeProcess()
    monkeypatch.setattr(
        local_backend_module.subprocess,
        "Popen",
        lambda command, **kwargs: (
            captured.update(command=command, kwargs=kwargs) or process
        ),
    )
    monkeypatch.setattr(
        local_backend_module.requests,
        "get",
        lambda url, timeout: SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [{"id": "qwen3-omni-30b-a3b-instruct"}]
            },
        ),
    )

    backend = LocalVLLMBackend(
        model="qwen3-omni-30b-a3b-instruct",
        max_num_seqs=32,
        internal_port=19003,
    )
    backend.load()

    assert captured["command"][captured["command"].index("--max-num-seqs") + 1] == "32"
    assert callable(captured["kwargs"]["preexec_fn"])
    assert backend.is_ready is True
    assert backend._delegate.base_url == "http://127.0.0.1:19003/v1"
    assert backend._delegate.enable_thinking is False


def test_qwen3_omni_audio_batch_is_forwarded_and_preserves_order():
    backend = LocalVLLMBackend(model="qwen3-omni-30b-a3b-instruct")
    delegate = _mark_ready(backend)

    responses = backend.generate_batch(
        ["first prompt", "second prompt"],
        temperature=0.0,
        max_tokens=32,
        audio_base64_list=["Zmlyc3Q=", "c2Vjb25k"],
        audio_mime_type="audio/wav",
    )

    assert [response.text for response in responses] == [
        "response:first prompt",
        "response:second prompt",
    ]
    assert delegate.calls == [
        (
            ["first prompt", "second prompt"],
            {
                "model": None,
                "temperature": 0.0,
                "max_tokens": 32,
                "json_mode": False,
                "stop_sequences": None,
                "audio_base64_list": ["Zmlyc3Q=", "c2Vjb25k"],
                "audio_mime_type": "audio/wav",
            },
        )
    ]


def test_local_requests_remain_concurrent_for_vllm_scheduler(monkeypatch):
    class _ConcurrencyDetectingDelegate(_FakeDelegate):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.state_lock = threading.Lock()

        def generate(self, prompt, **kwargs):
            with self.state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with self.state_lock:
                self.active -= 1
            return LLMResponse(text="calm", model="served-model")

    backend = LocalVLLMBackend(model="qwen3-omni-30b-a3b-instruct")
    delegate = _ConcurrencyDetectingDelegate()
    _mark_ready(backend, delegate)

    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(
            executor.map(
                lambda index: backend.generate(
                    f"prompt {index}", audio_base64="YXVkaW8="
                ),
                range(16),
            )
        )

    assert [response.text for response in responses] == ["calm"] * 16
    assert delegate.max_active > 1


def test_text_only_model_rejects_audio_instead_of_ignoring_it():
    backend = LocalVLLMBackend(model="qwen3.5-27b")
    _mark_ready(backend)

    with pytest.raises(ValueError, match="does not support audio input"):
        backend.generate("classify", audio_base64="YXVkaW8=")


def test_audio_batch_length_must_match_prompts():
    backend = LocalVLLMBackend(model="qwen3-omni-30b-a3b-instruct")
    _mark_ready(backend)

    with pytest.raises(ValueError, match="same length"):
        backend.generate_batch(["one", "two"], audio_base64_list=["b25l"])


def test_close_terminates_private_vllm_process():
    backend = LocalVLLMBackend(model="qwen3-omni-30b-a3b-instruct")
    process = _FakeProcess()
    backend._process = process
    backend._delegate = _FakeDelegate()

    backend.close()

    assert process.terminated is True
    assert process.killed is False
    assert backend.is_ready is False
