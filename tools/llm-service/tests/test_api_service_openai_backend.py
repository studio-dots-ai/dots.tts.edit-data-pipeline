from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path
import threading
import time

import pytest
from fastapi.testclient import TestClient

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import api_service  # noqa: E402
from backends import LLMResponse  # noqa: E402


class _FakeOpenAIResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _FakeOpenAISession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        prompt = json["messages"][0]["content"]
        return _FakeOpenAIResponse(
            {
                "model": json["model"],
                "choices": [
                    {"message": {"role": "assistant", "content": f"ok:{prompt}"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        )


def _reset_options(monkeypatch):
    monkeypatch.setattr(api_service.app.state, "args", api_service.parse_args([]))


def test_health_includes_configured_openai_compatible_backend(monkeypatch):
    api_service.BACKENDS.clear()
    _reset_options(monkeypatch)
    monkeypatch.setattr(api_service.app.state.args, "gemini_model", "3flash")
    monkeypatch.setattr(api_service.app.state.args, "openai_base_url", "http://localhost:18112/v1")
    monkeypatch.setattr(api_service.app.state.args, "openai_backend_name", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "openai_model", "gemma-4-31b-it-mtp-tp2-32k")
    monkeypatch.setattr(api_service.app.state.args, "openai_timeout", 42)
    monkeypatch.setattr(api_service.app.state.args, "openai_batch_workers", 3)

    try:
        with TestClient(api_service.app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["backends"]["gemma4_mtp"] == {
            "ready": True,
            "models": ["gemma-4-31b-it-mtp-tp2-32k"],
        }
    finally:
        api_service.BACKENDS.clear()


def test_generate_endpoint_uses_configured_openai_backend(monkeypatch, caplog):
    api_service.BACKENDS.clear()
    _reset_options(monkeypatch)
    caplog.set_level("INFO", logger=api_service.__name__)
    monkeypatch.setattr(api_service.app.state.args, "gemini_model", "3flash")
    monkeypatch.setattr(api_service.app.state.args, "openai_base_url", "http://localhost:18112/v1")
    monkeypatch.setattr(api_service.app.state.args, "openai_backend_name", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "openai_model", "default-model")
    monkeypatch.setattr(api_service.app.state.args, "openai_timeout", 42)

    try:
        with TestClient(api_service.app) as client:
            backend = api_service.BACKENDS["gemma4_mtp"]
            fake_session = _FakeOpenAISession()
            backend.session = fake_session

            response = client.post(
                "/generate",
                json={
                    "backend": "gemma4_mtp",
                    "model": "override-model",
                    "prompt": "hello",
                    "temperature": 0.0,
                    "max_tokens": 16,
                    "json_mode": True,
                    "stop_sequences": ["END"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "ok:hello"
        assert data["model"] == "override-model"
        assert data["backend"] == "gemma4_mtp"
        assert fake_session.calls == [
            {
                "url": "http://localhost:18112/v1/chat/completions",
                "json": {
                    "model": "override-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.0,
                    "max_tokens": 16,
                    "stop": ["END"],
                    "response_format": {"type": "json_object"},
                },
                "timeout": 42.0,
            }
        ]
        assert any(
            "Generation completed: backend=gemma4_mtp" in record.message
            and "latency_ms=" in record.message
            and "prompt_tokens=1" in record.message
            and "completion_tokens=2" in record.message
            for record in caplog.records
        )
    finally:
        api_service.BACKENDS.clear()


def test_generate_endpoint_uses_service_default_backend_when_omitted(monkeypatch):
    api_service.BACKENDS.clear()
    _reset_options(monkeypatch)
    monkeypatch.setattr(api_service.app.state.args, "default_backend", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "gemini_model", "3flash")
    monkeypatch.setattr(api_service.app.state.args, "openai_base_url", "http://localhost:18112/v1")
    monkeypatch.setattr(api_service.app.state.args, "openai_backend_name", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "openai_model", "default-model")

    try:
        with TestClient(api_service.app) as client:
            backend = api_service.BACKENDS["gemma4_mtp"]
            fake_session = _FakeOpenAISession()
            backend.session = fake_session

            response = client.post(
                "/generate",
                json={
                    "prompt": "hello",
                    "temperature": 0.0,
                    "max_tokens": 16,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "ok:hello"
        assert data["model"] == "default-model"
        assert data["backend"] == "gemma4_mtp"
        assert fake_session.calls[0]["json"]["model"] == "default-model"
    finally:
        api_service.BACKENDS.clear()


def test_generate_endpoint_allows_four_real_overlapping_requests(monkeypatch):
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = 0
    peak = 0

    class ConcurrentBackend:
        def generate(self, prompt, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=2)
            time.sleep(0.02)
            with lock:
                active -= 1
            return LLMResponse(text=f"ok:{prompt}", model="gemini-2.5-pro")

    backend = ConcurrentBackend()
    monkeypatch.setattr(
        api_service,
        "_get_backend",
        lambda _name: ("gemini", backend),
    )
    with TestClient(api_service.app) as client:
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(
                executor.map(
                    lambda index: client.post(
                        "/generate",
                        json={
                            "backend": "gemini",
                            "prompt": f"request-{index}",
                            "max_tokens": 8,
                        },
                    ),
                    range(4),
                )
            )

    assert peak == 4
    assert [response.status_code for response in responses] == [200] * 4
    assert [response.json()["text"] for response in responses] == [
        f"ok:request-{index}" for index in range(4)
    ]


def test_generate_batch_endpoint_preserves_openai_backend_order(monkeypatch):
    api_service.BACKENDS.clear()
    _reset_options(monkeypatch)
    monkeypatch.setattr(api_service.app.state.args, "gemini_model", "3flash")
    monkeypatch.setattr(api_service.app.state.args, "openai_base_url", "http://localhost:18112/v1")
    monkeypatch.setattr(api_service.app.state.args, "openai_backend_name", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "openai_model", "default-model")
    monkeypatch.setattr(api_service.app.state.args, "openai_batch_workers", 2)

    try:
        with TestClient(api_service.app) as client:
            backend = api_service.BACKENDS["gemma4_mtp"]
            backend.session = _FakeOpenAISession()

            response = client.post(
                "/generate_batch",
                json={
                    "backend": "gemma4_mtp",
                    "prompts": ["first", "second", "third"],
                    "temperature": 0.1,
                    "max_tokens": 8,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert [item["text"] for item in data["results"]] == [
            "ok:first",
            "ok:second",
            "ok:third",
        ]
        assert [item["backend"] for item in data["results"]] == ["gemma4_mtp"] * 3
    finally:
        api_service.BACKENDS.clear()


def test_generate_batch_endpoint_forwards_per_prompt_audio(monkeypatch):
    calls = []

    class AudioBackend:
        is_ready = True

        def generate_batch(self, prompts, **kwargs):
            calls.append((prompts, kwargs))
            return [
                LLMResponse(text=f"ok:{prompt}", model="qwen3-omni")
                for prompt in prompts
            ]

    monkeypatch.setattr(
        api_service,
        "_get_backend",
        lambda _name: ("local", AudioBackend()),
    )

    with TestClient(api_service.app) as client:
        response = client.post(
            "/generate_batch",
            json={
                "backend": "local",
                "prompts": ["first", "second"],
                "audio_base64_list": ["Zmlyc3Q=", "c2Vjb25k"],
                "audio_mime_type": "audio/wav",
                "temperature": 0.0,
                "max_tokens": 32,
            },
        )

    assert response.status_code == 200
    assert [item["text"] for item in response.json()["results"]] == [
        "ok:first",
        "ok:second",
    ]
    assert calls[0][1]["audio_base64_list"] == ["Zmlyc3Q=", "c2Vjb25k"]
    assert calls[0][1]["audio_mime_type"] == "audio/wav"


def test_startup_launches_gemma4_mtp_and_registers_backend(monkeypatch, tmp_path):
    api_service.BACKENDS.clear()
    _reset_options(monkeypatch)
    launch_script = tmp_path / "launch_gemma4_mtp_vllm.sh"
    log_dir = tmp_path / "logs"
    captured = {}

    def _fake_run(cmd, *, capture_output, text, check, timeout):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "BASE_URL=http://127.0.0.1:18199/v1\n"
                "STATUS=reused\n"
                "PID=123\n"
                "LOG_FILE=/repo/log.log\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(api_service.subprocess, "run", _fake_run)
    monkeypatch.setattr(api_service.app.state.args, "gemini_model", "3flash")
    monkeypatch.setattr(api_service.app.state.args, "openai_backend_name", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "openai_model", "custom-mtp-model")
    monkeypatch.setattr(api_service.app.state.args, "openai_timeout", 42)
    monkeypatch.setattr(api_service.app.state.args, "openai_batch_workers", 3)
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_gpus", "0,1")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_launch_script", str(launch_script))
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_port", 18199)
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_launch_timeout", 12)
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_model_path", "/models/main with spaces")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_draft_model_path", "/models/draft")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_vllm_python", "/envs/vllm/bin/python")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_log_dir", str(log_dir))
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_max_model_len", 8192)
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_max_num_batched_tokens", 8192)
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_max_num_seqs", 4)
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_gpu_memory_utilization", 0.91)

    try:
        with TestClient(api_service.app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["backends"]["gemma4_mtp"] == {
            "ready": True,
            "models": ["custom-mtp-model"],
        }
        assert captured == {
            "cmd": [
                str(launch_script),
                "--gpus",
                "0,1",
                "--port",
                "18199",
                "--served-model-name",
                "custom-mtp-model",
                "--timeout",
                "12",
                "--model-path",
                "/models/main with spaces",
                "--draft-model-path",
                "/models/draft",
                "--vllm-python",
                "/envs/vllm/bin/python",
                "--log-dir",
                str(log_dir),
                "--max-model-len",
                "8192",
                "--max-num-batched-tokens",
                "8192",
                "--max-num-seqs",
                "4",
                "--gpu-memory-utilization",
                "0.91",
            ],
            "capture_output": True,
            "text": True,
            "check": False,
            "timeout": 72.0,
        }
    finally:
        api_service.BACKENDS.clear()


def test_main_accepts_gemma4_mtp_gpus_auto_without_value(monkeypatch):
    _reset_options(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "api_service.py",
            "--gemma4-mtp-gpus",
            "--gemma4-mtp-max-model-len",
            "8192",
            "--port",
            "18199",
        ],
    )
    monkeypatch.setattr(
        api_service.uvicorn,
        "run",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}),
    )

    api_service.main()

    assert captured["kwargs"]["port"] == 18199
    assert api_service.app.state.args.gemma4_mtp_gpus == "auto"
    assert api_service.app.state.args.gemma4_mtp_max_model_len == 8192


def test_launched_gemma4_mtp_pid_is_terminated_on_shutdown(monkeypatch, tmp_path):
    api_service.BACKENDS.clear()
    _reset_options(monkeypatch)
    launch_script = tmp_path / "launch_gemma4_mtp_vllm.sh"
    kill_calls = []

    def _fake_run(cmd, *, capture_output, text, check, timeout):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "BASE_URL=http://127.0.0.1:18199/v1\n"
                "STATUS=started\n"
                "PID=4321\n"
                "LOG_FILE=/repo/log.log\n"
            ),
            stderr="",
        )

    def _fake_kill(pid, sig):
        if sig == 0:
            raise ProcessLookupError
        kill_calls.append((pid, sig))

    monkeypatch.setattr(api_service.subprocess, "run", _fake_run)
    monkeypatch.setattr(api_service.os, "kill", _fake_kill)
    monkeypatch.setattr(api_service.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(api_service.os, "killpg", _fake_kill)
    monkeypatch.setattr(api_service.app.state.args, "gemini_model", "3flash")
    monkeypatch.setattr(api_service.app.state.args, "openai_backend_name", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "openai_model", "custom-mtp-model")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_gpus", "0")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_launch_script", str(launch_script))

    try:
        with TestClient(api_service.app) as client:
            assert client.get("/health").status_code == 200

        assert kill_calls == [(4321, api_service.signal.SIGTERM)]
        assert api_service.LAUNCHED_GEMMA4_MTP_PID is None
    finally:
        api_service.BACKENDS.clear()
        api_service.LAUNCHED_GEMMA4_MTP_PID = None
        api_service.LAUNCHED_GEMMA4_MTP_LOG_FILE = None


def test_shutdown_kills_workers_after_gemma4_group_leader_exits(monkeypatch):
    kill_calls = []

    def missing_leader(pid):
        raise ProcessLookupError

    monkeypatch.setattr(api_service, "LAUNCHED_GEMMA4_MTP_PID", 4321)
    monkeypatch.setattr(api_service.os, "getpgid", missing_leader)
    monkeypatch.setattr(api_service.os, "killpg", lambda pid, sig: kill_calls.append((pid, sig)))
    api_service._terminate_launched_gemma4_mtp_vllm(timeout=0)
    assert kill_calls == [(4321, api_service.signal.SIGTERM), (4321, api_service.signal.SIGKILL)]
    assert api_service.LAUNCHED_GEMMA4_MTP_PID is None


def test_startup_fails_when_gemma4_mtp_launcher_fails(monkeypatch, tmp_path):
    api_service.BACKENDS.clear()
    _reset_options(monkeypatch)
    launch_script = tmp_path / "launch_gemma4_mtp_vllm.sh"

    def _fake_run(cmd, *, capture_output, text, check, timeout):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="partial launcher output",
            stderr="server did not become ready",
        )

    monkeypatch.setattr(api_service.subprocess, "run", _fake_run)
    monkeypatch.setattr(api_service.app.state.args, "gemini_model", "3flash")
    monkeypatch.setattr(api_service.app.state.args, "openai_backend_name", "gemma4_mtp")
    monkeypatch.setattr(api_service.app.state.args, "openai_model", "custom-mtp-model")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_gpus", "0,1")
    monkeypatch.setattr(api_service.app.state.args, "gemma4_mtp_launch_script", str(launch_script))

    try:
        with pytest.raises(RuntimeError, match="Gemma 4 MTP launcher failed"):
            with TestClient(api_service.app):
                pass
    finally:
        api_service.BACKENDS.clear()
