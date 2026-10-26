"""Unit tests for Gemini backend retry cycling."""

import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from backends.gemini_backend import (  # noqa: E402
    GeminiAPIError,
    GeminiBackend,
    GeminiTokenBudgetExhausted,
)


@pytest.fixture(autouse=True)
def configured_backend(monkeypatch):
    monkeypatch.setenv("GEMINI_API_URL", "http://gemini.test/generate")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-placeholder")


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse(status_code=500, text="still unavailable")


def _ok_response(text="ok"):
    return _FakeResponse(
        payload={"candidates": [{"content": {"parts": [{"text": text}]}}]}
    )


def test_gemini_retry_starts_new_cycle_after_attempt_budget(monkeypatch):
    sleeps = []
    monkeypatch.setattr("backends.gemini_backend.time.sleep", sleeps.append)

    backend = GeminiBackend(
        max_retries=2,
        max_retry_cycles=2,
        retry_backoff=(0.0, 0.0),
        retry_cycle_sleep_sec=180.0,
    )
    backend.session = _FakeSession(
        [
            _FakeResponse(status_code=500, text="key unavailable"),
            _FakeResponse(status_code=500, text="key unavailable"),
            _ok_response("done"),
        ]
    )

    text, attempts, _raw = backend._request({"contents": []})

    assert text == "done"
    assert attempts == 3
    assert backend.session.calls == 3
    assert 180.0 in sleeps


def test_gemini_generate_does_not_expose_retry_count(monkeypatch):
    sleeps = []
    monkeypatch.setattr("backends.gemini_backend.time.sleep", sleeps.append)

    backend = GeminiBackend(
        max_retries=2,
        max_retry_cycles=2,
        retry_backoff=(0.0, 0.0),
        retry_cycle_sleep_sec=180.0,
    )
    backend.session = _FakeSession(
        [
            _FakeResponse(status_code=500, text="key unavailable"),
            _ok_response("done"),
        ]
    )

    response = backend.generate("hello")

    assert response.text == "done"
    assert response.usage == {}
    assert backend.session.calls == 2


def test_gemini_retry_reports_total_attempts_after_all_cycles(monkeypatch):
    sleeps = []
    monkeypatch.setattr("backends.gemini_backend.time.sleep", sleeps.append)

    backend = GeminiBackend(
        max_retries=2,
        max_retry_cycles=3,
        retry_backoff=(0.0, 0.0),
        retry_cycle_sleep_sec=180.0,
    )
    backend.session = _FakeSession([])

    with pytest.raises(GeminiAPIError, match="retry budget was exhausted"):
        backend._request({"contents": []})

    assert backend.session.calls == 6
    assert sleeps.count(180.0) == 2


def test_gemini_token_budget_exhausted_does_not_retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr("backends.gemini_backend.time.sleep", sleeps.append)

    backend = GeminiBackend(
        max_retries=2,
        max_retry_cycles=3,
        retry_backoff=(0.0, 0.0),
        retry_cycle_sleep_sec=180.0,
    )
    backend.session = _FakeSession(
        [_FakeResponse(payload={"candidates": [{"finishReason": "MAX_TOKENS"}]})]
    )

    with pytest.raises(GeminiTokenBudgetExhausted):
        backend._request({"contents": []})

    assert backend.session.calls == 1
    assert sleeps == []


def test_gemini_requires_user_configuration(monkeypatch):
    for name in ("GEMINI_API_URL", "GEMINI_API_KEY", "GEMINI_3FLASH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    backend = GeminiBackend()
    assert not backend.is_ready
    with pytest.raises(GeminiAPIError, match="GEMINI_API_URL"):
        backend.generate("hello")


def test_gemini_model_specific_key_precedence(monkeypatch):
    monkeypatch.setenv("GEMINI_3FLASH_API_KEY", "model-test-placeholder")
    backend = GeminiBackend()
    assert backend.headers["api-key"] == "model-test-placeholder"
    assert backend.url == "http://gemini.test/generate"
    assert backend.is_ready
