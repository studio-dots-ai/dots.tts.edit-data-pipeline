import importlib.util
import sys

from pathlib import Path

from fastapi.testclient import TestClient


def _load_gateway_module():
    module_path = Path(__file__).resolve().parents[1] / "api_service.py"
    spec = importlib.util.spec_from_file_location(
        "aligner_gateway_service", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


api_service = _load_gateway_module()


def test_gateway_health_requires_all_configured_backends_in_deep_mode(monkeypatch):
    client = TestClient(api_service.app)

    async def fake_backend_health(url):
        return False

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_backend_health_ok", fake_backend_health)

    response = client.get("/health?deep=true")

    assert response.status_code == 503
    assert "qwen3" in response.json()["detail"]


def test_gateway_health_shallow_skips_backend_probe(monkeypatch):
    client = TestClient(api_service.app)
    calls = []

    async def fake_backend_health(url):
        calls.append(url)
        return False

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_backend_health_ok", fake_backend_health)

    response = client.get("/health")

    assert response.status_code == 200
    assert calls == []
    assert response.json() == {
        "status": "ok",
        "backends": {
            "qwen3": "http://qwen3-backend",
        },
    }


def test_gateway_health_requires_at_least_one_configured_backend(monkeypatch):
    client = TestClient(api_service.app)

    monkeypatch.setattr(api_service, "QWEN3_URL", None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "No aligner backends configured"


def test_gateway_auto_routes_simple_text_to_qwen3(monkeypatch):
    client = TestClient(api_service.app)
    captured = {}

    async def fake_proxy(method, url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "status": "success",
            "language": "eng",
            "level": "char",
            "num_chars": 2,
            "alignments": [
                {"char": "hi", "start": 0.0, "end": 0.4},
            ],
        }

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_proxy_json", fake_proxy)

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "Hello world",
            "language": "eng",
        },
    )

    assert response.status_code == 200
    assert captured["url"] == "http://qwen3-backend/align"
    assert response.json()["backend"] == "qwen3"


def test_gateway_auto_routes_mixed_text_to_qwen3_and_infers_zho(monkeypatch):
    client = TestClient(api_service.app)
    captured = {}

    async def fake_proxy(method, url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "status": "success",
            "language": "Chinese",
            "level": "token",
            "num_chars": 4,
            "alignments": [
                {"char": "今", "start": 0.1, "end": 0.2},
                {"char": "天", "start": 0.2, "end": 0.3},
                {"char": "Qwen3", "start": 0.3, "end": 0.5},
                {"char": "demo", "start": 0.5, "end": 0.7},
            ],
        }

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_proxy_json", fake_proxy)

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "今天Qwen3 demo开始",
        },
    )

    assert response.status_code == 200
    assert captured["url"] == "http://qwen3-backend/align"
    assert captured["payload"]["language"] == "Chinese"
    assert response.json()["backend"] == "qwen3"
    assert response.json()["language"] == "zho"


def test_gateway_auto_routes_abbreviation_to_qwen3(monkeypatch):
    client = TestClient(api_service.app)
    captured = {}

    async def fake_proxy(method, url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "status": "success",
            "language": "English",
            "level": "token",
            "num_chars": 3,
            "alignments": [
                {"char": "Mr", "start": 0.0, "end": 0.2},
                {"char": "Brown", "start": 0.2, "end": 0.5},
                {"char": "called", "start": 0.5, "end": 0.9},
            ],
        }

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_proxy_json", fake_proxy)

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "Mr. Brown called Dr. Smith before noon.",
            "language": "eng",
        },
    )

    assert response.status_code == 200
    assert captured["url"] == "http://qwen3-backend/align"
    assert captured["payload"]["language"] == "English"
    assert response.json()["backend"] == "qwen3"


def test_gateway_auto_preserves_word_level(monkeypatch):
    client = TestClient(api_service.app)
    captured = {}

    async def fake_proxy(method, url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "status": "success",
            "language": "English",
            "level": "word",
            "num_chars": 1,
            "alignments": [
                {"char": "Room", "start": 0.0, "end": 0.5},
            ],
        }

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_proxy_json", fake_proxy)

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "Room 301",
            "word_level": True,
        },
    )

    assert response.status_code == 200
    assert captured["url"] == "http://qwen3-backend/align"
    assert captured["payload"]["word_level"] is True
    assert response.json()["backend"] == "qwen3"
    assert response.json()["level"] == "word"


def test_gateway_auto_align_batch_routes_each_item_independently(monkeypatch):
    client = TestClient(api_service.app)
    captured = []

    async def fake_proxy(method, url, payload):
        captured.append({"url": url, "payload": payload})
        if url.startswith("http://qwen3-backend"):
            return {
                "status": "success",
                "language": "English",
                "level": "token",
                "num_chars": 1,
                "alignments": [{"char": "Room301", "start": 0.0, "end": 0.5}],
            }
        return {
            "status": "success",
            "language": payload.get("language", "eng"),
            "level": "char",
            "num_chars": 1,
            "alignments": [{"char": "hello", "start": 0.0, "end": 0.5}],
        }

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_proxy_json", fake_proxy)

    response = client.post(
        "/align_batch",
        json={
            "items": [
                {"audio": "dummy.wav", "text": "Hello world", "language": "eng"},
                {"audio": "dummy.wav", "text": "Room 301 is ready"},
            ]
        },
    )

    assert response.status_code == 200
    assert captured == [
        {
            "url": "http://qwen3-backend/align",
            "payload": {
                "audio": "dummy.wav",
                "text": "Hello world",
                "language": "English",
                "word_level": False,
                "separate_punctuation": False,
                "debug": False,
            },
        },
        {
            "url": "http://qwen3-backend/align",
            "payload": {
                "audio": "dummy.wav",
                "text": "Room 301 is ready",
                "language": "English",
                "word_level": False,
                "separate_punctuation": False,
                "debug": False,
            },
        },
    ]
    assert response.json()["results"][0]["backend"] == "qwen3"
    assert response.json()["results"][1]["backend"] == "qwen3"


def test_gateway_passes_queries_to_qwen3_backend(monkeypatch):
    client = TestClient(api_service.app)
    captured = {}

    async def fake_proxy(method, url, payload):
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = payload
        return {
            "status": "success",
            "backend": "qwen3",
            "language": "Chinese",
            "level": "token",
            "num_chars": 2,
            "alignments": [
                {"char": "你", "start": 0.1, "end": 0.3},
                {"char": "好", "start": 0.3, "end": 0.5},
            ],
            "answers": [{"start": 0.1, "end": 0.3}],
            "query_tokenization_matches_default": True,
        }

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_proxy_json", fake_proxy)

    response = client.post(
        "/align",
        json={
            "backend": "qwen3",
            "audio": "dummy.wav",
            "text": "你好",
            "language": "zho",
            "queries": ["<q>你</q>好"],
        },
    )

    assert response.status_code == 200
    assert captured == {
        "method": "POST",
        "url": "http://qwen3-backend/align",
        "payload": {
            "audio": "dummy.wav",
            "text": "你好",
            "language": "Chinese",
            "word_level": False,
            "separate_punctuation": False,
            "debug": False,
            "queries": ["<q>你</q>好"],
        },
    }
    data = response.json()
    assert data["backend"] == "qwen3"
    assert data["language"] == "zho"
    assert data["answers"] == [{"start": 0.1, "end": 0.3}]
    assert data["query_tokenization_matches_default"] is True


def test_gateway_qwen3_accepts_short_language_aliases(monkeypatch):
    client = TestClient(api_service.app)
    captured = []

    async def fake_proxy(method, url, payload):
        captured.append({"method": method, "url": url, "payload": payload})
        return {
            "status": "success",
            "backend": "qwen3",
            "language": payload["language"],
            "level": "token",
            "num_chars": 1,
            "alignments": [{"char": payload["text"], "start": 0.0, "end": 0.5}],
        }

    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    monkeypatch.setattr(api_service, "_proxy_json", fake_proxy)

    en_response = client.post(
        "/align",
        json={
            "backend": "qwen3",
            "audio": "dummy.wav",
            "text": "Hello",
            "language": "en",
        },
    )
    zh_response = client.post(
        "/align",
        json={
            "backend": "qwen3",
            "audio": "dummy.wav",
            "text": "你好",
            "language": "zh",
        },
    )

    assert en_response.status_code == 200
    assert zh_response.status_code == 200
    assert captured[0]["payload"]["language"] == "English"
    assert captured[1]["payload"]["language"] == "Chinese"
    assert en_response.json()["language"] == "eng"
    assert zh_response.json()["language"] == "zho"


def test_gateway_rejects_retired_backend():
    response = TestClient(api_service.app).post("/align", json={
        "backend": "ctc", "audio": "dummy.wav", "text": "hello", "language": "eng",
    })
    assert response.status_code == 400
    assert "Unknown backend" in response.json()["detail"]


def test_gateway_auto_requires_language_for_ambiguous_script(monkeypatch):
    monkeypatch.setattr(api_service, "QWEN3_URL", "http://qwen3-backend")
    response = TestClient(api_service.app).post("/align", json={
        "audio": "dummy.wav", "text": "こんにちは",
    })
    assert response.status_code == 400
    assert "language is required" in response.json()["detail"]
