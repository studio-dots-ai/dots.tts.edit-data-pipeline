import json

from speech_edit.clients.services import AlignmentClient


class _DummyResponse:
    def __init__(self, status_code=200, json_payload=None):
        self.status_code = status_code
        self._json_payload = json_payload or {}
        self.text = json.dumps(self._json_payload)

    def json(self):
        return self._json_payload


def test_align_returns_alignments_by_default(monkeypatch):
    captured = {}

    def _fake_post(url, json):
        captured["url"] = url
        captured["json"] = json
        return _DummyResponse(
            json_payload={
                "alignments": [{"char": "a", "start": 0.1, "end": 0.2}],
                "answers": [{"start": 0.1, "end": 0.2}],
            }
        )

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _fake_post)

    client = AlignmentClient(base_url="http://localhost:8029")
    result = client.align(b"RIFFfake", "hello", language="eng")

    assert captured["url"] == "http://localhost:8029/align"
    assert captured["json"]["backend"] == "auto"
    assert "queries" not in captured["json"]
    assert result == [{"char": "a", "start": 0.1, "end": 0.2}]


def test_align_returns_full_response_when_queries_are_requested(monkeypatch):
    captured = {}

    def _fake_post(url, json):
        captured["json"] = json
        return _DummyResponse(
            json_payload={
                "backend": "qwen3",
                "alignments": [{"char": "hello", "start": 0.1, "end": 0.5}],
                "answers": [{"start": 0.1, "end": 0.5}],
                "query_tokenization_matches_default": False,
            }
        )

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _fake_post)

    client = AlignmentClient(base_url="http://localhost:8029")
    result = client.align(
        b"RIFFfake",
        "hello",
        language="eng",
        backend="qwen3",
        queries=["<q>hello</q>"],
    )

    assert captured["json"]["queries"] == ["<q>hello</q>"]
    assert result["answers"] == [{"start": 0.1, "end": 0.5}]
    assert result["query_tokenization_matches_default"] is False


def test_align_can_return_full_response_without_queries(monkeypatch):
    def _fake_post(url, json):
        return _DummyResponse(
            json_payload={
                "backend": "ctc",
                "alignments": [{"char": "a", "start": 0.1, "end": 0.2}],
            }
        )

    monkeypatch.setattr("speech_edit.clients.services.requests.post", _fake_post)

    client = AlignmentClient(base_url="http://localhost:8029")
    result = client.align(b"RIFFfake", "hello", return_response=True)

    assert result["backend"] == "ctc"
    assert result["alignments"] == [{"char": "a", "start": 0.1, "end": 0.2}]
