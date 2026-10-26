from __future__ import annotations

import importlib.util
import sys

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _load_service_module():
    module_path = Path(__file__).resolve().parents[1] / "forced_alignment_service.py"
    spec = importlib.util.spec_from_file_location(
        "qwen3_forced_alignment_service", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service = _load_service_module()


def _make_result(units: list[str], times: list[tuple[float, float]]):
    return [
        SimpleNamespace(text=unit, start_time=start, end_time=end)
        for unit, (start, end) in zip(units, times)
    ]


class _EnglishProcessor:
    def tokenize_text(self, text: str, language: str) -> list[str]:
        mapping = {
            "But even so, it's nice": ["But", "even", "so", "it's", "nice"],
            "But": ["But"],
            "But ": ["But"],
            " even so, ": ["even", "so"],
            " even so, it's nice": ["even", "so", "it's", "nice"],
            "even so": ["even", "so"],
            ", it's nice": ["it's", "nice"],
            "it's nice": ["it's", "nice"],
            "But even so, it": ["But", "even", "so", "it"],
            "'": ["'"],
            "s nice": ["s", "nice"],
        }
        return list(mapping.get(text, []))

    def build_aligner_input_text(self, units: list[str]) -> str:
        return "|".join(units)


class _ChineseProcessor:
    def tokenize_text(self, text: str, language: str) -> list[str]:
        return list(text)


def test_align_endpoint_returns_answers_for_queries(monkeypatch):
    client = TestClient(service.app)
    captured = {}

    class FakeAligner:
        aligner_processor = _EnglishProcessor()

        def align(self, audio, text, language):
            raise AssertionError("query path should call align_with_units")

        def align_with_units(self, audio, units):
            captured["units"] = list(units)
            return _make_result(
                units,
                [(0.1, 0.2), (0.25, 0.45), (0.5, 0.6), (0.6, 0.85), (0.9, 1.2)],
            )

    monkeypatch.setattr(service, "_decode_audio", lambda audio: "dummy.wav")
    monkeypatch.setattr(service, "_validate_language", lambda language: "English")
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: FakeAligner())

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "But even so, it's nice",
            "language": "English",
            "queries": [
                "<q>But</q> even so, it's nice",
                "But even so, <q></q>it's nice",
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert captured["units"] == ["But", "even", "so", "it's", "nice"]
    assert data["answers"] == [
        {"start": 0.1, "end": 0.2},
        {"start": 0.6, "end": 0.6},
    ]
    assert data["query_tokenization_matches_default"] is True


def test_align_batch_queries_uses_single_batched_forward(monkeypatch):
    client = TestClient(service.app)
    captured = {}

    class FakeAligner:
        aligner_processor = _EnglishProcessor()

        def _align_prepared(self, audios, word_lists, aligner_input_texts):
            captured["batch_size"] = len(audios)
            captured["word_lists"] = [list(words) for words in word_lists]
            return [
                _make_result(
                    word_lists[0],
                    [
                        (0.1, 0.2),
                        (0.25, 0.45),
                        (0.5, 0.6),
                        (0.6, 0.85),
                        (0.9, 1.2),
                    ],
                ),
                _make_result(
                    word_lists[1],
                    [
                        (1.1, 1.2),
                        (1.25, 1.45),
                        (1.5, 1.6),
                        (1.6, 1.85),
                        (1.9, 2.2),
                    ],
                ),
            ]

    monkeypatch.setattr(service, "_decode_audio", lambda audio: "dummy.wav")
    monkeypatch.setattr(service, "normalize_audios", lambda audios: list(audios))
    monkeypatch.setattr(service, "_validate_language", lambda language: "English")
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: FakeAligner())

    response = client.post(
        "/align_batch",
        json={
            "items": [
                {
                    "audio": "dummy.wav",
                    "text": "But even so, it's nice",
                    "language": "English",
                    "queries": ["<q>But</q> even so, it's nice"],
                },
                {
                    "audio": "dummy.wav",
                    "text": "But even so, it's nice",
                    "language": "English",
                    "queries": ["But <q>even so</q>, it's nice"],
                },
            ]
        },
    )

    assert response.status_code == 200
    assert captured["batch_size"] == 2
    data = response.json()
    assert data["results"][0]["answers"] == [{"start": 0.1, "end": 0.2}]
    assert data["results"][1]["answers"] == [{"start": 1.25, "end": 1.6}]


def test_align_endpoint_reports_false_when_query_boundaries_change_qwen_tokenization(
    monkeypatch,
):
    client = TestClient(service.app)
    captured = {}

    class FakeAligner:
        aligner_processor = _EnglishProcessor()

        def align(self, audio, text, language):
            raise AssertionError("query path should call align_with_units")

        def align_with_units(self, audio, units):
            captured["units"] = list(units)
            return _make_result(
                units,
                [
                    (0.1, 0.2),
                    (0.25, 0.45),
                    (0.5, 0.6),
                    (0.6, 0.72),
                    (0.72, 0.76),
                    (0.76, 0.84),
                    (0.9, 1.2),
                ],
            )

    monkeypatch.setattr(service, "_decode_audio", lambda audio: "dummy.wav")
    monkeypatch.setattr(service, "_validate_language", lambda language: "English")
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: FakeAligner())

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "But even so, it's nice",
            "language": "English",
            "queries": ["But even so, it<q>'</q>s nice"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert captured["units"] == ["But", "even", "so", "it", "'", "s", "nice"]
    assert data["query_tokenization_matches_default"] is False
    assert data["answers"] == [{"start": 0.72, "end": 0.76}]


def test_align_endpoint_rejects_queries_with_mismatched_plain_text(monkeypatch):
    client = TestClient(service.app)

    class FakeAligner:
        aligner_processor = _ChineseProcessor()

    monkeypatch.setattr(service, "_decode_audio", lambda audio: "dummy.wav")
    monkeypatch.setattr(service, "_validate_language", lambda language: "Chinese")
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: FakeAligner())

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "开发边界",
            "language": "Chinese",
            "queries": ["<q>开发</q>边界", "开发<q>约定</q>"],
        },
    )

    assert response.status_code == 400
    assert "same plain text" in response.json()["detail"]


def test_align_endpoint_omits_query_fields_for_regular_alignment(monkeypatch):
    client = TestClient(service.app)

    class FakeAligner:
        aligner_processor = _ChineseProcessor()

        def align(self, audio, text, language):
            return [_make_result(list(text), [(0.1, 0.2)] * len(text))]

    monkeypatch.setattr(service, "_decode_audio", lambda audio: "dummy.wav")
    monkeypatch.setattr(service, "_validate_language", lambda language: "Chinese")
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: FakeAligner())

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "开发",
            "language": "Chinese",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "answers" not in data
    assert "query_tokenization_matches_default" not in data


def test_align_endpoint_accepts_word_level_for_english(monkeypatch):
    client = TestClient(service.app)

    class FakeAligner:
        aligner_processor = _EnglishProcessor()

        def align(self, audio, text, language):
            return [
                _make_result(
                    ["But", "even", "so"],
                    [(0.1, 0.2), (0.25, 0.45), (0.5, 0.6)],
                )
            ]

    monkeypatch.setattr(service, "_decode_audio", lambda audio: "dummy.wav")
    monkeypatch.setattr(service, "_validate_language", lambda language: "English")
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: FakeAligner())

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "But even so",
            "language": "English",
            "word_level": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["level"] == "word"
    assert [item["char"] for item in data["alignments"]] == ["But", "even", "so"]


def test_align_endpoint_accepts_word_level_for_chinese_as_char_level(monkeypatch):
    client = TestClient(service.app)

    class FakeAligner:
        aligner_processor = _ChineseProcessor()

        def align(self, audio, text, language):
            return [_make_result(list(text), [(0.1, 0.2), (0.2, 0.3)])]

    monkeypatch.setattr(service, "_decode_audio", lambda audio: "dummy.wav")
    monkeypatch.setattr(service, "_validate_language", lambda language: "Chinese")
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: FakeAligner())

    response = client.post(
        "/align",
        json={
            "audio": "dummy.wav",
            "text": "开发",
            "language": "Chinese",
            "word_level": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["level"] == "char"
    assert [item["char"] for item in data["alignments"]] == ["开", "发"]
