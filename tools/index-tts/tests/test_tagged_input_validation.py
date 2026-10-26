import importlib.util
import sys
from pathlib import Path


def _load_service_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "api_service.py"
    )
    spec = importlib.util.spec_from_file_location(
        "indextts_synthesis_service", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service = _load_service_module()


def test_parse_emotion_tags_distinguishes_none_from_untagged():
    segments = service.parse_emotion_tags(
        "<happy>Hello</happy><none> still</none><sad> world</sad>"
    )
    assert [segment.emotion for segment in segments] == ["happy", None, "sad"]

    segments = service.parse_emotion_tags("<happy>Hello</happy> still <sad>world</sad>")
    assert [segment.emotion for segment in segments] == ["happy", "sad"]
    assert service._has_untagged_emotion_text(
        "<happy>Hello</happy> still <sad>world</sad>"
    )
    assert not service._has_untagged_emotion_text(
        "<happy>Hello</happy><none> still</none><sad>world</sad>"
    )


def test_join_segment_texts_reuses_continuation_spacing_rule():
    class StubModel:
        def __init__(self):
            self.calls = []

        def _join_continuation_text(self, prefix_text: str, text: str) -> str:
            self.calls.append((prefix_text, text))
            prefix = prefix_text.strip()
            suffix = text.strip()
            if not prefix:
                return suffix
            if not suffix:
                return prefix
            return f"{prefix} <JOIN> {suffix}"

    helper_owner = StubModel()
    original_model = service.TTS_MODEL
    service.TTS_MODEL = helper_owner
    try:
        assert service._join_segment_texts(["Hello", "world", "again"]) == (
            "Hello <JOIN> world <JOIN> again"
        )
        assert helper_owner.calls == [
            ("", "Hello"),
            ("Hello", "world"),
            ("Hello <JOIN> world", "again"),
        ]
    finally:
        service.TTS_MODEL = original_model
